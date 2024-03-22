from llmlingua import PromptCompressor
import json
from tqdm import tqdm
import argparse
from pathlib import Path


def load_data(
    data_path=None,
    global_rank=-1,
    world_size=-1,
    question_prefix='question:',
    title_prefix='title:',
    passage_prefix='context:',
    n_ctx=100
):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []
    for k, example in enumerate(data):
        if global_rank > -1 and not k % world_size == global_rank:
            continue
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = k
        f = title_prefix + " {} " + passage_prefix + " {}"
        contexts = example['ctxs'][:n_ctx]
        passages = [f.format(c['title'], c['text']) for c in contexts]
        for c in example['ctxs']:
            if not 'score' in c:
                c['score'] = 1.0 / (k + 1)
        example['passages'] = passages
        examples.append(example)
    # egrave: is this needed?
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples


parser = argparse.ArgumentParser()
parser.add_argument("--l", type=int, default=0)
parser.add_argument("--q", type=int, default=0)
parser.add_argument("--n", type=int, default=500)
parser.add_argument("--n_ctx", type=int, default=100)
parser.add_argument("--target_token", type=int, default=64)
parser.add_argument("--dataset", type=str, default='NQ')
parser.add_argument("--d", type=str, default='test')

opt = parser.parse_args()
if opt.q == 0:
    opt.q = opt.n
target_token = opt.target_token
checkpoint_path = Path(
    f"compress_data/{opt.dataset}/{opt.d}/n_ctx-{opt.n_ctx}_{target_token}")
checkpoint_path.mkdir(parents=True, exist_ok=True)
examples = load_data(
    data_path=f'open_domain_data/{opt.dataset}/{opt.d}.json', n_ctx=opt.n_ctx)

llm_lingua = PromptCompressor(
    device_map='auto')
prompts = []
start_l = opt.l * opt.n
end_l = start_l + opt.q

examples = examples[start_l:end_l]
for ex in tqdm(examples, desc='Length'):
    prompt = dict()
    compressed_prompt = llm_lingua.compress_prompt(
        ex['passages'],
        instruction='Please write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant). Only give me the answer and do not output any other words.',
        question=ex['question'],
        target_token=target_token,
        condition_compare=True,
        condition_in_question='after',
        rank_method='longllmlingua',
        use_sentence_level_filter=False,
        context_budget="+100",
        # enable dynamic_context_compression_ratio
        dynamic_context_compression_ratio=0.4,
        reorder_context="sort",
    )
    prompt['compressed_prompt'] = compressed_prompt
    # prompt['question'] = ex['question']
    # prompt['answers'] = ex['answers']
    prompt['id'] = ex['id']
    # prompt['ctxs'] = ex['ctxs']
    # if 'target' in ex.keys():
    #     prompt['target'] = ex['target']
    prompts.append(prompt)
with open(f"{checkpoint_path}/{opt.d}_{opt.l}.json", "w") as f:
    json.dump(prompts, f, indent=4)
# > {'compressed_prompt': 'Question: Sam bought a dozen boxes, each with 30 highlighter pens inside, for $10 each box. He reanged five of boxes into packages of sixlters each and sold them $3 per.
# He sold the rest theters separately at the of three pens $2. How much did make in total, dollars?\nLets think step step\nSam bought 1 boxes x00 oflters.\nHe bought 12 * 300ters in total\nSam then took 5 boxes 6ters0ters.\n
# He sold these boxes for 5 *5\nAfterelling these  boxes there were 3030 highlighters remaining.\nThese form 330 / 3 = 110 groups of three pens.\nHe sold each of these groups for $2 each, so made 110 * 2 = $220 from them.\n
# In total, then, he earned $220 + $15 = $235.\nSince his original cost was $120, he earned $235 - $120 = $115 in profit.\nThe answer is 115',
#  'origin_tokens': 2365,
#  'compressed_tokens': 211,
#  'ratio': '11.2x',
#  'saving': ', Saving $0.1 in GPT-4.'}
