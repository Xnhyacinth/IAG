from datasets import load_dataset
import transformers
import datasets 
import torch
config = datasets.DownloadConfig(resume_download=True, max_retries=100) 

# dataset = datasets.load_from_disk(f'dataset/Image/NQ/test')
# print(dataset['compressed_ctxs_5'][0]['compressed_prompt'][204:])
# dataset = datasets.load_dataset( "codeparrot/self-instruct-starcoder", cache_dir="./hf_cache", download_config=config)
# data = load_dataset("Xnhyacinth/Image", 'WQ', download_config=config)
# print(data)
# data.save_to_disk('dataset/Image/WQ')
# model_name = 't5-3b'
# tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, local_files_only=False)
# model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name, resume_download=True, local_files_only=False)
# from huggingface_hub import snapshot_download
# import transformers
# # snapshot_download(repo_id='google/flan-t5-base',
# #                   repo_type='model',
# #                   local_dir='./models/flan-t5-base',
# #                   resume_download=True)
# a = transformers.AutoModelForSeq2SeqLM.from_pretrained(
#             't5-large', resume_download=True
#         )
# print(a.config.d_model)
a = transformers.AutoModelForCausalLM.from_pretrained(
    'models/llama2/7b',
    torch_dtype=torch.float16,
    device_map='auto'
)
print(a.model.layers)