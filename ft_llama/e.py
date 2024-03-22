#!/usr/bin/env python3
"""Given a data file with LM QA predictions, evaluate the predictions.
"""
import argparse
import json
import logging
import os
import statistics
import sys
from copy import deepcopy

from tqdm import tqdm

from evaluation import ems, best_subspan_em
logger = logging.getLogger(__name__)

METRICS = [
    (best_subspan_em, "best_subspan_em"), (ems, "exact_match")
]


def main(
    path,
    all_files,
    output_path,
):
    metrics = dict()
    for filename in all_files:
        dataset = filename.split('.jsonl')[0]
        # if not filename.endswith("gz"):
        #     continue
        all_examples = []
        with open(f'{path}{filename}') as fin:
            for line in tqdm(fin):
                input_example = json.loads(line)
                all_examples.append(input_example)

        # Compute normal metrics in parallel, if applicable
        logger.info("Computing metrics")
        all_example_metrics = []
        for example in tqdm(all_examples):
            all_example_metrics.append(get_metrics_for_example(example))

        # Average metrics across examples
        for (_, metric_name) in METRICS:
            average_metric_value = statistics.mean(
                example_metrics[metric_name] for (example_metrics, _) in all_example_metrics
            )
            logger.info(f"{metric_name}: {average_metric_value}")
            if metric_name not in metrics.keys():
                metrics[metric_name] = dict()
            metrics[metric_name][dataset] = average_metric_value
        # if output_path:
        #     with xopen(output_path + '.gz', "w") as f, open(output_path, "w", encoding="utf-8") as fo:
        #         for (example_metrics, example) in all_example_metrics:
        #             example_with_metrics = deepcopy(example)
        #             for metric_name, metric_value in example_metrics.items():
        #                 example_with_metrics[f"metric_{metric_name}"] = metric_value
        #             f.write(json.dumps(example_with_metrics) + "\n")
        #             fo.write(json.dumps(example_with_metrics) + "\n")
        #         f.write(f"{metric_name}: {average_metric_value}" + "\n")
        #         fo.write(f"{metric_name}: {average_metric_value}" + "\n")
    # metrics = dict(sorted(metrics.items()))
    with open(output_path, "w", encoding="utf-8") as fo:
        for metric_name, metric_value in metrics.items():
            fo.write(f"{metric_name}:" + "\n")
            json.dump(dict(sorted(metric_value.items())), fo, ensure_ascii=False, indent=4)
            fo.write(f"\n")


def get_metrics_for_example(example):
    gold_answers = example["answer"]
    model_answer = example["pred"]

    # NOTE: we take everything up to the first newline, since otherwise models could hack
    # the metric by simply copying te input context (as the gold answer is guaranteed
    # to occur in the input context).
    model_answer = model_answer.split("\n")[0].strip()

    example_metrics = {}
    for (metric, metric_name) in METRICS:
        example_metrics[metric_name] = metric(prediction=model_answer, ground_truths=gold_answers)
    return (example_metrics, example)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    path = f"predictions/{args.model}/"
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    
    out_path = f"predictions/{args.model}/results.jsonl"
    
    logger.info("running %s", " ".join(sys.argv))
    main(
        path,
        all_files,
        out_path
    )
    logger.info("finished running %s", sys.argv[0])
