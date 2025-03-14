import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from tqdm import tqdm

from timelychat.models import get_model
from utils.metrics import bertscore, bleu, rmsle, rouge
from utils.postprocess import get_postprocess_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, required=True, choices=["vllm", "openai", "hf"])
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["time", "response"])
    parser.add_argument("--icl-method", type=str, default="zeroshot", choices=["zeroshot", "fewshot", "cot"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--save-results", action="store_true")
    args = parser.parse_args()

    torch.multiprocessing.set_start_method("spawn")

    # Load data
    data = load_dataset("anonymous17711771/timelychat", split="eval")

    # Load model
    model_config = {
        "num_gpus": args.num_gpus,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    gen_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }
    model = get_model(args.model_type, args.model_name, fewshot_examples=data, **model_config)

    # Generate
    results = []
    for example in tqdm(data):
        system_prompt, user_prompt, response_format = model.make_prompt(task=args.task, example=example, **gen_kwargs)
        output = model.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=response_format,
            **gen_kwargs,
        )
        results.append(output)

    # Postprocess
    postprocess_fn = get_postprocess_fn(args.task, args.icl_method)
    results = [postprocess_fn(result) for result in results]

    # Save results
    if args.save_results:
        dics = []
        for example, result in zip(data, results):
            dics.append(
                {
                    "generated": result,
                    "context": example["context"],
                    "speaker_list": example["speaker_list"],
                    "target_speaker": example["target_speaker"],
                    "time_elapsed": example["time_elapsed"],
                    "timely_response": example["timely_response"],
                }
            )

        os.makedirs("results", exist_ok=True)
        save_path = f"results/turn-level_{args.model_name.replace('/', '--')}_{args.task}_{args.icl_method}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        print(f"Saving results to {save_path}...")
        with open(save_path, "w") as f:
            for dic in dics:
                f.write(json.dumps(dic) + "\n")

    # Calculate metrics
    metrics = {}
    if args.task == "time":
        y_true = [postprocess_fn(example["time_elapsed"]) for example in data]
        metrics["RMSLE"] = rmsle(y_true=y_true, y_pred=results)
    elif args.task == "response":
        y_true = [example["timely_response"] for example in data]
        metrics["BLEU-2"] = bleu(refs=y_true, preds=results)
        metrics["ROUGE-L"] = rouge(refs=y_true, preds=results)
        metrics["BERTScore"] = bertscore(refs=y_true, preds=results)

    # Print metrics
    print("========== Evaluation Settings ==========")
    print(f"Model: {args.model_name}")
    print(f"Task: {args.task}")
    print(f"ICL method: {args.icl_method}")
    print("========== Results ==========")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")


if __name__ == "__main__":
    main()
