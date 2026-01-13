import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from tqdm import tqdm

from timelychat.models import get_model
from utils.metrics import bertscore, bleu, f1_score, fpr, precision, recall, rmsle, rouge
from utils.postprocess import get_postprocess_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, required=True, choices=["vllm", "openai", "anthropic", "hf"])
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
    parser.add_argument("--do-incremental", action="store_true", help="Process examples incrementally")
    parser.add_argument("--no-special-tokens", action="store_true", help="Use no special tokens")
    parser.add_argument("--utterance-first", action="store_true", help="Use utterance first")
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
    if args.no_special_tokens:
        gen_kwargs["no_special_tokens"] = True
    if args.utterance_first:
        gen_kwargs["utterance_first"] = True

    model = get_model(args.model_type, args.model_name, fewshot_examples=data, **model_config)

    # Generate
    results = []
    all_examples = []
    all_y_true = []

    if args.do_incremental:
        # Incremental processing
        for example in tqdm(data):
            context = example["context"]
            speaker_list = example["speaker_list"]
            context_len = len(context)

            # Process each example incrementally
            for i in range(context_len):
                # Incremental slicing of the context
                incremental_context = context[: i + 1]
                incremental_speaker_list = speaker_list[: i + 1]

                # Check if it's the last iteration
                is_last = i == context_len - 1

                # Determine timely_response and target_speaker
                if is_last:
                    # Last iteration: use original timely_response and target_speaker
                    incremental_timely_response = example["timely_response"]
                    incremental_time_elapsed = example["time_elapsed"]
                    incremental_target_speaker = example["target_speaker"]
                else:
                    # Not the last iteration: next utterance is timely_response, next utterance's speaker is target_speaker
                    incremental_timely_response = context[i + 1]
                    incremental_time_elapsed = "0 minutes"
                    incremental_target_speaker = speaker_list[i + 1]

                # Create incremental example
                incremental_example = {
                    "context": incremental_context,
                    "speaker_list": incremental_speaker_list,
                    "target_speaker": incremental_target_speaker,
                    "time_elapsed": incremental_time_elapsed,
                    "timely_response": incremental_timely_response,
                }

                # Generate
                system_prompt, user_prompt, response_format = model.make_prompt(
                    task=args.task, example=incremental_example, **gen_kwargs
                )
                output = model.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_format=response_format,
                    **gen_kwargs,
                )
                results.append(output)
                all_examples.append(incremental_example)

                # Save y_true
                if args.task == "time":
                    all_y_true.append(incremental_time_elapsed)
                elif args.task == "response":
                    all_y_true.append(incremental_timely_response)
    else:
        # Original processing (non-incremental)
        for example in tqdm(data):
            system_prompt, user_prompt, response_format = model.make_prompt(
                task=args.task, example=example, **gen_kwargs
            )
            output = model.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_format=response_format,
                **gen_kwargs,
            )
            results.append(output)
            all_examples.append(example)

            # Save y_true
            if args.task == "time":
                all_y_true.append(example["time_elapsed"])
            elif args.task == "response":
                all_y_true.append(example["timely_response"])

    # Postprocess
    postprocess_fn = get_postprocess_fn(args.task, args.icl_method)
    results = [postprocess_fn(result) for result in results]

    # Save results
    if args.save_results:
        dics = []
        for example, result in zip(all_examples, results):
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
        incremental_suffix = "_incremental" if args.do_incremental else ""
        save_path = f"results/turn-level_{args.model_name.replace('/', '--')}_{args.task}_{args.icl_method}{incremental_suffix}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        print(f"Saving results to {save_path}...")
        with open(save_path, "w") as f:
            for dic in dics:
                f.write(json.dumps(dic) + "\n")

    # Calculate metrics
    metrics = {}
    if args.task == "time":
        if args.do_incremental:
            y_true = [postprocess_fn(time_elapsed) for time_elapsed in all_y_true]
            metrics["Precision"] = precision(y_true=y_true, y_pred=results)
            metrics["Recall"] = recall(y_true=y_true, y_pred=results)
            metrics["FPR"] = fpr(y_true=y_true, y_pred=results)
            metrics["F1"] = f1_score(y_true=y_true, y_pred=results)
        else:
            y_true = [postprocess_fn(example["time_elapsed"]) for example in data]
        metrics["RMSLE"] = rmsle(y_true=y_true, y_pred=results)
    elif args.task == "response":
        if args.do_incremental:
            y_true = all_y_true
        else:
            y_true = [example["timely_response"] for example in data]
        metrics["BLEU-2"] = bleu(refs=y_true, preds=results)
        metrics["ROUGE-L"] = rouge(refs=y_true, preds=results)
        metrics["BERTScore"] = bertscore(refs=y_true, preds=results)

    # Print metrics
    print("========== Evaluation Settings ==========")
    print(f"Model: {args.model_name}")
    print(f"Task: {args.task}")
    print(f"ICL method: {args.icl_method}")
    print(f"Incremental: {args.do_incremental}")
    print("========== Results ==========")
    for metric, value in metrics.items():
        if metric == "RMSLE":
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value:.2f}")


if __name__ == "__main__":
    main()
