import argparse
import asyncio
import glob
import json
import os

from anthropic import AsyncAnthropic, DefaultAioHttpClient
from tqdm import tqdm

from timelychat.prompts import Evaluation, get_laaj_prompts, get_laaj_rubrics


async def evaluate_example(
    client: AsyncAnthropic,
    evaluator: str,
    idx: int,
    rubric: dict,
    system_prompt: str,
    user_prompt: str,
    semaphore: asyncio.Semaphore,
    max_iter: int = 3,
) -> dict:
    async with semaphore:
        last_error = None
        for attempt in range(max_iter):
            try:
                response = await client.beta.messages.parse(
                    model=evaluator,
                    max_tokens=1024,
                    betas=["structured-outputs-2025-11-13"],
                    messages=[{"role": "user", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    output_format=Evaluation,
                )
                output = response.parsed_output
                score = output.score
                explanation = output.explanation
                return {"idx": idx, "metric": rubric["metric"], "score": score, "explanation": explanation}
            except AttributeError as e:
                last_error = e
                if attempt < max_iter - 1:
                    continue
        if last_error is not None:
            print(f"Failed after {max_iter} attempts. Error: {last_error}")
            return {
                "idx": idx,
                "metric": rubric["metric"],
                "score": 1,
                "explanation": "Failed after 3 attempts. Error: AttributeError",
            }


async def main(args: argparse.Namespace) -> None:
    model_name = args.model_name.replace("/", "--")
    icl_method = f"{args.icl_method}" if args.icl_method else ""
    file_pattern = f"./results/{args.setting}_{model_name}_response_{icl_method}*.jsonl"
    matched_files = sorted(glob.glob(file_pattern))
    if not matched_files:
        raise FileNotFoundError(f"No files matched for pattern: {file_pattern}")
    file_path = matched_files[-1]

    with open(file_path) as f:
        data = [json.loads(line) for line in f]

    async with AsyncAnthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        http_client=DefaultAioHttpClient(),
    ) as client:
        semaphore = asyncio.Semaphore(100)

        responses = []
        rubrics = get_laaj_rubrics(args.setting)
        system_template, user_template = get_laaj_prompts(args.setting)
        for rubric in rubrics:
            tasks = []
            for idx, example in enumerate(data):
                system_prompt = system_template.format(
                    metric=rubric["metric"], criteria=rubric["criteria"], steps=rubric["steps"]
                )
                if args.setting == "turn-level":
                    context = "\n".join(
                        f"{spk}: {utt}" for spk, utt in zip(example["speaker_list"], example["context"])
                    )
                    model_response = f"{example['target_speaker']}: {example['generated']}"
                    user_prompt = user_template.format(
                        context=context, time_elapsed=example["time_elapsed"], response=model_response
                    )
                elif args.setting == "dialog-level":
                    context = ""
                    for spk, time, utt in zip(
                        example["alt_speaker_list"], example["time_elapseds"], example["context"]
                    ):
                        if time == "0 minutes":
                            context += f"{spk.capitalize()}: {utt}\n"
                        else:
                            context += f"{spk.capitalize()}: ({time} later) {utt}\n"
                    user_prompt = f"### Conversation ###\n{context.strip()}"
                else:
                    raise ValueError(f"Invalid setting: {args.setting}")

                tasks.append(
                    evaluate_example(
                        client,
                        args.evaluator,
                        idx,
                        rubric,
                        system_prompt,
                        user_prompt,
                        semaphore,
                        max_iter=args.max_iter,
                    )
                )

            rubric_responses = []
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Evaluating {rubric['metric']}"):
                result = await coro
                rubric_responses.append(result)

            rubric_responses.sort(key=lambda x: x["idx"])
            responses.extend(rubric_responses)

    with open(f"./results/{args.evaluator}-eval_{args.setting}_{model_name}_{args.icl_method}.jsonl", "w") as f:
        for response in responses:
            f.write(json.dumps(response) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, required=True, choices=["turn-level", "dialog-level"])
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--evaluator", type=str, default="claude-sonnet-4-5")
    parser.add_argument("--icl-method", type=str, default=None)
    parser.add_argument("--max-iter", type=int, default=3, help="Maximum number of retry attempts for AttributeError")
    args = parser.parse_args()
    asyncio.run(main(args))
