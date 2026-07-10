import argparse
import contextlib
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, ContextManager, List, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm

from timelychat.models import BaseModel, get_model
from timelychat.prompts import Output
from utils.postprocess import convert_to_minutes, format_minutes, postprocess_response

simulator_prompt = """You are a user simulator (user) engaging in an event-driven dialogue with a dialogue agent (agent).
Given the dialogue context, your task is to proceed the conversation by one turn under the following assumptions:
1. agent responds after the elapsed time specified in the parentheses from the previous user utterance. If the delay is "0 minutes", agent is assumed to respond immediately.
2. user is assumed to respond to agent without any delay."""


class SimulatorRefusal(Exception):
    """The simulator declined to continue this dialogue.

    Distinct from a transient null parse: a refusal is deterministic for a given
    conversation, so retrying cannot succeed. The dialog is dropped and later
    removed from every arm by analysis/align_subsample.py, keeping the subsample
    matched.
    """


def make_simulator(simulator: str, max_iter: int = 3) -> Callable[[str, str], str]:
    """Return call(system_prompt, user_prompt) -> answer for the simulator backend.

    The backend is chosen from the model name, so --simulator swaps the provider,
    not just the model string.

    Both providers occasionally return a null parse (refusal, truncation), which
    surfaces as AttributeError on `.answer`. laaj.py:38 retries on exactly this;
    do the same here. A simulator turn cannot be skipped -- dropping a dialog would
    break the matched subsample across arms -- so exhausting the retries raises.
    """

    def with_retries(parse_once: Callable[[str, str], str], system_prompt: str, user_prompt: str) -> str:
        for attempt in range(max_iter):
            try:
                return parse_once(system_prompt, user_prompt)
            except SimulatorRefusal:
                raise  # deterministic for this conversation; retrying cannot help
            except AttributeError:
                if attempt == max_iter - 1:
                    raise
        raise AssertionError("unreachable")

    if simulator.startswith("gpt-"):
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        def parse_once(system_prompt: str, user_prompt: str) -> str:
            completion = client.beta.chat.completions.parse(
                model=simulator,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                response_format=Output,
            )
            message = completion.choices[0].message
            if getattr(message, "refusal", None):
                raise SimulatorRefusal(f"{simulator} refused: {message.refusal}")
            return message.parsed.answer

    elif simulator.startswith("claude-"):
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        def parse_once(system_prompt: str, user_prompt: str) -> str:
            response = client.beta.messages.parse(
                model=simulator,
                max_tokens=1024,
                betas=["structured-outputs-2025-11-13"],
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                output_format=Output,
            )
            if response.parsed_output is None and response.stop_reason == "refusal":
                raise SimulatorRefusal(f"{simulator} refused (stop_reason=refusal)")
            return response.parsed_output.answer

    else:
        raise ValueError(f"Unsupported simulator: {simulator!r}. Expected a gpt-* or claude-* model name.")

    return lambda sp, up: with_retries(parse_once, sp, up)


def simulate_conversation(
    agent: BaseModel,
    simulate: Callable[[str, str], str],
    context: List[str],
    speaker_list: List[str],
    time_elapsed: List[str],
    target_speaker: str,
    num_turns: int = 10,
    agent_lock: ContextManager = contextlib.nullcontext(),
    **gen_kwargs,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    agent_speaker = target_speaker
    user_speaker = "A" if agent_speaker == "B" else "B"
    speakers_for_simulator = ["user" if spk == user_speaker else "agent" for spk in speaker_list]
    # convert_to_minutes() returns 0.0 for unparseable text (utils/postprocess.py:119),
    # which is indistinguishable from a genuine "0 minutes". Keep the raw strings so a
    # run can be audited for silent parse failures.
    raw_time_outputs = []

    for _ in range(num_turns):
        if speaker_list[-1] == agent_speaker:
            # user turn
            history = "\n".join(
                [
                    f"{spk}: ({time} later) {utt}"
                    for spk, time, utt in zip(speakers_for_simulator, time_elapsed, context)
                ]
            )
            user_prompt = f"Conversation:\n{history}"
            simulator_response = simulate(simulator_prompt, user_prompt)
            simulator_response = simulator_response.split("later)")[-1].strip()
            simulator_response = simulator_response.split("user:")[-1].strip()
            simulator_response = simulator_response.split("agent:")[-1].strip()
            context.append(simulator_response)
            speaker_list.append(user_speaker)
            time_elapsed.append("0 minutes")
            speakers_for_simulator.append("user")
        else:
            # agent turn
            example = {
                "context": context,
                "speaker_list": speaker_list,
                "target_speaker": agent_speaker,
                "time_elapsed": "0 minutes",  # temporary for time interval prediction
                # The agent creates its own delays; prompted agents must see them in the
                # history. HfModel ignores this and renders every context turn as
                # "0 minutes later", matching TIMER's training format.
                "time_elapseds": time_elapsed,
            }
            # Generate time interval first
            system_prompt, user_prompt, response_format = agent.make_prompt(task="time", example=example)
            with agent_lock:
                output = agent.generate(system_prompt, user_prompt, response_format=response_format, **gen_kwargs)
            raw_time_outputs.append(output)
            time_interval = format_minutes(convert_to_minutes(output))
            # Generate response
            example.update({"time_elapsed": time_interval})
            system_prompt, user_prompt, response_format = agent.make_prompt(task="response", example=example)
            with agent_lock:
                output = agent.generate(system_prompt, user_prompt, response_format=response_format, **gen_kwargs)
            agent_response = postprocess_response(output)
            context.append(agent_response)
            speaker_list.append(agent_speaker)
            time_elapsed.append(time_interval)
            speakers_for_simulator.append("agent")
    return context, speakers_for_simulator, time_elapsed, raw_time_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, required=True, choices=["vllm", "openai", "anthropic", "hf"])
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--simulator", type=str, default="gpt-4o")
    parser.add_argument("--num-turns", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--num-dialogs", type=int, default=0, help="subsample size; 0 = use the full eval split")
    parser.add_argument("--seed", type=int, default=0, help="subsample seed; fixed across simulators for a matched comparison")
    parser.add_argument("--workers", type=int, default=8, help="dialogs simulated concurrently")
    args = parser.parse_args()

    torch.multiprocessing.set_start_method("spawn")

    # Load data
    data = load_dataset("anonymous17711771/timelychat", split="eval")
    idxs = list(range(len(data)))
    if args.num_dialogs:
        random.Random(args.seed).shuffle(idxs)
        idxs = sorted(idxs[: args.num_dialogs])  # same subset across simulators (seed-controlled)

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
    agent = get_model(args.model_type, args.model_name, fewshot_examples=data, **model_config)
    simulate = make_simulator(args.simulator)

    # A local agent holds one copy of the weights, so its generate() is serialized;
    # only the simulator's API calls overlap across threads.
    agent_lock = threading.Lock() if args.model_type in ("hf", "vllm") else contextlib.nullcontext()

    # Generate
    refused = []

    def run_dialog(idx: int) -> dict:
        example = data[idx]
        try:
            return _run_dialog(idx, example)
        except SimulatorRefusal as e:
            # Keep the run alive; analysis/align_subsample.py drops this idx from every arm.
            print(f"\n[refusal] dialog idx={idx} dropped: {e}")
            refused.append(idx)
            return None

    def _run_dialog(idx: int, example) -> dict:
        context, alt_speaker_list, time_elapseds, raw_time_outputs = simulate_conversation(
            agent=agent,
            simulate=simulate,
            context=[example["context"][0]],
            speaker_list=[example["speaker_list"][0]],
            time_elapsed=["0 minutes"],
            target_speaker=example["target_speaker"],
            num_turns=args.num_turns,
            agent_lock=agent_lock,
            **gen_kwargs,
        )
        return {
            "idx": idx,
            "context": context,
            "alt_speaker_list": alt_speaker_list,
            "time_elapseds": time_elapseds,
            "raw_time_outputs": raw_time_outputs,  # audit trail; ignored by laaj.py
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = [r for r in tqdm(pool.map(run_dialog, idxs), total=len(idxs)) if r is not None]
    results.sort(key=lambda r: r["idx"])
    if refused:
        print(f"{len(refused)} dialog(s) refused by the simulator and dropped: {sorted(refused)}")

    # Save results
    os.makedirs("results", exist_ok=True)
    save_path = f"results/dialog-level_{args.model_name.replace('/', '--')}_{args.simulator}_T{args.num_turns}_n{len(idxs)}_seed{args.seed}.jsonl"
    print(f"Saving results to {save_path}...")
    with open(save_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
