import argparse
import json
import os
import time
from typing import List, Tuple

import torch
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

from timelychat.models import BaseModel, get_model
from timelychat.prompts import Output
from utils.postprocess import convert_to_minutes, postprocess_response

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

simulator_prompt = """You are a user simulator (user) engaging in an event-driven dialogue with a dialogue agent (agent).
Given the dialogue context, your task is to proceed the conversation by one turn under the following assumptions:
1. agent responds after the elapsed time specified in the parentheses from the previous user utterance. If the delay is "0 minutes", agent is assumed to respond immediately.
2. user is assumed to respond to agent without any delay."""


def simulate_conversation(
    agent: BaseModel,
    simulator: str,
    context: List[str],
    speaker_list: List[str],
    time_elapsed: List[str],
    target_speaker: str,
    num_turns: int = 10,
    **gen_kwargs,
) -> Tuple[List[str], List[str], List[str]]:
    agent_speaker = target_speaker
    user_speaker = "A" if agent_speaker == "B" else "B"
    speakers_for_simulator = ["user" if spk == user_speaker else "agent" for spk in speaker_list]

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
            completion = client.beta.chat.completions.parse(
                model=simulator,
                messages=[{"role": "system", "content": simulator_prompt}, {"role": "user", "content": user_prompt}],
                response_format=Output,
            )
            parsed = completion.choices[0].message.parsed
            simulator_response = parsed.answer
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
            }
            # Generate time interval first
            system_prompt, user_prompt, response_format = agent.make_prompt(task="time", example=example)
            output = agent.generate(system_prompt, user_prompt, response_format=response_format, **gen_kwargs)
            time_interval = f"{int(convert_to_minutes(output))} minutes"
            # Generate response
            example.update({"time_elapsed": time_interval})
            system_prompt, user_prompt, response_format = agent.make_prompt(task="response", example=example)
            output = agent.generate(system_prompt, user_prompt, response_format=response_format, **gen_kwargs)
            agent_response = postprocess_response(output)
            context.append(agent_response)
            speaker_list.append(agent_speaker)
            time_elapsed.append(time_interval)
            speakers_for_simulator.append("agent")
    return context, speakers_for_simulator, time_elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, required=True, choices=["vllm", "openai", "hf"])
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
    agent = get_model(args.model_type, args.model_name, fewshot_examples=data, **model_config)

    # Generate
    results = []
    for example in tqdm(data):
        context, speaker_list, time_elapsed = simulate_conversation(
            agent=agent,
            simulator=args.simulator,
            context=[example["context"][0]],
            speaker_list=[example["speaker_list"][0]],
            time_elapsed=["0 minutes"],
            target_speaker=example["target_speaker"],
            num_turns=args.num_turns,
            **gen_kwargs,
        )

        results.append(
            {
                "context": context,
                "speaker_list": speaker_list,
                "time_elapsed": time_elapsed,
            }
        )

    # Save results
    os.makedirs("results", exist_ok=True)
    save_path = f"results/dialog-level_{args.model_name.replace('/', '--')}_{args.simulator}_T{args.num_turns}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"Saving results to {save_path}...")
    with open(save_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
