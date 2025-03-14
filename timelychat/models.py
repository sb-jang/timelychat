import json
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import pydantic
import torch
from openai import OpenAI
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from vllm import LLM, SamplingParams

from timelychat.prompts import (
    CoTOutput,
    Output,
    cot_response_ex1,
    cot_response_ex2,
    cot_time_ex1,
    cot_time_ex2,
    get_instruction,
)


class BaseModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def make_prompt(
        self, task: str, example: Dict[str, Union[str, List[str]]]
    ) -> Tuple[str, str, Optional[pydantic.BaseModel]]:
        raise NotImplementedError

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


def get_model(
    model_type: str, model_name: str, fewshot_examples: List[Dict[str, Union[str, List[str]]]], **kwargs
) -> BaseModel:
    model_classes = {
        "vllm": VLLMModel,
        "openai": OpenAIModel,
        "hf": HfModel,
    }
    return model_classes[model_type](model_name, fewshot_examples=fewshot_examples, **kwargs)


class VLLMModel(BaseModel):
    def __init__(self, model_name: str, fewshot_examples: List[Dict[str, Union[str, List[str]]]], **kwargs):
        super().__init__(model_name)
        self.fewshot_examples = self._make_fewshot_examples(fewshot_examples)
        self.cot_time_examples = [cot_time_ex1, cot_time_ex2]
        self.cot_response_examples = [cot_response_ex1, cot_response_ex2]
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=kwargs.get("num_gpus", 1),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.8),
        )

    def _make_fewshot_examples(
        self, fewshot_examples: List[Dict[str, Union[str, List[str]]]]
    ) -> Dict[str, List[Dict[str, str]]]:
        delayed, instant = [], []
        for ex in fewshot_examples:
            delayed_ex = {
                "context": "\n".join([f"{spk}: {utt}" for spk, utt in zip(ex["speaker_list"], ex["context"])]),
                "target_speaker": ex["target_speaker"],
                "time_elapsed": ex["time_elapsed"],
                "response": ex["timely_response"],
            }
            instant_ex = {
                "context": "\n".join([f"{spk}: {utt}" for spk, utt in zip(ex["speaker_list"], ex["context"])]),
                "target_speaker": ex["target_speaker"],
                "time_elapsed": "0 minutes",
                "response": ex["untimely_response"],
            }
            delayed.append(delayed_ex)
            instant.append(instant_ex)
        return {"delayed": delayed, "instant": instant}

    def make_prompt(
        self, task: str, example: Dict[str, Union[str, List[str]]], **kwargs
    ) -> Tuple[str, str, Optional[pydantic.BaseModel]]:
        history = "\n".join([f"{spk}: {utt}" for spk, utt in zip(example["speaker_list"], example["context"])])
        sampled_exs = [random.choice(self.fewshot_examples["delayed"]), random.choice(self.fewshot_examples["instant"])]
        random.shuffle(sampled_exs)
        cot_exs = (
            random.sample(self.cot_time_examples, 2) if task == "time" else random.sample(self.cot_response_examples, 2)
        )

        system_prompt, user_prompt = get_instruction(task=task, icl_method=kwargs.get("icl_method", "zeroshot"))
        return (
            system_prompt.format(output_format=""),
            user_prompt.format(
                context=history,
                target_speaker=example["target_speaker"],
                time_elapsed=example["time_elapsed"],
                ex1_context=sampled_exs[0]["context"],
                ex1_target_speaker=sampled_exs[0]["target_speaker"],
                ex1_time_elapsed=sampled_exs[0]["time_elapsed"],
                ex1_response=sampled_exs[0]["response"],
                ex2_context=sampled_exs[1]["context"],
                ex2_target_speaker=sampled_exs[1]["target_speaker"],
                ex2_time_elapsed=sampled_exs[1]["time_elapsed"],
                ex2_response=sampled_exs[1]["response"],
                cot_ex1=cot_exs[0],
                cot_ex2=cot_exs[1],
            ),
            None,
        )

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 0.95),
            skip_special_tokens=True,
            max_tokens=kwargs.get("max_new_tokens", 100),
        )
        prompt = system_prompt + "\n\n" + user_prompt
        outputs = self.model.generate(prompt, sampling_params)
        generated_text = outputs[0].outputs[0].text
        return generated_text


class OpenAIModel(BaseModel):
    def __init__(self, model_name: str, fewshot_examples: List[Dict[str, Union[str, List[str]]]], **kwargs):
        super().__init__(model_name)
        self.fewshot_examples = self._make_fewshot_examples(fewshot_examples)
        self.cot_time_examples = [cot_time_ex1, cot_time_ex2]
        self.cot_response_examples = [cot_response_ex1, cot_response_ex2]
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _make_fewshot_examples(
        self, fewshot_examples: List[Dict[str, Union[str, List[str]]]]
    ) -> Dict[str, List[Dict[str, str]]]:
        delayed, instant = [], []
        for ex in fewshot_examples:
            delayed_ex = {
                "context": "\n".join([f"{spk}: {utt}" for spk, utt in zip(ex["speaker_list"], ex["context"])]),
                "target_speaker": ex["target_speaker"],
                "time_elapsed": ex["time_elapsed"],
                "response": ex["timely_response"],
            }
            instant_ex = {
                "context": "\n".join([f"{spk}: {utt}" for spk, utt in zip(ex["speaker_list"], ex["context"])]),
                "target_speaker": ex["target_speaker"],
                "time_elapsed": "0 minutes",
                "response": ex["untimely_response"],
            }
            delayed.append(delayed_ex)
            instant.append(instant_ex)
        return {"delayed": delayed, "instant": instant}

    def make_prompt(
        self, task: str, example: Dict[str, Union[str, List[str]]], **kwargs
    ) -> Tuple[str, str, Optional[pydantic.BaseModel]]:
        history = "\n".join([f"{spk}: {utt}" for spk, utt in zip(example["speaker_list"], example["context"])])
        sampled_exs = [random.choice(self.fewshot_examples["delayed"]), random.choice(self.fewshot_examples["instant"])]
        random.shuffle(sampled_exs)
        cot_exs = (
            random.sample(self.cot_time_examples, 2) if task == "time" else random.sample(self.cot_response_examples, 2)
        )

        system_prompt, user_prompt = get_instruction(task=task, icl_method=kwargs.get("icl_method", "zeroshot"))
        if self.model_name.startswith("gpt-4o"):
            system_prompt = system_prompt.format(output_format="")
            response_format = CoTOutput if kwargs.get("icl_method", "zeroshot") == "cot" else Output
        else:
            reasoning_str = "\n    reasoning: str" if kwargs.get("icl_method", "zeroshot") == "cot" else ""
            output_str = (
                "answer: str // digit + unit (e.g., 5 minutes)" if task == "time" else "answer: str // response"
            )
            system_prompt = system_prompt.format(
                output_format=f"\n\nYou will output a json object containing the following information:\n{{{reasoning_str}\n    {output_str}\n}}"
            )
            response_format = None

        user_prompt = user_prompt.format(
            context=history,
            target_speaker=example["target_speaker"],
            time_elapsed=example["time_elapsed"],
            ex1_context=sampled_exs[0]["context"],
            ex1_target_speaker=sampled_exs[0]["target_speaker"],
            ex1_time_elapsed=sampled_exs[0]["time_elapsed"],
            ex1_response=sampled_exs[0]["response"],
            ex2_context=sampled_exs[1]["context"],
            ex2_target_speaker=sampled_exs[1]["target_speaker"],
            ex2_time_elapsed=sampled_exs[1]["time_elapsed"],
            ex2_response=sampled_exs[1]["response"],
            cot_ex1=cot_exs[0],
            cot_ex2=cot_exs[1],
        )
        return system_prompt, user_prompt, response_format

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        max_retries = kwargs.get("max_retries", 3)
        for attempt in range(max_retries):
            try:
                if self.model_name.startswith("gpt-4o"):
                    completion = self.client.beta.chat.completions.parse(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        response_format=kwargs.get("response_format", None),
                        temperature=kwargs.get("temperature", 1.0),
                        top_p=kwargs.get("top_p", 0.95),
                    )
                    parsed = completion.choices[0].message.parsed
                    return parsed.answer
                else:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        response_format={"type": "json_object"},
                        temperature=kwargs.get("temperature", 1.0),
                        top_p=kwargs.get("top_p", 0.95),
                    )
                    parsed = json.loads(completion.choices[0].message.content)
                    return parsed["answer"]
            except (json.JSONDecodeError, pydantic.ValidationError) as e:
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts. Error: {str(e)}")
                    return "Error: Invalid response format"
                print(f"Attempt {attempt + 1} failed. Retrying...")
                continue


class HfModel(BaseModel):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def make_prompt(self, task: str, example: Dict[str, Union[str, List[str]]], **kwargs) -> str:
        prompt = f"<spk> {example['speaker_list'][0]}: <utt> {example['context'][0]} "
        prompt += " ".join(
            [
                f"<spk> {spk}: <time> 0 minutes later <utt> {utt}"
                for spk, utt in zip(example["speaker_list"][1:], example["context"][1:])
            ]
        )
        prompt += f" <spk> {example['target_speaker']}: <time>"
        if task == "response":
            prompt += f" {example['time_elapsed']} later <utt>"
        return "", prompt, None

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        prompt = system_prompt + user_prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 100),
                do_sample=True,
                temperature=kwargs.get("temperature", 1.0),
                top_p=kwargs.get("top_p", 0.95),
                num_beams=kwargs.get("num_beams", 3),
                no_repeat_ngram_size=kwargs.get("no_repeat_ngram_size", 2),
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
