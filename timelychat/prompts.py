from typing import Tuple

from pydantic import BaseModel


# response formats for structured output of GPT-4o
class Output(BaseModel):
    answer: str


class CoTOutput(BaseModel):
    reasoning: str
    answer: str


desc_time = """You are given a conversation between two speakers.
Your task is to estimate a time interval needed until the next response, considering the duration of the event in the conversation ranging from 0 minutes to 24 hours (1 day).
If the next response is expected to be immediate, you will output "0 minutes".
Otherwise, you will output a digit and a unit of time (e.g., 5 minutes, 2 hours).{output_format}"""

desc_response = """You are given a conversation between two speakers and the elapsed time since the last utterance.
Your task is to generate the next response that aligns well with the temporal context represented by the time interval in parentheses.{output_format}"""

# Zero-shot
inst_time_zeroshot = """### Dialogue context ###
{context}

Answer format: n (0<=n<=1440) minutes
The estimated time interval is:"""

inst_response_zeroshot = """### Dialogue context ###
{context}

### Next response ###
{target_speaker}: ({time_elapsed} later)"""


# Few-shot (n=2)
inst_time_fewshot = """### Dialogue context ###
{ex1_context}

Answer format: n (0<=n<=1440) minutes
The estimated time interval is: {ex1_time_elapsed}

[Example 2]
### Dialogue context ###
{ex2_context}

Answer format: n (0<=n<=1440) minutes
The estimated time interval is: {ex2_time_elapsed}

### Dialogue context ###
{context}

Answer format: n (0<=n<=1440) minutes
The estimated time interval is:"""

inst_response_fewshot = """### Dialogue context ###
{ex1_context}

### Next response ###
{ex1_target_speaker}: ({ex1_time_elapsed} later) {ex1_response}

[Example 2]
### Dialogue context ###
{ex2_context}

### Next response ###
{ex2_target_speaker}: ({ex2_time_elapsed} later) {ex2_response}

### Dialogue context ###
{context}

### Next response ###
{target_speaker}: ({time_elapsed} later)"""


# CoT
cot_time_ex1 = """### Dialogue context ###
A: I just got home. What a day!
B: It's already 11 p.m., and you're just getting home? That must have been a really tough day today.
A: Whoa, I need a shower. I'm exhausted.
B: Let the shower wash away all your fatigue.

### Time interval ###
Let's think step by step.
It is natural that A goes to take a shower after B's last utterance. Typically, a shower takes about 20 minutes, so we can expect A will respond in 20 minutes.

Therefore, the answer: 20 minutes"""

cot_time_ex2 = """### Dialogue context ###
A: I've been really into watching movies lately.
B: What genre do you like?
A: Recently, I've been watching a lot of thrillers.
B: Oh, I haven't watched many thrillers. Any recommendations?

Answer format: n (0<=n<=1440) minutes
The estimated time interval is:
Let's think step by step.
It is natural that A will recommend a thriller movie to B. It takes little time to think of one, so we can expect A will respond immediately.

Therefore, the answer: 0 minutes"""

cot_response_ex1 = """### Dialogue context ###
A: I just got home. What a day!
B: It's already 11 p.m., and you're just getting home? That must have been a really tough day today.
A: Whoa, I need a shower. I'm exhausted.
B: Let the shower wash away all your fatigue.

### Time interval ###
20 minutes

### Next response ###
Let's think step by step.
It seems that A took 20 minutes to take a shower. It is likely that A will talk about the feeling after taking a shower.

Therefore, the answer: A: I feel much better now. Have you been waiting long?"""

cot_response_ex2 = """### Dialogue context ###
A: I've been really into watching movies lately.
B: What genre do you like?
A: Recently, I've been watching a lot of thrillers.
B: Oh, I haven't watched many thrillers. Any recommendations?

### Time interval ###
0 minutes

### Next response ###
Let's think step by step.
It seems that A took no time to think of one. It is likely that A will recommend a thriller movie.

Therefore, the answer: A: Have you seen Zodiac? It's one of the best I've ever seen."""

inst_time_cot = """[Example 1]
{cot_ex1}

[Example 2]
{cot_ex2}

### Dialogue context ###
{context}

### Time interval ###
Let's think step by step.
"""

inst_response_cot = """[Example 1]
{cot_ex1}

[Example 2]
{cot_ex2}

### Dialogue context ###
{context}

### Time interval ###
{time_elapsed}

### Next response ###
Let's think step by step.
"""


def get_instruction(task: str, icl_method: str) -> Tuple[str, str]:
    """
    :param task: which task to perform
    :param icl_method: which ICL method to use
    :return: (system_prompt, user_prompt)
    """
    if task == "time":
        if icl_method == "zeroshot":
            return desc_time, inst_time_zeroshot
        elif icl_method == "fewshot":
            return desc_time, inst_time_fewshot
        elif icl_method == "cot":
            return desc_time, inst_time_cot
        else:
            raise ValueError(f"Invalid ICL method: {icl_method}")
    elif task == "response":
        if icl_method == "zeroshot":
            return desc_response, inst_response_zeroshot
        elif icl_method == "fewshot":
            return desc_response, inst_response_fewshot
        elif icl_method == "cot":
            return desc_response, inst_response_cot
        else:
            raise ValueError(f"Invalid ICL method: {icl_method}")
    else:
        raise ValueError(f"Invalid task: {task}")
