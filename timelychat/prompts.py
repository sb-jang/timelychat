from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel


# response formats for structured output of GPT-4o
class Output(BaseModel):
    answer: str


class CoTOutput(BaseModel):
    reasoning: str
    answer: str


# response format for llm-as-a-judge
class Evaluation(BaseModel):
    score: int
    explanation: str


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


turn_level_system_template = """You will be given a conversation between two individuals via messaging, along with the elapsed time since the last utterance.
You will then be given a potential response for the next turn.
Your task is to rate the response on one metric. Please make sure you read and understand these instructions carefully.

Evaluation Criteria:
{metric} (1-5): {criteria}

Evaluation Steps:
{steps}"""

dialog_level_system_template = """You will be given a conversation between a dialogue agent and a user.
Throughout the conversation, the agent proactively determines the delay of its response to the user's previous message, simulating delayed responses due to event experiences that take certain time to process.
At each agent's turn, the delay is provided in the parentheses followed by the message. If no parentheses are provided, it means the agent responded immediately.
Your task is to rate the dialogue agent on one metric. Please make sure you read and understand these instructions carefully.

Evaluation Criteria:
{metric} (1-5): {criteria}

Evaluation Steps:
{steps}"""

turn_level_user_template = """### Dialogue Context ###
{context}

### Elapsed Time ###
{time_elapsed}

### Model Response ###
{response}"""

turn_level_rubrics = [
    {
        "metric": "Naturalness",
        "criteria": "the extent to which the response reads naturally given the dialogue context.",
        "steps": "1. Assess the flow and coherence of the response in the conversation: Consider how seamlessly the response connects with the previous message.\n2. Evaluate the tone and style compatibility: Determine if the response's tone and style match those of the previous messages.\n3. Rate on a scale from 1 to 5, where 1 indicates the response is unnatural or inappropriate, and 5 indicates a perfectly natural continuation of the conversation.",
    },
    {
        "metric": "Time-Specificity",
        "criteria": "the extent to which the response ONLY makes sense when the specified time has passed, contrary to a time-agnostic response that makes sense regardless of time.",
        "steps": "1. Read the provided conversation and take note of the elapsed time since the previous message.\n2. Consider the context of the conversation, focusing on how the passage of time might affect the relevance or appropriateness of the resopnse.\n3. Evaluate whether the potential response provided is time-specific. That is, determine if the response directly relates to or is clearly influenced by the elapsed time between the last utterance and the response.",
    },
]

dialog_level_rubrics = [
    {
        "metric": "Coherence",
        "criteria": "the extent to which the agent maintains a good conversation flow.",
        "steps": "1. Assess the flow and coherence of the agent's responses in the conversation.\n2. Evaluate the tone and style compatibility throughout the conversation.\n3. Rate on a scale from 1 to 5, where 1 indicates the agent's responses are incoherent or inappropriate, and 5 indicates the agent's responses are perfectly coherent and appropriate.",
    },
    {
        "metric": "Delay-Appropriateness",
        "criteria": "the extent to which the agent poses delays with appropriate frequency and duration.",
        "steps": "1. Assess whether the agent poses unnecessary or excessively frequent delays that could harm the conversation flow.\n2. Evaluate whether the amounts of delays (if not 0 minutes) reflect the typical duration of events implied in the corresponding message.\n3. Rate on a scale from 1 to 5, where 1 indicates the agent overuses and misuses delays, and 5 indicates the agent uses delays appropriately in terms of frequency and duration.",
    },
    {
        "metric": "Time-Specificity",
        "criteria": "the extent to which the agent's responses ONLY make sense when the specified time has passed, contrary to a time-agnostic responses that make sense regardless of time.",
        "steps": "1. Read the provided conversation and take note of the elapsed times since the previous messages.\n2. Consider the context of the conversation, focusing on how the passage of time might affect the relevance or appropriateness of the agent's responses.\n3. Evaluate whether the agent's responses are time-specific. That is, determine if the responses directly relate to or are clearly influenced by the elapsed times.\n4. Rate on a scale from 1 to 5, where 1 indicates the agent's responses are completely time-agnostic and unaffected by the passage of time, and 5 indicates the agent's responses are entirely time-specific; they only make sense because of the amount of time that has passed since the previous message.",
    },
]


def get_laaj_prompts(setting: str) -> Tuple[str, Optional[str]]:
    if setting == "turn-level":
        return turn_level_system_template, turn_level_user_template
    elif setting == "dialog-level":
        return dialog_level_system_template, None
    else:
        raise ValueError(f"Invalid setting: {setting}")


def get_laaj_rubrics(setting: str) -> List[Dict[str, str]]:
    if setting == "turn-level":
        return turn_level_rubrics
    elif setting == "dialog-level":
        return dialog_level_rubrics
    else:
        raise ValueError(f"Invalid setting: {setting}")
