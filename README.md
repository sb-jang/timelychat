# Timely Response Generation for Open-domain Dialogue Agents

This is the official repository for "From what to respond to when to respond: Timely Response Generation for Open-domain Dialogue Agents".

## Data

TimelyChat dataset (both train and evaluation sets) is available on [🤗Datasets Hub](https://huggingface.co/datasets/anonymous17711771/timelychat).

## Checkpoint

Timer-3B trained on TimelyChat-train is available on [🤗Model Hub](https://huggingface.co/anonymous17711771/timer-3b).

## Usage

### Environment setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Evaluation

- Turn-level evaluation

```
python evaluate_turn-level.py --model-type hf \
    --model-name anonymous17711771/timer-3b \
    --task time \
    --icl-method zeroshot \
    --save-results
```

- Dialog-level evaluation

```
python evaluate_dialog-level.py --model-type hf \
    --model-name anonymous17711771/timer-3b \
    --simulator gpt-4o \
    --num-turns 10
```
