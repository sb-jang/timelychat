# Timely Response Generation for Open-domain Dialogue Agents

This is the official repository for "[From what to respond to when to respond: Timely Response Generation for Open-domain Dialogue Agents](https://www.arxiv.org/abs/2506.14285)".

## Data

TimelyChat dataset (both train and evaluation sets) is available on [🤗Datasets Hub](https://huggingface.co/datasets/seongbo/timelychat).

## Checkpoint

Timer-3B trained on TimelyChat-train is available on [🤗Model Hub](https://huggingface.co/seongbo/timer-3b).

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
    --model-name seongbo/timer-3b \
    --task time \
    --icl-method zeroshot \
    --save-results
```

- Dialog-level evaluation

```
python evaluate_dialog-level.py --model-type hf \
    --model-name seongbo/timer-3b \
    --simulator gpt-4o \
    --num-turns 10
```

## Citation
```bibtex
@article{jang2025respond,
  title={From What to Respond to When to Respond: Timely Response Generation for Open-domain Dialogue Agents},
  author={Jang, Seongbo and Jeon, Minjin and Lee, Jaehoon and Lee, Seonghyeon and Lee, Dongha and Yu, Hwanjo},
  journal={arXiv preprint arXiv:2506.14285},
  year={2025}
}
```
