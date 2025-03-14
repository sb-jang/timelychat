from typing import List

import numpy as np
from evaluate import load
from torchmetrics.text import ROUGEScore, SacreBLEUScore


def rmsle(y_true: List[float], y_pred: List[float]) -> float:
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    squared_error = (log_true - log_pred) ** 2
    return np.sqrt(np.mean(squared_error))


def bleu(refs: List[str], preds: List[str]) -> float:
    refs = [[ref] for ref in refs]
    scorer = SacreBLEUScore(n_gram=2)
    bleu_score = scorer(preds, refs).item()
    return bleu_score * 100


def rouge(refs: List[str], preds: List[str]) -> float:
    scorer = ROUGEScore()
    rouge_scores = scorer(preds, refs)
    return rouge_scores["rougeL_fmeasure"].item() * 100


def bertscore(refs: List[str], preds: List[str]) -> float:
    scorer = load("bertscore")
    bertscore = scorer.compute(predictions=preds, references=refs, lang="en")
    num_examples = len(preds)
    return sum(bertscore["f1"]) / num_examples * 100
