from typing import List

import numpy as np

# NOTE: heavy deps (torchmetrics, evaluate) are imported lazily inside the
# text-generation metrics so the numeric timing metrics (rmsle/precision/recall/
# f1_score/fpr) can be used from saved result files without the full GPU stack.


def rmsle(y_true: List[float], y_pred: List[float]) -> float:
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    squared_error = (log_true - log_pred) ** 2
    return np.sqrt(np.mean(squared_error))


def bleu(refs: List[str], preds: List[str]) -> float:
    from torchmetrics.text import SacreBLEUScore

    refs = [[ref] for ref in refs]
    scorer = SacreBLEUScore(n_gram=2)
    bleu_score = scorer(preds, refs).item()
    return bleu_score * 100


def rouge(refs: List[str], preds: List[str]) -> float:
    from torchmetrics.text import ROUGEScore

    scorer = ROUGEScore()
    rouge_scores = scorer(preds, refs)
    return rouge_scores["rougeL_fmeasure"].item() * 100


def bertscore(refs: List[str], preds: List[str]) -> float:
    from evaluate import load

    scorer = load("bertscore")
    bertscore = scorer.compute(predictions=preds, references=refs, lang="en")
    num_examples = len(preds)
    return sum(bertscore["f1"]) / num_examples * 100


def _map_to_binary(value: float) -> int:
    """
    Map value to binary: 0 if close to 0.0, 1 otherwise.
    Input is assumed to be already converted to minutes (float).
    """
    # Consider values close to 0.0 as 0 (false), others as 1 (true)
    return 0 if abs(value) < 1e-6 else 1


def precision(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate precision with binary mapping.
    Values close to 0.0 are mapped to 0 (false), others to 1 (true).
    Precision = TP / (TP + FP)
    where TP = True Positives (correct predictions)
          FP = False Positives (incorrect predictions)
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    y_true_binary = [_map_to_binary(v) for v in y_true]
    y_pred_binary = [_map_to_binary(v) for v in y_pred]

    tp = sum(1 for true, pred in zip(y_true_binary, y_pred_binary) if true == 1 and pred == 1)
    fp = sum(1 for true, pred in zip(y_true_binary, y_pred_binary) if true == 0 and pred == 1)

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp) * 100


def recall(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate recall with binary mapping.
    Values close to 0.0 are mapped to 0 (false), others to 1 (true).
    Recall = TP / (TP + FN)
    where TP = True Positives (correct predictions)
          FN = False Negatives (missed correct predictions)
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    y_true_binary = [_map_to_binary(v) for v in y_true]
    y_pred_binary = [_map_to_binary(v) for v in y_pred]

    tp = sum(1 for true, pred in zip(y_true_binary, y_pred_binary) if true == 1 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true_binary, y_pred_binary) if true == 1 and pred == 0)

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn) * 100


def f1_score(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate F1 score with binary mapping.
    Values close to 0.0 are mapped to 0 (false), others to 1 (true).
    F1 = 2 * (precision * recall) / (precision + recall)
    """
    prec = precision(y_true, y_pred) / 100
    rec = recall(y_true, y_pred) / 100

    if prec + rec == 0:
        return 0.0

    return 2 * (prec * rec) / (prec + rec) * 100


def fpr(y_true: List[float], y_pred: List[float]) -> float:
    """
    Calculate False Positive Rate (FPR) with binary mapping.
    Values close to 0.0 are mapped to 0 (false), others to 1 (true).
    FPR = FP / (FP + TN) = FP / (actual negatives)
    where FP = False Positives (actual 0, predicted 1)
          TN = True Negatives (actual 0, predicted 0)
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    y_true_binary = [_map_to_binary(v) for v in y_true]
    y_pred_binary = [_map_to_binary(v) for v in y_pred]

    fp = sum(1 for true, pred in zip(y_true_binary, y_pred_binary) if true == 0 and pred == 1)
    tn = sum(1 for true, pred in zip(y_true_binary, y_pred_binary) if true == 0 and pred == 0)

    if fp + tn == 0:
        return 0.0

    return fp / (fp + tn) * 100
