import re
import math
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch


def tokenize(sentence: str) -> List[str]:
    sentence = sentence.lower().strip()
    sentence = re.sub(r"[^a-z0-9\s]", "", sentence)
    return sentence.split()


def compute_ngrams(tokens: List[str], n: int) -> Dict[tuple, int]:
    ngrams = defaultdict(int)
    for i in range(len(tokens) - n + 1):
        ngrams[tuple(tokens[i : i + n])] += 1
    return ngrams


def bleu_score(
    predictions: List[str],
    references: List[List[str]],
    max_n: int = 4,
) -> Dict[str, float]:
    scores = {}
    for n in range(1, max_n + 1):
        clip_count = 0
        total_count = 0
        pred_len = 0
        ref_len = 0

        for pred, refs in zip(predictions, references):
            pred_tokens = tokenize(pred)
            pred_ngrams = compute_ngrams(pred_tokens, n)

            max_ref_ngrams = defaultdict(int)
            best_ref_len = min(
                (abs(len(tokenize(r)) - len(pred_tokens)), len(tokenize(r)))
                for r in refs
            )[1]

            for ref in refs:
                ref_tokens = tokenize(ref)
                ref_ngrams = compute_ngrams(ref_tokens, n)
                for ng, cnt in ref_ngrams.items():
                    max_ref_ngrams[ng] = max(max_ref_ngrams[ng], cnt)

            for ng, cnt in pred_ngrams.items():
                clip_count += min(cnt, max_ref_ngrams[ng])
                total_count += cnt

            pred_len += len(pred_tokens)
            ref_len += best_ref_len

        if total_count == 0:
            scores[f"BLEU-{n}"] = 0.0
            continue

        precision = clip_count / total_count

        if pred_len < ref_len:
            bp = math.exp(1 - ref_len / (pred_len + 1e-8))
        else:
            bp = 1.0

        scores[f"BLEU-{n}"] = bp * precision * 100.0

    return scores


def meteor_score(predictions: List[str], references: List[List[str]]) -> float:
    def _align(pred_tokens, ref_tokens):
        matches = sum(1 for t in pred_tokens if t in ref_tokens)
        return matches

    total = 0.0
    for pred, refs in zip(predictions, references):
        pred_tokens = tokenize(pred)
        best = 0.0
        for ref in refs:
            ref_tokens = tokenize(ref)
            m = _align(pred_tokens, ref_tokens)
            if m == 0:
                continue
            P = m / (len(pred_tokens) + 1e-8)
            R = m / (len(ref_tokens) + 1e-8)
            if P + R == 0:
                continue
            F = 10 * P * R / (9 * P + R + 1e-8)
            chunks = 1
            penalty = 0.5 * (chunks / (m + 1e-8)) ** 3
            score = F * (1 - penalty)
            best = max(best, score)
        total += best

    return total / (len(predictions) + 1e-8) * 100.0


def rouge_l_score(predictions: List[str], references: List[List[str]]) -> float:
    def lcs_length(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    total = 0.0
    for pred, refs in zip(predictions, references):
        pred_tokens = tokenize(pred)
        best = 0.0
        for ref in refs:
            ref_tokens = tokenize(ref)
            lcs = lcs_length(pred_tokens, ref_tokens)
            P = lcs / (len(pred_tokens) + 1e-8)
            R = lcs / (len(ref_tokens) + 1e-8)
            if P + R == 0:
                continue
            F = 2 * P * R / (P + R + 1e-8)
            best = max(best, F)
        total += best

    return total / (len(predictions) + 1e-8) * 100.0


def cider_d_score(
    predictions: List[str],
    references: List[List[str]],
    n: int = 4,
    sigma: float = 6.0,
) -> float:
    def _ngrams_all(sentences, max_n):
        all_ngrams = []
        for sent in sentences:
            tokens = tokenize(sent)
            d = {}
            for nn in range(1, max_n + 1):
                ngs = compute_ngrams(tokens, nn)
                d.update({(nn, k): v for k, v in ngs.items()})
            all_ngrams.append(d)
        return all_ngrams

    all_preds = predictions
    all_refs_flat = [r for refs in references for r in refs]

    doc_freq = defaultdict(int)
    num_docs = len(references)
    for refs in references:
        seen = set()
        for ref in refs:
            tokens = tokenize(ref)
            for nn in range(1, n + 1):
                for ng in set(compute_ngrams(tokens, nn).keys()):
                    key = (nn, ng)
                    if key not in seen:
                        doc_freq[key] += 1
                        seen.add(key)

    def _tfidf(sent, max_n):
        tokens = tokenize(sent)
        vec = {}
        for nn in range(1, max_n + 1):
            ngs = compute_ngrams(tokens, nn)
            for ng, cnt in ngs.items():
                key = (nn, ng)
                tf = cnt / (len(tokens) + 1e-8)
                idf = math.log((num_docs + 1) / (doc_freq.get(key, 0) + 1))
                vec[key] = tf * idf
        return vec

    total = 0.0
    for pred, refs in zip(predictions, references):
        pred_vec = _tfidf(pred, n)
        pred_len = len(tokenize(pred))

        score = 0.0
        for ref in refs:
            ref_vec = _tfidf(ref, n)
            ref_len = len(tokenize(ref))

            dot = sum(pred_vec.get(k, 0) * v for k, v in ref_vec.items())
            pred_norm = math.sqrt(sum(v ** 2 for v in pred_vec.values()) + 1e-8)
            ref_norm = math.sqrt(sum(v ** 2 for v in ref_vec.values()) + 1e-8)

            penalty = math.exp(-(pred_len - ref_len) ** 2 / (2 * sigma ** 2))
            score += penalty * dot / (pred_norm * ref_norm + 1e-8)

        score /= len(refs)
        total += score

    return total / (len(predictions) + 1e-8) * 10.0 * 100.0


def compute_detection_metrics(
    preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> Dict[str, float]:
    pred_bin = (torch.sigmoid(preds) > threshold).long().cpu()
    targets = targets.long().cpu()

    tp = ((pred_bin == 1) & (targets == 1)).sum().item()
    fp = ((pred_bin == 1) & (targets == 0)).sum().item()
    fn = ((pred_bin == 0) & (targets == 1)).sum().item()
    tn = ((pred_bin == 0) & (targets == 0)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {
        "F1": f1 * 100.0,
        "IoU": iou * 100.0,
        "Precision": precision * 100.0,
        "Recall": recall * 100.0,
    }


def compute_caption_metrics(
    predictions: List[str],
    references: List[List[str]],
) -> Dict[str, float]:
    bleu = bleu_score(predictions, references)
    meteor = meteor_score(predictions, references)
    rouge = rouge_l_score(predictions, references)
    cider = cider_d_score(predictions, references)

    return {
        **bleu,
        "METEOR": meteor,
        "ROUGE-L": rouge,
        "CIDEr-D": cider,
    }


def compute_all_metrics(
    predictions: List[str],
    references: List[List[str]],
    pred_maps: torch.Tensor,
    gt_masks: torch.Tensor,
) -> Dict[str, float]:
    caption_metrics = compute_caption_metrics(predictions, references)
    detection_metrics = compute_detection_metrics(pred_maps, gt_masks)
    return {**caption_metrics, **detection_metrics}
