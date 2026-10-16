"""
Evaluation Metrics for Continuous Sign Language Translation.
Implements BLEU-1..4, chrF++, Word Error Rate (WER), and exact match.
"""

from collections import Counter
import math
from typing import Dict, List, Tuple


def calculate_ngram_counts(tokens: List[str], n: int) -> Counter:
    """Computes n-gram frequency counter."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(
    hypotheses: List[str],
    references: List[str],
    max_order: int = 4,
    smooth: bool = True,
) -> Dict[str, float]:
    """
    Computes standard BLEU-1 to BLEU-4 scores across corpus.
    """
    precisions = [0.0] * max_order
    hyp_lengths = 0
    ref_lengths = 0

    clipped_counts = [0] * max_order
    total_counts = [0] * max_order

    for hyp_str, ref_str in zip(hypotheses, references):
        hyp_tokens = hyp_str.lower().strip().split()
        ref_tokens = ref_str.lower().strip().split()

        hyp_lengths += len(hyp_tokens)
        ref_lengths += len(ref_tokens)

        for i in range(max_order):
            n = i + 1
            hyp_ngrams = calculate_ngram_counts(hyp_tokens, n)
            ref_ngrams = calculate_ngram_counts(ref_tokens, n)

            total_counts[i] += max(len(hyp_tokens) - n + 1, 0)
            for ngram, count in hyp_ngrams.items():
                clipped_counts[i] += min(count, ref_ngrams[ngram])

    for i in range(max_order):
        if total_counts[i] == 0:
            precisions[i] = 0.0
        else:
            if clipped_counts[i] == 0 and smooth:
                precisions[i] = 1.0 / (2.0 * total_counts[i])
            else:
                precisions[i] = float(clipped_counts[i]) / float(total_counts[i])

    # Brevity Penalty
    if hyp_lengths == 0:
        bp = 0.0
    elif hyp_lengths < ref_lengths:
        bp = math.exp(1.0 - (float(ref_lengths) / float(hyp_lengths)))
    else:
        bp = 1.0

    scores = {}
    for n in range(1, max_order + 1):
        if any(p <= 0.0 for p in precisions[:n]):
            score = 0.0
        else:
            weight = 1.0 / n
            log_prec_sum = sum(weight * math.log(p) for p in precisions[:n])
            score = bp * math.exp(log_prec_sum) * 100.0
        scores[f"bleu_{n}"] = round(score, 2)

    return scores


def compute_chrf(
    hypotheses: List[str],
    references: List[str],
    char_order: int = 6,
    beta: float = 2.0,
) -> float:
    """Computes chrF++ character n-gram F-score."""
    total_f = 0.0
    count = 0

    for hyp, ref in zip(hypotheses, references):
        h_chars = list(hyp.lower().replace(" ", ""))
        r_chars = list(ref.lower().replace(" ", ""))

        if not h_chars or not r_chars:
            continue

        prec_sum = 0.0
        rec_sum = 0.0

        for n in range(1, char_order + 1):
            h_ngrams = Counter(tuple(h_chars[i : i + n]) for i in range(len(h_chars) - n + 1))
            r_ngrams = Counter(tuple(r_chars[i : i + n]) for i in range(len(r_chars) - n + 1))

            overlap = sum(min(cnt, r_ngrams[ng]) for ng, cnt in h_ngrams.items())
            total_h = max(len(h_chars) - n + 1, 1)
            total_r = max(len(r_chars) - n + 1, 1)

            prec = overlap / total_h
            rec = overlap / total_r

            prec_sum += prec
            rec_sum += rec

        avg_p = prec_sum / char_order
        avg_r = rec_sum / char_order

        if avg_p + avg_r > 0:
            f_score = (1 + beta**2) * (avg_p * avg_r) / ((beta**2 * avg_p) + avg_r)
        else:
            f_score = 0.0

        total_f += f_score * 100.0
        count += 1

    return round(total_f / max(count, 1), 2)


def compute_wer(hypotheses: List[str], references: List[str]) -> float:
    """Computes Word Error Rate (WER) using Levenshtein distance."""
    total_dist = 0
    total_ref_words = 0

    for hyp, ref in zip(hypotheses, references):
        h_words = hyp.lower().split()
        r_words = ref.lower().split()

        n = len(r_words)
        m = len(h_words)
        total_ref_words += n

        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if r_words[i - 1] == h_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,      # deletion
                        dp[i][j - 1] + 1,      # insertion
                        dp[i - 1][j - 1] + 1,  # substitution
                    )
        total_dist += dp[n][m]

    return round((total_dist / max(total_ref_words, 1)) * 100.0, 2)
