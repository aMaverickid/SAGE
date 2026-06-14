"""Composite token-set agreement signal for SAGE-Causal triage and OCRS.

Compares two token sets — typically exemplar-derived top tokens (T_in) against
logit-lens-derived top tokens (T_out_pos / T_out_neg) — and returns a scalar
agreement in [0, 1]. The score combines three components:

  exact_jaccard:   |T_in ∩ T_out| / |T_in ∪ T_out|         (strict overlap)
  norm_jaccard:    Jaccard on normalized tokens             (case/whitespace robust)
  ngram_sim:       avg max char-trigram Jaccard             (morphological family)

Agreement is computed against pos and neg sets separately and the max is
returned, along with which direction matched. Mid-layer MLP features often
have their semantic family in neg_str rather than pos_str.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Literal, Tuple

# Weights for the composite score. Frozen for now; calibrate on pilot.
W_EXACT = 0.2
W_NORM = 0.4
W_NGRAM = 0.4
NGRAM_N = 3


def _normalize(token: str) -> str:
    """Lowercase, strip leading ▁ (SentencePiece) and surrounding punctuation."""
    t = token.lstrip("▁").strip()
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"^[^\w]+|[^\w]+$", "", t)
    return t.lower()


def _char_ngrams(s: str, n: int = NGRAM_N) -> set:
    s = s.lower()
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    union = sa | sb
    return len(sa & sb) / len(union)


def _ngram_sim(t_in: List[str], t_out: List[str]) -> float:
    """For each t_in token, max char-trigram Jaccard against any t_out token; average."""
    if not t_in or not t_out:
        return 0.0
    out_grams = [_char_ngrams(_normalize(t)) for t in t_out]
    sims = []
    for t in t_in:
        in_grams = _char_ngrams(_normalize(t))
        if not in_grams:
            sims.append(0.0)
            continue
        best = max(
            (len(in_grams & og) / len(in_grams | og)) if og else 0.0
            for og in out_grams
        )
        sims.append(best)
    return sum(sims) / len(sims)


@dataclass
class AgreementResult:
    """Composite agreement score and per-component breakdown.

    agreement: the final scalar in [0, 1] used by triage and OCRS.
    direction: which side of the logit-lens (pos/neg) matched best, or "none".
    pos_score / neg_score: per-direction composite scores.
    components: per-direction (exact, norm, ngram) for audit / debugging.
    """
    agreement: float
    direction: Literal["pos", "neg", "tie", "none"]
    pos_score: float
    neg_score: float
    components: dict


def _composite(t_in: List[str], t_out: List[str]) -> Tuple[float, dict]:
    in_norm = [_normalize(t) for t in t_in if _normalize(t)]
    out_norm = [_normalize(t) for t in t_out if _normalize(t)]
    exact = _jaccard(t_in, t_out)
    norm = _jaccard(in_norm, out_norm)
    ngram = _ngram_sim(t_in, t_out)
    score = W_EXACT * exact + W_NORM * norm + W_NGRAM * ngram
    return score, {"exact": exact, "norm": norm, "ngram": ngram}


def compute_agreement(
    t_in: List[str],
    t_out_pos: List[str],
    t_out_neg: List[str],
) -> AgreementResult:
    """Compute composite agreement of exemplar tokens against pos+neg logit-lens sets."""
    pos_score, pos_parts = _composite(t_in, t_out_pos)
    neg_score, neg_parts = _composite(t_in, t_out_neg)

    if pos_score == 0 and neg_score == 0:
        direction: Literal["pos", "neg", "tie", "none"] = "none"
    elif abs(pos_score - neg_score) < 1e-6:
        direction = "tie"
    elif pos_score > neg_score:
        direction = "pos"
    else:
        direction = "neg"

    return AgreementResult(
        agreement=max(pos_score, neg_score),
        direction=direction,
        pos_score=pos_score,
        neg_score=neg_score,
        components={"pos": pos_parts, "neg": neg_parts},
    )


# Triage path selection — thresholds calibrated on pilot, frozen for eval.
FAST_THRESHOLD = 0.40
DEEP_THRESHOLD = 0.15


def select_path(agreement: float, out_entropy: float, entropy_high: float = 0.7) -> str:
    """Return the triage path label given agreement and logit-lens entropy.

    out_entropy is expected normalized (0..1, e.g. raw_entropy / log(K)).
    """
    if agreement >= FAST_THRESHOLD and out_entropy < entropy_high:
        return "FAST"
    if agreement < DEEP_THRESHOLD or out_entropy >= entropy_high:
        return "DEEP"
    return "STANDARD"
