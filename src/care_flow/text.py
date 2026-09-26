from __future__ import annotations

import re
import string


_PUNCT_TABLE = str.maketrans({p: " " for p in string.punctuation})


def normalize_answer(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def answer_in_text(answer: str, text: str) -> bool:
    answer_norm = normalize_answer(answer)
    text_norm = normalize_answer(text)
    if not answer_norm:
        return False
    return answer_norm in text_norm


def exact_or_contains(prediction: str, target: str) -> bool:
    pred_norm = normalize_answer(prediction)
    target_norm = normalize_answer(target)
    if not pred_norm or not target_norm:
        return False
    return pred_norm == target_norm or target_norm in pred_norm


def choose_first_matching_answer(prediction: str, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if answer_in_text(candidate, prediction):
            return candidate
    return None

