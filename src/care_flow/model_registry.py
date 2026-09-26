from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    model_id: str
    dtype: str


MODELS = {
    "llava-next-7b": ModelSpec(
        key="llava-next-7b",
        label="LLaVA-NeXT-7B",
        model_id="llava-hf/llava-interleave-qwen-7b-hf",
        dtype="float16",
    ),
    "qwen3.5-9b": ModelSpec(
        key="qwen3.5-9b",
        label="Qwen3.5-9B",
        model_id="Qwen/Qwen3.5-9B",
        dtype="bfloat16",
    ),
    "internvl3-8b": ModelSpec(
        key="internvl3-8b",
        label="InternVL3-8B",
        model_id="OpenGVLab/InternVL3-8B-hf",
        dtype="bfloat16",
    ),
}


def get_model(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError as exc:
        choices = ", ".join(sorted(MODELS))
        raise ValueError(f"Unknown model {key!r}; choose one of: {choices}") from exc
