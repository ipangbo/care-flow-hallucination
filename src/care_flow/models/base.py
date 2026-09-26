from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class MultiModalBackend(ABC):
    """Backend contract required by the paper CARE-Flow computation."""

    @abstractmethod
    def generate_multimodal(
        self, prompt: str, image_paths: list[Path], max_new_tokens: int = 320
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def score_multimodal_completion(
        self,
        prompt: str,
        image_paths: list[Path],
        completion: str,
        image_gates: list[float],
    ) -> float:
        """Mean token log-likelihood under exact source value-output gates."""

    @abstractmethod
    def score_multimodal_candidates(
        self, prompt: str, image_paths: list[Path], candidates: list[str]
    ) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def generate_text(self, prompt: str, max_new_tokens: int = 96) -> str:
        raise NotImplementedError

    @abstractmethod
    def score_text_candidates(self, prompt: str, candidates: list[str]) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def reset_multimodal_state(self) -> None:
        """Drop backend-owned image states and caches before text-only inference."""
