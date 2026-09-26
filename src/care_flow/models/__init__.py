"""Paper-aligned multimodal backend for CARE-Flow."""

from .base import MultiModalBackend
from .internvl3 import InternVL3Backbone, TransformersVLMBackbone
from .qwen_vl import QwenVLBackend

__all__ = [
    "InternVL3Backbone",
    "MultiModalBackend",
    "QwenVLBackend",
    "TransformersVLMBackbone",
]
