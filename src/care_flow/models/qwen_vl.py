from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from .base import MultiModalBackend


ATTENTION_INTERFACE_NAME = "care_flow_source_value_output"


class QwenVLBackend(MultiModalBackend):
    """Paper-aligned CARE-Flow backend for supported Transformers VLMs.

    The intervention is applied after softmax and source-wise value
    aggregation but before the output projection at every language self-
    attention layer. It therefore implements Proposition 1 exactly; it is not
    pixel scaling, embedding scaling, attention-logit masking, or attention
    attribution.  The supported full-attention families are Qwen2-VL,
    Qwen2.5-VL, Qwen3-VL, LLaVA-Interleave-Qwen, and InternVL3-Qwen.  Qwen3.5
    is hybrid: the same intervention is applied to its explicit full-attention
    blocks, while its linear-attention blocks remain unchanged.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str | None = None,
        torch_dtype: str = "bfloat16",
        max_image_side: int | None = 768,
    ) -> None:
        import torch
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            Qwen2VLForConditionalGeneration,
            Qwen2_5_VLForConditionalGeneration,
            Qwen3VLForConditionalGeneration,
        )
        try:
            from transformers import Qwen3_5ForConditionalGeneration
        except ImportError:
            Qwen3_5ForConditionalGeneration = None
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = resolve_dtype(torch, torch_dtype, device)
        lowered_model_path = model_path.casefold()
        if "qwen3.5" in lowered_model_path or "qwen3_5" in lowered_model_path:
            if Qwen3_5ForConditionalGeneration is None:
                raise RuntimeError(
                    "This Transformers build does not provide Qwen3.5 support"
                )
            model_class = Qwen3_5ForConditionalGeneration
        elif "qwen3-vl" in lowered_model_path or "qwen3_vl" in lowered_model_path:
            model_class = Qwen3VLForConditionalGeneration
        elif "qwen2.5" in lowered_model_path or "qwen2_5" in lowered_model_path:
            model_class = Qwen2_5_VLForConditionalGeneration
        elif "qwen2-vl" in lowered_model_path or "qwen2_vl" in lowered_model_path:
            model_class = Qwen2VLForConditionalGeneration
        else:
            # LLaVA-Interleave-Qwen and InternVL3 both expose a Qwen2 language
            # decoder through the standard image-to-text auto class.
            model_class = AutoModelForImageTextToText
        ALL_ATTENTION_FUNCTIONS.register(
            ATTENTION_INTERFACE_NAME, source_gated_eager_attention_forward
        )

        self.torch = torch
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=False
        )
        self.model = model_class.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=True,
            attn_implementation="sdpa",
            local_files_only=False,
        )
        self.model.to(self.device)
        self.model.eval()
        self.model_type = str(getattr(self.model.config, "model_type", ""))
        self.max_image_side = max_image_side
        self._retained_multimodal_state: dict[str, Any] = {}

        if not self._attention_modules():
            raise RuntimeError("No Qwen language self-attention layers were found")

    def generate_multimodal(
        self, prompt: str, image_paths: list[Path], max_new_tokens: int = 320
    ) -> str:
        images = self._load_images(image_paths)
        prompt_text = self._chat_text(prompt, len(images), add_generation_prompt=True)
        inputs = self._to_device(
            self.processor(text=[prompt_text], images=images, return_tensors="pt")
        )
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        prompt_length = int(inputs["input_ids"].shape[1])
        text = self.processor.tokenizer.decode(
            output[0, prompt_length:], skip_special_tokens=True
        ).strip()
        # Generation caches are local to the call; keep no image-derived state.
        self._retained_multimodal_state.clear()
        return text

    def score_multimodal_completion(
        self,
        prompt: str,
        image_paths: list[Path],
        completion: str,
        image_gates: list[float],
    ) -> float:
        if len(image_gates) != len(image_paths):
            raise ValueError("One image gate is required for every input image")
        images = self._load_images(image_paths)
        prompt_text = self._chat_text(prompt, len(images), add_generation_prompt=True)
        inputs, labels = self._completion_inputs(prompt_text, completion, images=images)
        spans = self._image_token_spans(inputs["input_ids"][0], len(image_paths))
        if len(spans) != len(image_paths):
            raise RuntimeError(
                f"Expected {len(image_paths)} image token spans, found {len(spans)}"
            )
        with self.torch.inference_mode(), self._source_path_gates(spans, image_gates):
            output = self.model(**inputs, use_cache=False)
            score = mean_completion_logprob(output.logits, inputs["input_ids"], labels)
        self._retained_multimodal_state.clear()
        return float(score.detach().cpu())

    def source_alignment_diagnostics(
        self,
        prompt: str,
        image_paths: list[Path],
        completion: str,
    ) -> dict[str, Any]:
        """Measure source-localization baselines for one fixed completion.

        Raw attention is the completion-query attention mass assigned to each
        image-token span, averaged over heads and language layers. Embedding
        masking is the drop in mean completion log probability after zeroing
        one image span at the input to the first language decoder layer. Both
        diagnostics use exactly the same prompt and completion as CARE's exact
        source-path intervention.
        """

        if not image_paths:
            raise ValueError("At least one image is required")
        images = self._load_images(image_paths)
        prompt_text = self._chat_text(prompt, len(images), add_generation_prompt=True)
        inputs, labels = self._completion_inputs(prompt_text, completion, images=images)
        spans = self._image_token_spans(inputs["input_ids"][0], len(image_paths))
        if len(spans) != len(image_paths):
            raise RuntimeError(
                f"Expected {len(image_paths)} image token spans, found {len(spans)}"
            )
        # A logit at position t predicts the label at t + 1.
        query_positions = (
            (labels[0, 1:] != -100).nonzero(as_tuple=False).flatten().tolist()
        )
        if not query_positions:
            raise ValueError("Completion produced no scoreable query positions")

        attention_layers: list[list[float]] = []
        with (
            self.torch.inference_mode(),
            self._source_path_gates(spans, [1.0] * len(spans)),
            self._collect_source_attention(spans, query_positions, attention_layers),
        ):
            output = self.model(**inputs, use_cache=False)
            baseline = mean_completion_logprob(
                output.logits, inputs["input_ids"], labels
            )

        masked_scores: list[float] = []
        with self.torch.inference_mode():
            for span in spans:
                with self._mask_first_layer_span(span):
                    output = self.model(**inputs, use_cache=False)
                    score = mean_completion_logprob(
                        output.logits, inputs["input_ids"], labels
                    )
                masked_scores.append(float(score.detach().cpu()))

        if not attention_layers:
            raise RuntimeError("No source attention masses were collected")
        attention = self.torch.tensor(attention_layers, dtype=self.torch.float32).mean(0)
        baseline_value = float(baseline.detach().cpu())
        self._retained_multimodal_state.clear()
        return {
            "baseline_logprob": baseline_value,
            "raw_attention_mass": [float(value) for value in attention.tolist()],
            "embedding_mask_effect": [
                baseline_value - masked_score for masked_score in masked_scores
            ],
            "embedding_masked_logprobs": masked_scores,
            "attention_layers": len(attention_layers),
            "completion_query_tokens": len(query_positions),
            "image_token_counts": [end - start for start, end in spans],
        }

    def score_multimodal_candidates(
        self, prompt: str, image_paths: list[Path], candidates: list[str]
    ) -> dict[str, float]:
        images = self._load_images(image_paths)
        prompt_text = self._chat_text(prompt, len(images), add_generation_prompt=True)
        scores: dict[str, float] = {}
        with self.torch.inference_mode():
            for candidate in candidates:
                inputs, labels = self._completion_inputs(prompt_text, candidate, images=images)
                output = self.model(**inputs, use_cache=False)
                score = mean_completion_logprob(output.logits, inputs["input_ids"], labels)
                scores[candidate] = float(score.detach().cpu())
        self._retained_multimodal_state.clear()
        return scores

    def generate_text(self, prompt: str, max_new_tokens: int = 96) -> str:
        prompt_text = self._chat_text(prompt, 0, add_generation_prompt=True)
        inputs = self._to_device(self.processor(text=[prompt_text], return_tensors="pt"))
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        prompt_length = int(inputs["input_ids"].shape[1])
        return self.processor.tokenizer.decode(
            output[0, prompt_length:], skip_special_tokens=True
        ).strip()

    def score_text_candidates(self, prompt: str, candidates: list[str]) -> dict[str, float]:
        prompt_text = self._chat_text(prompt, 0, add_generation_prompt=True)
        scores: dict[str, float] = {}
        with self.torch.inference_mode():
            for candidate in candidates:
                inputs, labels = self._completion_inputs(prompt_text, candidate, images=None)
                output = self.model(**inputs, use_cache=False)
                score = mean_completion_logprob(output.logits, inputs["input_ids"], labels)
                scores[candidate] = float(score.detach().cpu())
        return scores

    def reset_multimodal_state(self) -> None:
        self._retained_multimodal_state.clear()
        for attribute in ("rope_deltas",):
            for owner in (self.model, getattr(self.model, "model", None)):
                if owner is not None and hasattr(owner, attribute):
                    setattr(owner, attribute, None)

    def _completion_inputs(self, prompt_text: str, completion: str, images: Any):
        processor_kwargs: dict[str, Any] = {"text": [prompt_text], "return_tensors": "pt"}
        if images is not None:
            processor_kwargs["images"] = images
        prompt_inputs = self.processor(**processor_kwargs)
        prompt_length = int(prompt_inputs["input_ids"].shape[1])

        full_kwargs = dict(processor_kwargs)
        full_kwargs["text"] = [prompt_text + " " + completion.strip()]
        inputs = self.processor(**full_kwargs)
        labels = inputs["input_ids"].clone()
        labels[:, :prompt_length] = -100
        inputs = self._to_device(inputs)
        labels = labels.to(inputs["input_ids"].device)
        return inputs, labels

    def _chat_text(self, prompt: str, num_images: int, add_generation_prompt: bool) -> str:
        if self.model_type == "llava":
            # The released LLaVA-Interleave template deliberately renders all
            # images before all text.  That collapses adjacent expanded image
            # tokens into one span, so source-wise intervention cannot tell
            # the images apart.  Preserve the checkpoint's Qwen chat markers
            # while interleaving explicit source labels with placeholders.
            body = "".join(
                f"Image {image_index}:\n<image>\n"
                for image_index in range(1, num_images + 1)
            ) + prompt
            suffix = "<|im_start|>assistant\n" if add_generation_prompt else ""
            return f"<|im_start|>user\n{body}<|im_end|>\n{suffix}"
        content = []
        for image_index in range(1, num_images + 1):
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def _load_images(self, image_paths: Sequence[Path]):
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        images = []
        for path in image_paths:
            with Image.open(path) as source:
                image = source.convert("RGB")
            if self.max_image_side and max(image.size) > self.max_image_side:
                image.thumbnail(
                    (self.max_image_side, self.max_image_side), Image.Resampling.LANCZOS
                )
            images.append(image)
        return images

    def _to_device(self, inputs):
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _language_model(self):
        for candidate in (
            getattr(getattr(self.model, "model", None), "language_model", None),
            getattr(self.model, "language_model", None),
        ):
            if candidate is not None:
                return candidate
        return None

    def _attention_modules(self) -> list[Any]:
        language_model = self._language_model()
        roots = (
            language_model,
            getattr(language_model, "model", None) if language_model is not None else None,
        )
        modules: list[Any] = []
        seen: set[int] = set()
        for root in roots:
            layers = getattr(root, "layers", None)
            if layers is None:
                continue
            for layer in layers:
                attention = getattr(layer, "self_attn", None)
                if attention is not None and id(attention) not in seen:
                    modules.append(attention)
                    seen.add(id(attention))
        return modules

    def _language_layers(self) -> list[Any]:
        language_model = self._language_model()
        for root in (
            language_model,
            getattr(language_model, "model", None) if language_model is not None else None,
        ):
            layers = getattr(root, "layers", None)
            if layers is not None:
                return list(layers)
        return []

    def _image_token_spans(
        self, input_ids, expected_images: int
    ) -> list[tuple[int, int]]:
        image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(self.model.config, "image_token_index", None)
        if image_token_id is None:
            for token in ("<|image_pad|>", "<image>", "<IMG_CONTEXT>"):
                candidate = self.processor.tokenizer.convert_tokens_to_ids(token)
                unk_id = getattr(self.processor.tokenizer, "unk_token_id", None)
                if (
                    candidate is not None
                    and int(candidate) >= 0
                    and (unk_id is None or int(candidate) != int(unk_id))
                ):
                    image_token_id = int(candidate)
                    break
        positions = (
            (input_ids == int(image_token_id)).nonzero(as_tuple=False).flatten().tolist()
            if image_token_id is not None and image_token_id >= 0
            else []
        )
        if not positions:
            return []
        spans: list[tuple[int, int]] = []
        start = previous = positions[0]
        for position in positions[1:]:
            if position == previous + 1:
                previous = position
                continue
            spans.append((start, previous + 1))
            start = previous = position
        spans.append((start, previous + 1))
        return spans[:expected_images]

    @contextmanager
    def _source_path_gates(
        self, spans: list[tuple[int, int]], gates: Sequence[float]
    ) -> Iterator[None]:
        modules = self._attention_modules()
        config_originals: dict[int, tuple[Any, str]] = {}
        for module in modules:
            module._care_flow_source_spans = tuple(spans)
            module._care_flow_source_gates = tuple(float(gate) for gate in gates)
            config = module.config
            if id(config) not in config_originals:
                config_originals[id(config)] = (config, config._attn_implementation)
                config._attn_implementation = ATTENTION_INTERFACE_NAME
        try:
            yield
        finally:
            for module in modules:
                module.__dict__.pop("_care_flow_source_spans", None)
                module.__dict__.pop("_care_flow_source_gates", None)
            for config, original in config_originals.values():
                config._attn_implementation = original

    @contextmanager
    def _collect_source_attention(
        self,
        spans: Sequence[tuple[int, int]],
        query_positions: Sequence[int],
        collector: list[list[float]],
    ) -> Iterator[None]:
        modules = self._attention_modules()
        for module in modules:
            module._care_flow_attention_spans = tuple(spans)
            module._care_flow_attention_queries = tuple(int(item) for item in query_positions)
            module._care_flow_attention_collector = collector
        try:
            yield
        finally:
            for module in modules:
                module.__dict__.pop("_care_flow_attention_spans", None)
                module.__dict__.pop("_care_flow_attention_queries", None)
                module.__dict__.pop("_care_flow_attention_collector", None)

    @contextmanager
    def _mask_first_layer_span(self, span: tuple[int, int]) -> Iterator[None]:
        layers = self._language_layers()
        if not layers:
            raise RuntimeError("No Qwen language decoder layers were found")
        start, end = (int(span[0]), int(span[1]))

        def mask_hidden_states(module, args, kwargs):
            del module
            hidden_states = kwargs.get("hidden_states")
            from_kwargs = hidden_states is not None
            if hidden_states is None:
                if not args:
                    raise RuntimeError("Decoder layer received no hidden states")
                hidden_states = args[0]
            masked = hidden_states.clone()
            bounded_start = max(0, min(start, int(masked.shape[1])))
            bounded_end = max(bounded_start, min(end, int(masked.shape[1])))
            masked[:, bounded_start:bounded_end, :] = 0
            if from_kwargs:
                updated = dict(kwargs)
                updated["hidden_states"] = masked
                return args, updated
            return (masked, *args[1:]), kwargs

        handle = layers[0].register_forward_pre_hook(mask_hidden_states, with_kwargs=True)
        try:
            yield
        finally:
            handle.remove()


def resolve_dtype(torch, torch_dtype: str, device: str):
    if str(device).startswith("cpu"):
        return torch.float32
    if torch_dtype == "float32":
        return torch.float32
    if torch_dtype == "float16":
        return torch.float16
    return torch.bfloat16


def repeat_key_value(value_states, groups: int):
    if groups == 1:
        return value_states
    return value_states.repeat_interleave(groups, dim=1)


def source_gated_eager_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Eager attention with the exact post-aggregation source intervention."""

    torch = query.__class__.__module__.split(".")[0]
    if torch != "torch":
        raise TypeError("CARE-Flow source gates require torch tensors")
    import torch as torch_module

    groups = int(getattr(module, "num_key_value_groups", 1))
    key_states = repeat_key_value(key, groups)
    value_states = repeat_key_value(value, groups)
    attention_weights = torch_module.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attention_weights = attention_weights + attention_mask
    attention_weights = torch_module.nn.functional.softmax(
        attention_weights, dim=-1, dtype=torch_module.float32
    ).to(query.dtype)
    attention_weights = torch_module.nn.functional.dropout(
        attention_weights, p=dropout, training=module.training
    )
    collect_source_attention_mass(module, attention_weights)
    output = torch_module.matmul(attention_weights, value_states).transpose(1, 2).contiguous()
    output = apply_source_grouped_value_output_gate(
        output,
        attention_weights,
        value_states,
        getattr(module, "_care_flow_source_spans"),
        getattr(module, "_care_flow_source_gates"),
    )
    return output, attention_weights


def collect_source_attention_mass(module, attention_weights) -> None:
    """Append one layer's mean completion-query mass for every source span."""

    collector = getattr(module, "_care_flow_attention_collector", None)
    spans = getattr(module, "_care_flow_attention_spans", None)
    queries = getattr(module, "_care_flow_attention_queries", None)
    if collector is None or spans is None or queries is None:
        return
    query_length = int(attention_weights.shape[-2])
    key_length = int(attention_weights.shape[-1])
    valid_queries = [int(item) for item in queries if 0 <= int(item) < query_length]
    if not valid_queries:
        return
    selected = attention_weights[:, :, valid_queries, :].float()
    masses: list[float] = []
    for raw_start, raw_end in spans:
        start = max(0, min(int(raw_start), key_length))
        end = max(start, min(int(raw_end), key_length))
        mass = selected[..., start:end].sum(dim=-1).mean()
        masses.append(float(mass.detach().cpu()))
    collector.append(masses)


def apply_source_grouped_value_output_gate(
    attention_output,
    attention_weights,
    value_states,
    source_spans: Sequence[tuple[int, int]],
    source_gates: Sequence[float],
):
    """Apply o + (g_k-1)c_k outside source k, before W_O.

    `attention_output` is [batch, query, heads, head_dim], while weights and
    repeated values are [batch, heads, query/key, ...]. Applying the gate in
    per-head space and then calling the model's linear output projection is
    exactly equivalent to gating c_{t<-I_k} after W_O by linearity.
    """

    import torch

    if len(source_spans) != len(source_gates):
        raise ValueError("source_spans and source_gates must have equal length")
    output = attention_output
    query_length = int(attention_weights.shape[-2])
    key_length = int(attention_weights.shape[-1])
    query_offset = max(0, key_length - query_length)
    for (raw_start, raw_end), raw_gate in zip(source_spans, source_gates):
        gate = float(raw_gate)
        if abs(gate - 1.0) < 1e-12:
            continue
        start = max(0, min(int(raw_start), key_length))
        end = max(start, min(int(raw_end), key_length))
        if start == end:
            continue
        contribution = attention_weights[..., start:end] @ value_states[:, :, start:end, :]
        contribution = contribution.transpose(1, 2).contiguous()

        absolute_queries = query_offset + torch.arange(
            query_length, device=attention_weights.device, dtype=torch.long
        )
        receiver_mask = (absolute_queries < start) | (absolute_queries >= end)
        output = output + (gate - 1.0) * contribution * receiver_mask.view(1, -1, 1, 1)
    return output


def mean_completion_logprob(logits, input_ids, labels):
    import torch

    shifted_logits = logits[:, :-1, :].float()
    shifted_ids = input_ids[:, 1:]
    mask = labels[:, 1:] != -100
    target_logits = shifted_logits.gather(-1, shifted_ids.unsqueeze(-1)).squeeze(-1)
    log_normalizers = torch.logsumexp(shifted_logits, dim=-1)
    selected = target_logits - log_normalizers
    if not bool(mask.any()):
        raise ValueError("Completion produced no scoreable tokens")
    return selected[mask].mean()
