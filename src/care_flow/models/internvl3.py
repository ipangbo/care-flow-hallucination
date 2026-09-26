from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence


class TransformersVLMBackbone:
    """Native-Transformers backbone for direct multi-image inference.

    This class intentionally implements the vanilla generation path only.  It
    does not claim to implement CARE-Flow's architecture-specific, post-softmax
    source-value intervention.
    """

    def __init__(
        self,
        model_path: str = "OpenGVLab/InternVL3-8B-hf",
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        attention_implementation: str = "sdpa",
        max_image_side: int | None = 768,
        allow_download: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[torch_dtype]
        local_only = not allow_download
        self.torch = torch
        self.device = torch.device(device)
        self.max_image_side = max_image_side
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=True,
            local_files_only=local_only,
            low_cpu_mem_usage=True,
            attn_implementation=attention_implementation,
        )
        self.model.to(self.device)
        self.model.eval()
        if tokenizer is not None:
            # The checkpoint tokenizer advertises 8k although the released
            # language config supports 32k. Multi-image dynamic tiling can
            # legitimately exceed 8k, so use the model's actual limit.
            text_config = getattr(self.model.config, "text_config", self.model.config)
            max_positions = int(
                getattr(
                    text_config,
                    "max_position_embeddings",
                    tokenizer.model_max_length,
                )
            )
            tokenizer.model_max_length = max_positions
        if (
            getattr(self.model.generation_config, "pad_token_id", None) is None
            and tokenizer is not None
        ):
            self.model.generation_config.pad_token_id = tokenizer.pad_token_id

    def generate_batch(
        self,
        prompts: Sequence[str],
        image_path_batches: Sequence[Sequence[Path]],
        max_new_tokens: int = 8,
    ) -> list[dict[str, Any]]:
        if len(prompts) != len(image_path_batches):
            raise ValueError("One image-path list is required for every prompt")
        conversations: list[list[dict[str, Any]]] = []
        for prompt, paths in zip(prompts, image_path_batches, strict=True):
            images = [self._load_image(path) for path in paths]
            content: list[dict[str, Any]] = []
            for index, image in enumerate(images, start=1):
                content.append({"type": "text", "text": f"Image {index}:\n"})
                content.append({"type": "image", "image": image})
            content.append({"type": "text", "text": "\n" + prompt})
            conversations.append([{"role": "user", "content": content}])

        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        inputs = inputs.to(self.device)
        input_width = int(inputs["input_ids"].shape[1])
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            input_tokens = [input_width] * len(prompts)
        else:
            input_tokens = [int(value) for value in attention_mask.sum(dim=1).tolist()]

        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        generated_ids = output_ids[:, input_width:]
        responses = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        per_item_elapsed = elapsed / max(1, len(prompts))
        return [
            {
                "raw_response": response.strip(),
                "input_tokens": input_tokens[index],
                "generated_tokens": int(generated_ids[index].numel()),
                "elapsed_sec": round(per_item_elapsed, 4),
                "batch_elapsed_sec": round(elapsed, 4),
                "batch_size": len(prompts),
            }
            for index, response in enumerate(responses)
        ]

    def peak_allocated_gib(self) -> float | None:
        if self.device.type != "cuda":
            return None
        return self.torch.cuda.max_memory_allocated(self.device) / (1024**3)

    def reset_peak_memory_stats(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def _load_image(self, path: Path) -> Any:
        from PIL import Image, ImageOps

        # Museum scans can legitimately exceed Pillow's generic decompression
        # bomb warning threshold; paths are local benchmark assets.
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as source:
            if self.max_image_side:
                # JPEG draft asks libjpeg to decode a lower-resolution DCT
                # level. The exact final size is still enforced below.
                source.draft("RGB", (self.max_image_side, self.max_image_side))
            image = ImageOps.exif_transpose(source).convert("RGB")
        if self.max_image_side and max(image.size) > self.max_image_side:
            image.thumbnail(
                (self.max_image_side, self.max_image_side),
                Image.Resampling.LANCZOS,
            )
        return image


class InternVL3Backbone(TransformersVLMBackbone):
    """Backward-compatible name for the InternVL3 native backbone."""
