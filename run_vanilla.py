from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from care_flow.data import read_jsonl
from care_flow.method import resolve_image_paths
from care_flow.model_registry import MODELS, get_model
from care_flow.models.internvl3 import TransformersVLMBackbone
from care_flow.text import exact_or_contains


LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LEGACY_ERROR_TYPES = {
    "WS": "operand-sub",
    "WI": "operator-error",
    "MI": "operand-omit",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a vanilla multi-image backbone on a CARE-Bench JSONL release."
    )
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--examples-name", default="examples.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional local checkpoint or Hugging Face ID overriding --model.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default=None,
        help="Defaults to the precision registered for the selected model.",
    )
    parser.add_argument("--attention-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument("--max-image-side", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260915)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Never download model files; require an existing local cache/path.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def build_prompt(row: dict[str, Any]) -> str:
    candidates = sorted(row.get("candidates") or [], key=lambda item: item["candidate_id"])
    if not candidates:
        return (
            f"{str(row.get('question') or '').strip()}\n\n"
            "Give only the shortest complete answer. Do not explain."
        )
    expected_ids = [chr(ord("A") + index) for index in range(len(candidates))]
    if len(candidates) < 2 or [candidate.get("candidate_id") for candidate in candidates] != expected_ids:
        raise ValueError(f"{row.get('instance_id')}: candidates must be contiguous from A")
    options = "\n".join(f"{candidate['candidate_id']}. {candidate['text']}" for candidate in candidates)
    answer_space = ", ".join(expected_ids)
    return (
        f"{str(row.get('question') or '').strip()}\n\n{options}\n\n"
        f"Answer with exactly one uppercase letter: {answer_space}. Do not explain."
    )


def parse_option(raw_text: str, valid_letters: Sequence[str] = LETTERS) -> str | None:
    text = raw_text.strip().upper()
    if not text:
        return None
    allowed = "".join(re.escape(letter) for letter in valid_letters)
    explicit = re.search(
        rf"\b(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:|=)?\s*([{allowed}])\b",
        text,
    )
    if explicit and explicit.group(1) in valid_letters:
        return explicit.group(1)
    isolated = set(re.findall(rf"(?<![A-Z])([{allowed}])(?![A-Z])", text))
    return next(iter(isolated)) if len(isolated) == 1 else None


def candidate_error_type(candidate: dict[str, Any]) -> str | None:
    raw = candidate.get("error_type_if_selected")
    if raw is None:
        raw = candidate.get("error_label_if_selected")
    if raw is None:
        return None
    value = str(raw).strip()
    return LEGACY_ERROR_TYPES.get(value.upper(), value)


def select_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = rows[args.offset :]
    if args.sample_size:
        rng = random.Random(args.sample_seed)
        selected = rng.sample(selected, min(args.sample_size, len(selected)))
        selected.sort(key=lambda row: str(row.get("instance_id")))
    selected = [
        row
        for index, row in enumerate(selected)
        if index % args.num_shards == args.shard_index
    ]
    if args.limit:
        selected = selected[: args.limit]
    return selected


def chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def run_batch(
    rows: Sequence[dict[str, Any]],
    backend: TransformersVLMBackbone,
    dataset_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    prompts = [build_prompt(row) for row in rows]
    image_paths = [resolve_image_paths(row, dataset_dir) for row in rows]
    generations = backend.generate_batch(
        prompts,
        image_paths,
        max_new_tokens=args.max_new_tokens,
    )
    return [
        build_record(row, generation, args)
        for row, generation in zip(rows, generations, strict=True)
    ]


def build_record(
    row: dict[str, Any], generation: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    candidates = sorted(row["candidates"], key=lambda item: item["candidate_id"])
    candidate_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    raw_response = str(generation["raw_response"])
    prediction = parse_option(raw_response, tuple(candidate_by_id)) if candidates else raw_response.strip()
    gold_value = row.get("gold_candidate_id")
    gold = str(gold_value) if gold_value is not None else None
    reference_answer = str(row.get("reference_answer") or "").strip() or None
    selected = candidate_by_id.get(prediction)
    correct = (
        prediction == gold
        if candidates
        else bool(reference_answer and exact_or_contains(raw_response, reference_answer))
    )
    return {
        "instance_id": row.get("instance_id"),
        "split": row.get("split"),
        "source_mix": row.get("source_mix"),
        "task_type": row.get("task_type"),
        "task_type_primary": row.get("task_type_primary", row.get("task_type")),
        "num_images": row.get("num_images", len(row.get("images") or [])),
        "requires_all_images": row.get("requires_all_images"),
        "error_focus": row.get("error_focus"),
        "model": args.model,
        "model_id": args.resolved_model_path,
        "model_label": args.resolved_model_label,
        "question": row.get("question"),
        "candidates": [
            {
                "candidate_id": candidate.get("candidate_id"),
                "text": candidate.get("text"),
                "error_type_if_selected": candidate_error_type(candidate),
            }
            for candidate in candidates
        ],
        "gold_candidate_id": gold,
        "reference_answer": reference_answer,
        "prediction": prediction,
        "prediction_text": selected.get("text") if selected else raw_response.strip(),
        "raw_response": raw_response,
        "correct": correct,
        "chosen_error_type": (
            candidate_error_type(selected)
            if selected
            else ("open_answer_error" if not candidates else "parse_failure")
        ),
        **{key: generation[key] for key in generation},
    }


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float] | None:
    if not trials:
        return None
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return [center - radius, center + radius]


def rate_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    runtime_errors = sum("runtime_error" in record for record in records)
    correct = sum(bool(record.get("correct")) for record in records)
    parse_failures = sum(
        record.get("prediction") is None and "runtime_error" not in record
        for record in records
    )
    return {
        "n": len(records),
        "correct": correct,
        "accuracy": correct / len(records) if records else None,
        "accuracy_95ci_wilson": wilson_interval(correct, len(records)),
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / len(records) if records else None,
        "runtime_errors": runtime_errors,
    }


def grouped_summary(records: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(key))].append(record)
    return {name: rate_summary(group) for name, group in sorted(groups.items())}


def latency_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = [float(record["elapsed_sec"]) for record in records if "elapsed_sec" in record]
    if not values:
        return {"mean_sec": None, "median_sec": None, "p95_sec": None}
    values.sort()
    return {
        "mean_sec": statistics.mean(values),
        "median_sec": statistics.median(values),
        "p95_sec": values[max(0, math.ceil(0.95 * len(values)) - 1)],
    }


def summarize_records(
    records: Sequence[dict[str, Any]], protocol: dict[str, Any], wall_time_sec: float | None = None,
    peak_gib: float | None = None,
) -> dict[str, Any]:
    incorrect = [
        record for record in records
        if "runtime_error" not in record and not record.get("correct")
    ]
    return {
        "protocol": protocol,
        "overall": rate_summary(records),
        "by_task_type": grouped_summary(records, "task_type"),
        "by_num_images": grouped_summary(records, "num_images"),
        "by_requires_all_images": grouped_summary(records, "requires_all_images"),
        "incorrect_choice_error_counts": dict(
            sorted(Counter(record.get("chosen_error_type") for record in incorrect).items())
        ),
        "latency": latency_summary(records),
        "wall_time_sec": wall_time_sec,
        "peak_torch_allocated_gib": peak_gib,
    }


def protocol_from_args(args: argparse.Namespace, dataset_dir: Path, examples_path: Path) -> dict[str, Any]:
    return {
        "dataset_dir": str(dataset_dir),
        "examples": str(examples_path),
        "model": args.model,
        "model_id": args.resolved_model_path,
        "model_label": args.resolved_model_label,
        "zero_shot": True,
        "decoding": "greedy",
        "answer_space": list(LETTERS),
        "parse_failures_count_as_wrong": True,
        "max_image_side": args.max_image_side,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "attention_implementation": args.attention_implementation,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "sample_size": args.sample_size,
        "sample_seed": args.sample_seed,
    }


def write_summary(
    records: Sequence[dict[str, Any]], args: argparse.Namespace, dataset_dir: Path,
    examples_path: Path, wall_time_sec: float, peak_gib: float | None,
) -> Path:
    summary = summarize_records(
        records,
        protocol_from_args(args, dataset_dir, examples_path),
        wall_time_sec=wall_time_sec,
        peak_gib=peak_gib,
    )
    path = args.output.with_suffix(".summary.json")
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    model_spec = get_model(args.model)
    args.resolved_model_path = args.model_path or model_spec.model_id
    args.resolved_model_label = model_spec.label
    args.dtype = args.dtype or model_spec.dtype
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    dataset_dir = args.dataset_dir.resolve()
    examples_path = dataset_dir / args.examples_name
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = select_rows(read_jsonl(examples_path), args)
    if not rows:
        raise RuntimeError("No examples selected")
    existing = load_existing(args.output) if args.resume else []
    done = {str(record.get("instance_id")) for record in existing}
    pending = [row for row in rows if str(row.get("instance_id")) not in done]

    started = time.perf_counter()
    backend = TransformersVLMBackbone(
        model_path=args.resolved_model_path,
        device=args.device,
        torch_dtype=args.dtype,
        attention_implementation=args.attention_implementation,
        max_image_side=args.max_image_side,
        allow_download=not args.local_files_only,
    )
    backend.reset_peak_memory_stats()
    records = list(existing)
    mode = "a" if args.resume and args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        completed_now = 0
        for batch in chunks(pending, args.batch_size):
            try:
                batch_records = run_batch(batch, backend, dataset_dir, args)
            except Exception as error:
                if not args.continue_on_error:
                    raise
                batch_records = []
                for row in batch:
                    try:
                        batch_records.extend(run_batch([row], backend, dataset_dir, args))
                    except Exception as item_error:
                        batch_records.append(
                            {
                                "instance_id": row.get("instance_id"),
                                "task_type": row.get("task_type"),
                                "num_images": row.get("num_images", len(row.get("images") or [])),
                                "requires_all_images": row.get("requires_all_images"),
                                "model": args.model,
                                "model_id": args.resolved_model_path,
                                "model_label": args.resolved_model_label,
                                "gold_candidate_id": row.get("gold_candidate_id"),
                                "prediction": None,
                                "correct": False,
                                "chosen_error_type": "runtime_error",
                                "runtime_error": f"{type(item_error).__name__}: {item_error}",
                                "batch_runtime_error": f"{type(error).__name__}: {error}",
                                "traceback": traceback.format_exc(),
                            }
                        )
            for record in batch_records:
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                completed_now += 1
            handle.flush()
            if not args.quiet:
                running = rate_summary(records)
                print(
                    f"[{args.resolved_model_label}:shard{args.shard_index}] "
                    f"{completed_now}/{len(pending)} accuracy={running['accuracy']} "
                    f"runtime_errors={running['runtime_errors']}",
                    flush=True,
                )

    wall_time = time.perf_counter() - started
    summary_path = write_summary(
        records, args, dataset_dir, examples_path, wall_time, backend.peak_allocated_gib()
    )
    print(json.dumps(rate_summary(records), indent=2))
    print(f"records={args.output}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
