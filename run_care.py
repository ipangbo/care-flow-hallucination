from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from care_flow.data import read_jsonl
from care_flow.model_registry import MODELS, get_model
from care_flow.method import (
    CertificationConfig,
    FlowConfig,
    VerificationConfig,
    grouped_summaries,
    run_care_flow_on_row,
    select_rows,
    summarize_records,
)
from care_flow.models.qwen_vl import QwenVLBackend


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run paper CARE-Flow: pre-response Bridge proposal, exact source-grouped "
            "attention value-output certification, hard reset, and paired null-Bridge verification."
        )
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--examples-name", default="examples.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--source-filter", nargs="*", default=None)
    parser.add_argument(
        "--ids-file", default=None, help="Optional JSON list of instance IDs."
    )
    parser.add_argument(
        "--baseline-predictions",
        nargs="*",
        default=None,
        help=(
            "Optional Vanilla JSONL files. Their prediction field becomes the "
            "paired initial hypothesis, including null predictions for parse "
            "failures; candidate likelihoods are still computed for CARE's "
            "residual verifier."
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=17)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)

    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional local checkpoint or Hugging Face ID overriding --model.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        default=None,
        help="Defaults to the precision registered for the selected model.",
    )
    parser.add_argument("--max-image-side", type=int, default=768)

    parser.add_argument("--epsilon", type=float, default=0.9)
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--kappa-min", type=float, default=0.0)
    parser.add_argument("--epsilon0", type=float, default=1e-8)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=6.0)
    parser.add_argument("--max-atoms", type=int, default=5)
    parser.add_argument("--proposal-max-new-tokens", type=int, default=320)
    parser.add_argument("--answer-max-new-tokens", type=int, default=96)
    parser.add_argument("--residual-rerank-weight", type=float, default=1.5)
    parser.add_argument("--factual-support-weight", type=float, default=2.0 / 3.0)
    parser.add_argument("--residual-rerank-margin", type=float, default=0.4)
    parser.add_argument("--baseline-margin-cap", type=float, default=0.75)
    parser.add_argument(
        "--selective-only",
        action="store_true",
        help="Abstain when no candidate passes verification instead of retaining the initial hypothesis.",
    )
    parser.add_argument(
        "--joint-proposal",
        action="store_true",
        help="Use one joint multi-image Bridge proposal instead of source-isolated acquisition.",
    )
    parser.add_argument(
        "--hard-unknown-only",
        action="store_true",
        help="Ablation: replace every failed atom with unknown instead of retaining a tentative value.",
    )

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    started = time.time()
    model_spec = get_model(args.model)
    model_path = args.model_path or model_spec.model_id
    torch_dtype = args.torch_dtype or model_spec.dtype
    dataset_dir = Path(args.dataset_dir).resolve()
    examples_path = dataset_dir / args.examples_name
    rows = read_jsonl(examples_path)
    if args.ids_file:
        selected_ids = set(
            json.loads(Path(args.ids_file).read_text(encoding="utf-8"))
        )
        rows = [
            row
            for row in rows
            if str(row.get("instance_id")) in selected_ids
        ]
    selected = select_rows(
        rows,
        splits=set(args.splits) if args.splits else None,
        source_filter=set(args.source_filter) if args.source_filter else None,
        limit=args.limit,
        offset=args.offset,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    if not selected:
        raise RuntimeError("No benchmark examples were selected")
    if args.baseline_predictions:
        baseline_by_id = load_baseline_predictions(args.baseline_predictions)
        missing = [
            str(row.get("instance_id"))
            for row in selected
            if str(row.get("instance_id")) not in baseline_by_id
        ]
        if missing:
            raise RuntimeError(
                f"Missing paired Vanilla predictions for {len(missing)} selected rows; "
                f"first IDs: {missing[:5]}"
            )
        paired_rows = []
        for row in selected:
            paired = baseline_by_id[str(row.get("instance_id"))]
            if row.get("candidates"):
                paired_rows.append(
                    dict(row, _care_initial_candidate_id=paired["candidate_id"])
                )
            else:
                paired_rows.append(dict(row, _care_initial_answer=paired["answer"]))
        selected = paired_rows

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    loaded_records = load_existing_records(output_path) if args.resume else []
    # Runtime failures are retryable.  Remove them before resuming so a
    # transient OOM or corrupt decode cannot permanently occupy an instance
    # ID or create duplicate records when the run is restarted.
    records = [record for record in loaded_records if not record.get("runtime_error")]
    if args.resume and len(records) != len(loaded_records):
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    done_ids = {str(record.get("instance_id")) for record in records}
    pending = [row for row in selected if str(row.get("instance_id")) not in done_ids]

    certification = CertificationConfig(
        epsilon=args.epsilon,
        kappa=args.kappa,
        rho=args.rho,
        kappa_min=args.kappa_min,
        epsilon0=args.epsilon0,
    )
    verification = VerificationConfig(gamma=args.gamma, eta=args.eta)
    flow = FlowConfig(
        max_atoms=args.max_atoms,
        proposal_max_new_tokens=args.proposal_max_new_tokens,
        answer_max_new_tokens=args.answer_max_new_tokens,
        retain_initial_on_rejection=not args.selective_only,
        source_isolated_proposal=not args.joint_proposal,
        retain_tentative_atoms=not args.hard_unknown_only,
        residual_rerank_weight=args.residual_rerank_weight,
        factual_support_weight=args.factual_support_weight,
        residual_rerank_margin=args.residual_rerank_margin,
        baseline_margin_cap=args.baseline_margin_cap,
    )
    backend = QwenVLBackend(
        model_path=model_path,
        device=args.device,
        torch_dtype=torch_dtype,
        max_image_side=args.max_image_side,
    )

    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        iterator = tqdm(
            pending,
            desc=f"CARE-Flow:{dataset_dir.name}",
            disable=args.quiet,
            dynamic_ncols=True,
        )
        for row in iterator:
            try:
                record = run_care_flow_on_row(
                    row, backend, dataset_dir, certification, verification, flow
                )
                record.update(
                    {
                        "model": model_spec.key,
                        "model_label": model_spec.label,
                        "model_path": model_path,
                    }
                )
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                record = {
                    "instance_id": row.get("instance_id"),
                    "split": row.get("split"),
                    "source_mix": row.get("source_mix"),
                    "task_type": row.get("task_type"),
                    "runtime_error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "model": model_spec.key,
                    "model_label": model_spec.label,
                    "model_path": model_path,
                }
                release_cuda_cache()
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if not args.quiet and not record.get("runtime_error"):
                iterator.set_postfix(
                    coverage=f"{running_rate(records, lambda item: not item.get('care_abstained', True)):.3f}",
                    selective=f"{running_selective_accuracy(records):.3f}",
                    last_sec=record.get("elapsed_sec"),
                )

    summary = {
        "dataset_dir": str(dataset_dir),
        "examples": str(examples_path),
        "selected_count": len(selected),
        "completed_count": len(records),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "model": model_spec.key,
        "model_label": model_spec.label,
        "model_path": model_path,
        "torch_dtype": torch_dtype,
        "baseline_predictions": args.baseline_predictions,
        "certification": asdict(certification),
        "verification": asdict(verification),
        "flow": asdict(flow),
        "overall": summarize_records(records),
        "by_task_type": grouped_summaries(records, "task_type"),
        "by_error_focus": grouped_summaries(records, "error_focus"),
        "wall_time_sec": round(time.time() - started, 3),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(f"[CARE-Flow] records: {output_path}")
    print(f"[CARE-Flow] summary: {summary_path}")


def load_existing_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_baseline_predictions(paths: list[str]) -> dict[str, dict[str, str | None]]:
    predictions: dict[str, dict[str, str | None]] = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        for record in load_existing_records(path):
            instance_id = str(record.get("instance_id") or record.get("sample_id"))
            if not instance_id or instance_id == "None":
                raise ValueError(f"Baseline record without an instance ID in {path}")
            candidate_id = record.get("prediction")
            if candidate_id is not None:
                candidate_id = str(candidate_id)
            answer = record.get("prediction_text") or record.get("raw_response")
            if answer is not None:
                answer = str(answer).strip() or None
            value = {"candidate_id": candidate_id, "answer": answer}
            if instance_id in predictions and predictions[instance_id] != value:
                raise ValueError(f"Conflicting Vanilla prediction for {instance_id}")
            predictions[instance_id] = value
    return predictions


def running_rate(records: list[dict], predicate) -> float:
    valid = [record for record in records if not record.get("runtime_error")]
    return sum(bool(predicate(record)) for record in valid) / max(1, len(valid))


def running_selective_accuracy(records: list[dict]) -> float:
    accepted = [
        record
        for record in records
        if not record.get("runtime_error") and not record.get("care_abstained", True)
    ]
    return sum(bool(record.get("care_correct")) for record in accepted) / max(1, len(accepted))


def release_cuda_cache() -> None:
    """Let a continued shard recover after an isolated CUDA allocation failure."""

    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # Error reporting must not be replaced by best-effort cleanup failures.
        pass


if __name__ == "__main__":
    main()
