from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence


LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BENCHMARKS = ("mmiu", "muirbench", "mibench", "mantis", "mihbench", "mirb")
MIRB_TASKS = (
    "analogy",
    "codeu",
    "count",
    "arxiv",
    "attribute",
    "visual_chain",
    "plot_code",
    "sightseeing",
    "food",
    "image_jigsaw",
    "3d_scene",
)
MMIU_ARCHIVES = (
    "Low-level-semantic.zip",
    "High-level-obj-semantic.zip",
    "High-level-sub-semantic.zip",
    "2D-spatial.zip",
    "3D-spatial.zip",
    "Discrete-temporal.zip",
    "Continuous-temporal.zip",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and normalize the six public multi-image benchmarks used by CARE."
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=(*BENCHMARKS, "all"),
        required=True,
        help="Repeat to prepare several datasets; use 'all' for all six.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optional per-benchmark limit for converter smoke tests.",
    )
    parser.add_argument(
        "--mmiu-archive",
        action="append",
        default=[],
        choices=MMIU_ARCHIVES,
        help="Download only selected MMIU image archives (repeatable).",
    )
    parser.add_argument(
        "--mmiu-max-images",
        type=int,
        default=8,
        help="Shared-backbone protocol cap for MMIU (default: 8; use 0 for no cap).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = list(BENCHMARKS) if "all" in args.benchmark else list(dict.fromkeys(args.benchmark))
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        records = PREPARERS[name](args.output_root, args.max_records, args)
        out_dir = args.output_root / name
        validate_records(records, out_dir)
        write_jsonl(out_dir / "examples.jsonl", records)
        print(json.dumps({"benchmark": name, "records": len(records), "output": str(out_dir)}))


def prepare_muirbench(root: Path, limit: int, _: argparse.Namespace) -> list[dict[str, Any]]:
    from datasets import load_dataset

    out_dir = root / "muirbench"
    rows = load_dataset("MUIRBENCH/MUIRBENCH", split="test", cache_dir=str(root / "_cache"))
    records: list[dict[str, Any]] = []
    for row in rows:
        if reached_limit(records, limit):
            break
        instance_id = f"muirbench_{row['idx']}"
        image_files = []
        for index, image in enumerate(row["image_list"], start=1):
            rel = f"images/{instance_id}_{index:02d}.jpg"
            save_image(image, out_dir / rel)
            image_files.append(rel)
        question, options = replace_sequential_placeholders(
            str(row["question"]), [str(value) for value in row["options"]]
        )
        records.append(
            make_record(
                instance_id=instance_id,
                benchmark="MuirBench",
                task=str(row.get("task") or "unknown"),
                question=question,
                options=options,
                gold=row["answer"],
                image_files=image_files,
            )
        )
    return records


def prepare_mibench(root: Path, limit: int, _: argparse.Namespace) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    out_dir = root / "mibench"
    raw_dir = root / "_downloads" / "mibench"
    files = (
        "multi_image_instruction.json",
        "multimodal_knowledge_seeking.json",
        "multimodal_in_context_learning.json",
    )
    for filename in files:
        hf_hub_download("StarBottle/MIBench", filename, repo_type="dataset", local_dir=raw_dir)
    archive = Path(
        hf_hub_download("StarBottle/MIBench", "image.zip", repo_type="dataset", local_dir=raw_dir)
    )
    extract_zip(archive, out_dir)
    records: list[dict[str, Any]] = []
    for filename in files:
        rows = json.loads((raw_dir / filename).read_text(encoding="utf-8"))
        for row in rows:
            if reached_limit(records, limit):
                return records
            if row.get("type") != "multiple-choice" or not row.get("options"):
                continue
            question, options = replace_sequential_placeholders(
                str(row.get("question") or ""), [str(value) for value in row["options"]]
            )
            image_files = [normalize_path(value) for value in row.get("image") or []]
            records.append(
                make_record(
                    instance_id=f"mibench_{row['id']}",
                    benchmark="MIBench",
                    task=str(row.get("task") or Path(filename).stem),
                    question=question,
                    options=options,
                    gold=row.get("answer"),
                    image_files=image_files,
                )
            )
    return records


def prepare_mmiu(root: Path, limit: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    out_dir = root / "mmiu"
    raw_dir = root / "_downloads" / "mmiu"
    parquet = Path(
        hf_hub_download(
            "FanqingM/MMIU-Benchmark", "all.parquet", repo_type="dataset", local_dir=raw_dir
        )
    )
    for filename in args.mmiu_archive or MMIU_ARCHIVES:
        archive = Path(
            hf_hub_download(
                "FanqingM/MMIU-Benchmark", filename, repo_type="dataset", local_dir=raw_dir
            )
        )
        extract_zip(archive, out_dir)
    records: list[dict[str, Any]] = []
    for row_index, row in pd.read_parquet(parquet).iterrows():
        if reached_limit(records, limit):
            break
        image_files = [resolve_existing_relative(out_dir, value) for value in row["input_image_path"]]
        if len(image_files) < 2 or (
            args.mmiu_max_images and len(image_files) > args.mmiu_max_images
        ):
            continue
        if not all((out_dir / value).is_file() for value in image_files):
            continue
        parsed = parse_labeled_options(str(row["options"]))
        if len(parsed) < 2:
            continue
        labels, options = zip(*parsed)
        records.append(
            make_record(
                instance_id=f"mmiu_{row_index:06d}",
                benchmark="MMIU",
                task=str(row.get("task") or "unknown"),
                question=str(row.get("question") or "").strip(),
                options=list(options),
                option_labels=list(labels),
                gold=row.get("output"),
                image_files=image_files,
            )
        )
    return records


def prepare_mantis(root: Path, limit: int, _: argparse.Namespace) -> list[dict[str, Any]]:
    from datasets import load_dataset

    out_dir = root / "mantis"
    rows = load_dataset("TIGER-Lab/Mantis-Eval", split="test", cache_dir=str(root / "_cache"))
    records: list[dict[str, Any]] = []
    for row in rows:
        if reached_limit(records, limit):
            break
        instance_id = f"mantis_{row['id']}"
        image_files = []
        for index, image in enumerate(row["images"], start=1):
            rel = f"images/{instance_id}_{index:02d}.png"
            save_image(image, out_dir / rel)
            image_files.append(rel)
        question = replace_question_placeholders(str(row["question"]))
        options = [strip_option_prefix(str(value)) for value in row.get("options") or []]
        records.append(
            make_record(
                instance_id=instance_id,
                benchmark="Mantis",
                task=str(row.get("category") or row.get("question_type") or "unknown"),
                question=question,
                options=options,
                gold=row.get("answer"),
                image_files=image_files,
                reference_answer=None if options else row.get("answer"),
            )
        )
    return records


def prepare_mihbench(root: Path, limit: int, _: argparse.Namespace) -> list[dict[str, Any]]:
    from datasets import load_dataset

    out_dir = root / "mihbench"
    rows = load_dataset("MM-Hallu/MIHBench", split="train", cache_dir=str(root / "_cache"))
    records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if reached_limit(records, limit):
            break
        image_files = []
        for image_index, image in enumerate(row["images"], start=1):
            rel = f"images/mihbench_{row_index:05d}_{image_index:02d}.jpg"
            save_image(image, out_dir / rel)
            image_files.append(rel)
        records.append(
            make_record(
                instance_id=f"mihbench_{row_index:05d}",
                benchmark="MIHBench",
                task=str(row.get("task") or "unknown"),
                question=str(row.get("question") or "").strip(),
                options=["Yes", "No"],
                gold=str(row.get("label") or "").strip(),
                image_files=image_files,
            )
        )
    return records


def prepare_mirb(root: Path, limit: int, _: argparse.Namespace) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    out_dir = root / "mirb"
    raw_dir = root / "_downloads" / "mirb"
    for task in MIRB_TASKS:
        hf_hub_download("VLLMs/MIRB", f"{task}.json", repo_type="dataset", local_dir=raw_dir)
    archive = Path(
        hf_hub_download("VLLMs/MIRB", "images.zip", repo_type="dataset", local_dir=raw_dir)
    )
    extract_zip(archive, out_dir)
    records: list[dict[str, Any]] = []
    for task in MIRB_TASKS:
        rows = json.loads((raw_dir / f"{task}.json").read_text(encoding="utf-8"))
        for row_index, row in enumerate(rows):
            if reached_limit(records, limit):
                return records
            question, labels, options = split_embedded_options(str(row["questions"]))
            gold = row.get("answers")
            reference_answer = None if options else gold
            records.append(
                make_record(
                    instance_id=f"mirb_{task}_{row_index:04d}",
                    benchmark="MIRB",
                    task=task,
                    question=question,
                    options=options,
                    option_labels=labels or None,
                    gold=gold,
                    image_files=[normalize_path(value) for value in row["images"]],
                    reference_answer=reference_answer,
                )
            )
    return records


def make_record(
    *,
    instance_id: str,
    benchmark: str,
    task: str,
    question: str,
    options: Sequence[str],
    gold: Any,
    image_files: Sequence[str],
    option_labels: Sequence[str] | None = None,
    reference_answer: Any | None = None,
) -> dict[str, Any]:
    labels = list(option_labels or LABELS[: len(options)])
    if len(labels) != len(options):
        raise ValueError(f"{instance_id}: option label/text counts differ")
    candidates: list[dict[str, Any]] = []
    gold_id = resolve_gold_candidate(gold, labels, options) if options else None
    for index, (label, option) in enumerate(zip(labels, options, strict=True)):
        candidate_id = LABELS[index]
        candidates.append(
            {
                "candidate_id": candidate_id,
                "text": str(option).strip(),
                "is_correct": candidate_id == gold_id,
                "error_label_if_selected": None,
            }
        )
    images = [
        {
            "display_index": index,
            "image_uid": f"{instance_id}_I{index}",
            "source": benchmark,
            "local_file": normalize_path(path),
        }
        for index, path in enumerate(image_files, start=1)
    ]
    reference = None if candidates else str(reference_answer if reference_answer is not None else gold).strip()
    return {
        "instance_id": instance_id,
        "split": "test",
        "source_mix": [benchmark],
        "task_type": task,
        "error_focus": [],
        "answer_type": "multiple_choice" if candidates else "open",
        "num_images": len(images),
        "num_candidates": len(candidates),
        "images": images,
        "question": question.strip(),
        "candidates": candidates,
        "gold_candidate_id": gold_id,
        "reference_answer": reference,
        # Public benchmarks do not expose atom-level gold evidence. CARE never
        # consumes this field during inference; all sources are listed only so
        # output diagnostics have a conservative reference set.
        "question_evidence_set": [f"I{index}" for index in range(1, len(images) + 1)],
    }


def resolve_gold_candidate(gold: Any, labels: Sequence[str], options: Sequence[str]) -> str:
    text = str(gold).strip()
    normalized_labels = [str(value).strip().upper() for value in labels]
    upper = text.upper().strip("()[] .:")
    if upper in normalized_labels:
        return LABELS[normalized_labels.index(upper)]
    letter = re.fullmatch(r"(?:OPTION\s*)?([A-Z])", upper)
    if letter and letter.group(1) in normalized_labels:
        return LABELS[normalized_labels.index(letter.group(1))]
    if re.fullmatch(r"\d+", text):
        number = int(text)
        if 1 <= number <= len(options):
            return LABELS[number - 1]
        if 0 <= number < len(options):
            return LABELS[number]
    normalized_gold = normalize_text(text)
    matches = [index for index, value in enumerate(options) if normalize_text(str(value)) == normalized_gold]
    if len(matches) == 1:
        return LABELS[matches[0]]
    raise ValueError(f"Cannot map gold answer {gold!r} to options {list(options)!r}")


def split_embedded_options(text: str) -> tuple[str, list[str], list[str]]:
    text = text.strip().strip('"')
    pattern = re.compile(
        r"(?ms)^\s*([A-Z]|\d{1,2})[\.)]\s+(.+?)(?=^\s*(?:[A-Z]|\d{1,2})[\.)]\s+|\Z)"
    )
    matches = list(pattern.finditer(text))
    if len(matches) >= 2:
        question = text[: matches[0].start()].strip()
        return question, [match.group(1) for match in matches], [match.group(2).strip() for match in matches]
    marker = re.search(r"(?i)\b(?:your options are|options)\s*:\s*", text)
    if marker:
        tail = text[marker.end() :]
        inline = list(
            re.finditer(r"(?s)(?:^|,\s*)([A-Z])\.\s*(.+?)(?=,\s*[A-Z]\.\s*|\Z)", tail)
        )
        if len(inline) >= 2:
            return (
                text[: marker.start()].strip(),
                [match.group(1) for match in inline],
                [match.group(2).strip().rstrip(".") for match in inline],
            )
    return text, [], []


def parse_labeled_options(text: str) -> list[tuple[str, str]]:
    return [
        (match.group(1), match.group(2).strip())
        for match in re.finditer(r"(?m)^\s*([A-Z])\s*[:\.)]\s*(.+?)\s*$", text)
    ]


def replace_sequential_placeholders(question: str, options: Sequence[str]) -> tuple[str, list[str]]:
    index = 1

    def replace(_: re.Match[str]) -> str:
        nonlocal index
        value = f"Image {index}"
        index += 1
        return value

    question = re.sub(r"<image>", replace, question)
    return question, [re.sub(r"<image>", replace, value) for value in options]


def replace_question_placeholders(question: str) -> str:
    index = 0

    def replace(_: re.Match[str]) -> str:
        nonlocal index
        index += 1
        return f"Image {index}"

    return re.sub(r"<image>", replace, question)


def strip_option_prefix(text: str) -> str:
    return re.sub(r"^\s*(?:\([A-Z]\)|[A-Z][\.)])\s*", "", text).strip()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.casefold())).strip()


def normalize_path(value: Any) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def resolve_existing_relative(root: Path, value: Any) -> str:
    normalized = normalize_path(value)
    candidates = [normalized]
    parts = normalized.split("/")
    candidates.extend("/".join(parts[offset:]) for offset in range(1, min(3, len(parts))))
    return next((item for item in candidates if (root / item).is_file()), normalized)


def save_image(image: Any, path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, quality=92)


def extract_zip(archive: Path, out_dir: Path) -> None:
    marker = out_dir / f".extracted_{archive.stem}"
    if marker.is_file():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (out_dir / member.filename).resolve()
            if out_dir.resolve() not in target.parents and target != out_dir.resolve():
                raise ValueError(f"Unsafe archive member: {member.filename}")
        handle.extractall(out_dir)
    marker.write_text("ok\n", encoding="utf-8")


def validate_records(records: Sequence[dict[str, Any]], dataset_dir: Path) -> None:
    if not records:
        raise RuntimeError(f"No records were prepared in {dataset_dir}")
    seen: set[str] = set()
    for row in records:
        instance_id = str(row["instance_id"])
        if instance_id in seen:
            raise ValueError(f"Duplicate instance_id: {instance_id}")
        seen.add(instance_id)
        if len(row["images"]) < 2:
            raise ValueError(f"{instance_id}: expected at least two images")
        for image in row["images"]:
            path = dataset_dir / image["local_file"]
            if not path.is_file():
                raise FileNotFoundError(f"{instance_id}: missing image {path}")
        candidates = row["candidates"]
        if candidates:
            ids = [value["candidate_id"] for value in candidates]
            if ids != list(LABELS[: len(ids)]):
                raise ValueError(f"{instance_id}: candidates are not contiguous from A")
            if sum(value["is_correct"] for value in candidates) != 1:
                raise ValueError(f"{instance_id}: expected exactly one correct option")
        elif not row.get("reference_answer"):
            raise ValueError(f"{instance_id}: open question has no reference answer")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def reached_limit(records: Sequence[Any], limit: int) -> bool:
    return bool(limit and len(records) >= limit)


PREPARERS = {
    "muirbench": prepare_muirbench,
    "mibench": prepare_mibench,
    "mmiu": prepare_mmiu,
    "mantis": prepare_mantis,
    "mihbench": prepare_mihbench,
    "mirb": prepare_mirb,
}


if __name__ == "__main__":
    main()
