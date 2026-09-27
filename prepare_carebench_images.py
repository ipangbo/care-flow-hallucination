from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
from pathlib import Path


DOMAINS = ("coco_vqa", "te_papa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the public source images referenced by CARE-Bench annotations."
    )
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/care_bench"))
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Extracted source-image tree. Required for local COCO materialization.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download each image from original_url (recommended for Te Papa only).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy local source files instead of creating relative symbolic links.",
    )
    return parser.parse_args()


def read_registry(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def index_source_files(source_root: Path) -> dict[str, Path]:
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    index: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in source_root.rglob("*"):
        if not path.is_file():
            continue
        if path.name in index and index[path.name] != path:
            duplicates.add(path.name)
        else:
            index[path.name] = path.resolve()
    if duplicates:
        preview = sorted(duplicates)[:5]
        raise RuntimeError(
            f"Source tree contains duplicate basenames; first duplicates: {preview}"
        )
    return index


def materialize_local(
    registry: list[dict], source_root: Path, domain_dir: Path, copy_files: bool
) -> None:
    index = index_source_files(source_root)
    missing = [
        Path(str(image["local_file"])).name
        for image in registry
        if Path(str(image["local_file"])).name not in index
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} referenced images are missing; first names: {missing[:5]}"
        )
    for image in registry:
        destination = domain_dir / str(image["local_file"])
        if destination.exists() or destination.is_symlink():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = index[destination.name]
        if copy_files:
            shutil.copy2(source, destination)
        else:
            destination.symlink_to(os.path.relpath(source, destination.parent))


def download_registry(registry: list[dict], domain_dir: Path) -> None:
    for index, image in enumerate(registry, start=1):
        destination = domain_dir / str(image["local_file"])
        if destination.is_file():
            continue
        url = str(image.get("original_url") or "").strip()
        if not url:
            raise ValueError(f"Missing original_url for {image.get('image_uid')}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "CARE-Bench research dataset downloader/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
            "wb"
        ) as handle:
            shutil.copyfileobj(response, handle)
        temporary.replace(destination)
        if index % 100 == 0:
            print(f"downloaded {index}/{len(registry)}")


def main() -> None:
    args = parse_args()
    domain_dir = (args.dataset_root / args.domain).resolve()
    registry_path = domain_dir / "images.jsonl"
    if not registry_path.is_file():
        raise FileNotFoundError(registry_path)
    registry = read_registry(registry_path)
    if args.download:
        download_registry(registry, domain_dir)
    else:
        if args.source_root is None:
            raise ValueError("--source-root is required unless --download is used")
        materialize_local(registry, args.source_root.resolve(), domain_dir, args.copy)
    print(f"ready: {len(registry)} images for {args.domain} in {domain_dir / 'images'}")


if __name__ == "__main__":
    main()
