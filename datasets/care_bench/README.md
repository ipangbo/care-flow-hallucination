# CARE-Bench annotations

This directory contains annotation-only releases for the three CARE-Bench
domains. No image binary is tracked in Git.

| Domain | Questions | Unique source images |
|---|---:|---:|
| COCO-VQA | 11,132 | 32,507 |
| Te Papa | 368 | 1,248 |
| Generated | 500 | 1,925 |

Each domain contains:

- `examples.jsonl`: questions, answer choices, gold answers, error annotations,
  evidence/source fields, reasoning operators, difficulty, and compact image
  references used directly by the runners;
- `images.jsonl`: the deduplicated image metadata and source/rights records.

The `local_file` entries are portable paths relative to the domain directory.
Place source images under that domain's `images/` directory. The repository
root README gives the COCO and Te Papa acquisition commands. Generated-domain
images are intentionally excluded from this Git release.
