# CARE-Flow: Mitigating Inter-Image Hallucination in Multi-Image MLLMs

This directory contains only the code needed to reproduce the main CARE
experiments:

1. download and normalize a public multi-image benchmark;
2. run a zero-shot Vanilla backbone;
3. run CARE with the same Vanilla prediction as its paired initial hypothesis;
4. write per-example JSONL predictions and an aggregate JSON summary.

It intentionally excludes datasets, model weights, figures, ablations, dataset
construction code, and paper-writing utilities, so the directory can be
compressed and shared directly.

## Included models

| CLI name | Checkpoint | Note |
|---|---|---|
| `llava-next-7b` | [`llava-hf/llava-interleave-qwen-7b-hf`](https://huggingface.co/llava-hf/llava-interleave-qwen-7b-hf) | The multi-image LLaVA-NeXT-Interleave 7B release. |
| `qwen3.5-9b` | [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) | Native multimodal Qwen3.5 9B. |
| `internvl3-8b` | [`OpenGVLab/InternVL3-8B-hf`](https://huggingface.co/OpenGVLab/InternVL3-8B-hf) | Native Transformers InternVL3 8B. |

`--model-path /path/to/checkpoint` can override any registered checkpoint while
retaining the corresponding model adapter.

## Installation

Python 3.10+ and a CUDA-capable PyTorch installation are recommended. From this
directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Set `HF_TOKEN` before downloading if your Hugging Face account/token is needed:

```bash
export HF_TOKEN=...
```

The 7B--9B checkpoints normally require a GPU with roughly 20--24 GiB of free
memory in BF16/FP16. Reducing `--max-image-side` lowers visual-token memory.

## Obtaining the datasets

No dataset is included. `prepare_data.py` downloads the official release and
converts it to the common manifest expected by both runners:

```text
datasets/<name>/
├── examples.jsonl
└── images/ (or the official extracted image tree)
```

Prepare one dataset:

```bash
python prepare_data.py --benchmark mantis --output-root datasets
```

Prepare all six:

```bash
python prepare_data.py --benchmark all --output-root datasets
```

The official sources used by the converter are:

- **MMIU:** [`FanqingM/MMIU-Benchmark`](https://huggingface.co/datasets/FanqingM/MMIU-Benchmark). This is the largest
  download (the complete image archives are about 25 GB). For a partial local
  preparation, repeat `--mmiu-archive ARCHIVE.zip`; omit the option for all
  seven official archives. The shared three-backbone protocol keeps 2--8 image
  examples by default (`--mmiu-max-images 0` disables the upper cap); this
  avoids changing the context limit across backbones.
- **MuirBench:** [`MUIRBENCH/MUIRBENCH`](https://huggingface.co/datasets/MUIRBENCH/MUIRBENCH).
- **MIBench:** [`StarBottle/MIBench`](https://huggingface.co/datasets/StarBottle/MIBench).
- **Mantis:** [`TIGER-Lab/Mantis-Eval`](https://huggingface.co/datasets/TIGER-Lab/Mantis-Eval).
- **MIHBench:** [`MM-Hallu/MIHBench`](https://huggingface.co/datasets/MM-Hallu/MIHBench).
- **MIRB:** [`VLLMs/MIRB`](https://huggingface.co/datasets/VLLMs/MIRB); the converter uses the eleven
  image-based tasks from the official repository and excludes its text-only
  and concatenated-image ablations.

For a quick format check, add `--max-records 5`. This limits converted rows,
although archives required by an official release still have to be downloaded.
The converter validates that every referenced image exists and that every
multiple-choice instance has exactly one gold option.

The normalized manifest preserves official questions, options, answers, task
names, and image order. Public benchmarks without atom-level evidence labels
conservatively list all input images in `question_evidence_set`; this field is
used only for post-hoc source-recall diagnostics and is never exposed to CARE
during inference. Mantis and MIRB contain both multiple-choice and open-answer
items. Multiple-choice accuracy uses the option ID; open answers use normalized
exact/contained-answer matching.

## Run one main experiment

First run the paired Vanilla prediction:

```bash
python run_vanilla.py \
  --model qwen3.5-9b \
  --dataset-dir datasets/muirbench \
  --device cuda:0 \
  --output results/qwen3.5-9b_muirbench_vanilla.jsonl
```

Then run CARE using that prediction as the paired initial hypothesis:

```bash
python run_care.py \
  --model qwen3.5-9b \
  --dataset-dir datasets/muirbench \
  --device cuda:0 \
  --baseline-predictions results/qwen3.5-9b_muirbench_vanilla.jsonl \
  --output results/qwen3.5-9b_muirbench_care.jsonl
```

Both commands support `--resume`, `--limit`, `--sample-size`, `--num-shards`,
and `--shard-index`. A small end-to-end smoke run is therefore:

```bash
python run_vanilla.py --model internvl3-8b \
  --dataset-dir datasets/mantis --limit 2 --device cuda:0 \
  --output results/smoke_vanilla.jsonl

python run_care.py --model internvl3-8b \
  --dataset-dir datasets/mantis --limit 2 --device cuda:0 \
  --baseline-predictions results/smoke_vanilla.jsonl \
  --output results/smoke_care.jsonl
```

To run the complete 3-backbone x 6-dataset matrix sequentially:

```bash
chmod +x run_all.sh
DEVICE=cuda:0 ./run_all.sh datasets results
```

For multiple GPUs, launch the same command with different
`--num-shards/--shard-index` values and distinct output files, then concatenate
the JSONL shards before summarizing.

## What the CARE runner executes

For each example, `run_care.py` performs the main paper path only:

1. proposes a compact, source-indexed CARE-Bridge from the images;
2. certifies each atom by intervening on the corresponding image source's
   post-softmax value contribution in every supported language-attention block;
3. drops all image-derived state;
4. scores the candidate response against factual and null Bridges;
5. applies the paired verification/reranking rule and writes the result.

For Qwen3.5, the intervention is applied to its explicit full-attention blocks;
linear-attention blocks remain unchanged. The LLaVA-NeXT and InternVL3
checkpoints above both expose Qwen-family language attention through their
native Transformers implementations.

## Outputs

Each command writes:

- `OUTPUT.jsonl`: one complete record per example, including the Vanilla/CARE
  answer, correctness, Bridge atoms, certification effects, verification
  scores, source diagnostics, and latency;
- `OUTPUT.summary.json`: overall accuracy plus task/error groupings and the
  exact model/configuration used.

`run_vanilla.py` additionally reports parse failures, latency, and peak PyTorch
allocated GPU memory. Runtime failures are fatal by default; use
`--continue-on-error` only for long sharded runs where failed rows will be
inspected and retried later.

## Directory contents

```text
care_core_release/
├── README.md
├── pyproject.toml
├── requirements.txt
├── prepare_data.py
├── run_vanilla.py
├── run_care.py
├── run_all.sh
└── src/care_flow/
    ├── data.py
    ├── method.py
    ├── model_registry.py
    ├── text.py
    └── models/
        ├── base.py
        ├── internvl3.py
        └── qwen_vl.py
```
