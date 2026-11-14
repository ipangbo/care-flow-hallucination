#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODEL_ID = "llava-hf/llava-interleave-qwen-7b-hf"
MODEL_NAME = "LLaVA-Interleave-Qwen-7B"
MODEL_IDS = {MODEL_ID}
SHAPE = (32, 32)  # The Qwen language decoder, not the vision encoder.

# Original HCA data supplied by the user (reference only; not loaded automatically).
# Rows are decoder layers; columns are original attention heads, before reordering.
# hca = np.array([
#     [0.29, 0.25, 0.31, 0.28, 0.29, 0.3, 0.29, 0.27, 0.24, 0.34, 0.28, 0.28, 0.37, 0.32, 0.28, 0.32, 0.26, 0.3, 0.29, 0.3, 0.32, 0.27, 0.3, 0.3, 0.29, 0.31, 0.32, 0.29, 0.25, 0.31, 0.31, 0.3],
#     [0.29, 0.26, 0.2, 0.35, 0.3, 0.3, 0.33, 0.3, 0.29, 0.27, 0.25, 0.29, 0.3, 0.33, 0.3, 0.29, 0.34, 0.38, 0.22, 0.33, 0.29, 0.3, 0.29, 0.29, 0.29, 0.29, 0.29, 0.35, 0.29, 0.3, 0.29, 0.34],
#     [0.32, 0.3, 0.29, 0.29, 0.29, 0.3, 0.25, 0.3, 0.29, 0.3, 0.29, 0.3, 0.3, 0.36, 0.35, 0.29, 0.32, 0.33, 0.29, 0.29, 0.29, 0.3, 0.3, 0.29, 0.29, 0.31, 0.32, 0.38, 0.29, 0.31, 0.29, 0.32],
#     [0.2, 0.3, 0.31, 0.29, 0.29, 0.3, 0.33, 0.3, 0.29, 0.3, 0.3, 0.3, 0.3, 0.23, 0.27, 0.29, 0.29, 0.3, 0.29, 0.29, 0.29, 0.38, 0.3, 0.25, 0.29, 0.29, 0.3, 0.25, 0.29, 0.26, 0.29, 0.33],
#     [0.25, 0.35, 0.4, 0.32, 0.24, 0.24, 0.3, 0.35, 0.29, 0.29, 0.3, 0.26, 0.3, 0.32, 0.29, 0.33, 0.29, 0.28, 0.33, 0.29, 0.36, 0.29, 0.32, 0.33, 0.3, 0.3, 0.36, 0.29, 0.29, 0.3, 0.29, 0.31],
#     [0.22, 0.27, 0.3, 0.27, 0.3, 0.23, 0.38, 0.27, 0.3, 0.29, 0.36, 0.26, 0.3, 0.31, 0.31, 0.32, 0.31, 0.28, 0.34, 0.32, 0.3, 0.29, 0.33, 0.33, 0.29, 0.32, 0.34, 0.3, 0.34, 0.25, 0.3, 0.29],
#     [0.35, 0.3, 0.27, 0.33, 0.3, 0.35, 0.29, 0.27, 0.31, 0.27, 0.32, 0.33, 0.27, 0.33, 0.3, 0.27, 0.22, 0.3, 0.29, 0.34, 0.33, 0.27, 0.32, 0.25, 0.28, 0.3, 0.28, 0.29, 0.26, 0.33, 0.29, 0.27],
#     [0.3, 0.33, 0.29, 0.24, 0.3, 0.32, 0.31, 0.31, 0.26, 0.28, 0.31, 0.3, 0.32, 0.33, 0.32, 0.35, 0.33, 0.26, 0.35, 0.34, 0.27, 0.31, 0.35, 0.26, 0.29, 0.3, 0.35, 0.27, 0.29, 0.27, 0.28, 0.23],
#     [0.3, 0.39, 0.35, 0.34, 0.31, 0.26, 0.31, 0.34, 0.32, 0.32, 0.3, 0.35, 0.33, 0.29, 0.44, 0.26, 0.32, 0.35, 0.3, 0.25, 0.31, 0.37, 0.36, 0.25, 0.31, 0.33, 0.27, 0.37, 0.33, 0.35, 0.32, 0.38],
#     [0.27, 0.23, 0.43, 0.22, 0.33, 0.25, 0.43, 0.3, 0.29, 0.27, 0.32, 0.3, 0.35, 0.27, 0.51, 0.34, 0.3, 0.3, 0.27, 0.37, 0.31, 0.27, 0.33, 0.35, 0.3, 0.34, 0.34, 0.33, 0.29, 0.29, 0.28, 0.34],
#     [0.3, 0.23, 0.43, 0.34, 0.37, 0.35, 0.49, 0.37, 0.32, 0.37, 0.35, 0.35, 0.31, 0.3, 0.59, 0.29, 0.29, 0.29, 0.35, 0.29, 0.29, 0.33, 0.34, 0.32, 0.37, 0.32, 0.32, 0.35, 0.29, 0.22, 0.29, 0.37],
#     [0.31, 0.33, 0.47, 0.39, 0.31, 0.35, 0.42, 0.37, 0.3, 0.3, 0.37, 0.38, 0.29, 0.24, 0.53, 0.36, 0.31, 0.3, 0.29, 0.28, 0.38, 0.31, 0.34, 0.3, 0.3, 0.32, 0.35, 0.32, 0.3, 0.3, 0.31, 0.25],
#     [0.29, 0.29, 0.41, 0.32, 0.36, 0.29, 0.35, 0.53, 0.33, 0.28, 0.52, 0.59, 0.32, 0.28, 0.38, 0.37, 0.24, 0.26, 0.34, 0.32, 0.41, 0.3, 0.35, 0.28, 0.32, 0.32, 0.27, 0.32, 0.37, 0.33, 0.33, 0.31],
#     [0.36, 0.35, 0.35, 0.28, 0.34, 0.29, 0.36, 0.55, 0.43, 0.32, 0.54, 0.73, 0.34, 0.31, 0.31, 0.36, 0.33, 0.38, 0.32, 0.37, 0.41, 0.33, 0.38, 0.32, 0.29, 0.3, 0.32, 0.32, 0.36, 0.33, 0.38, 0.26],
#     [0.32, 0.25, 0.36, 0.3, 0.37, 0.37, 0.31, 0.51, 0.27, 0.4, 0.48, 0.62, 0.33, 0.32, 0.29, 0.32, 0.33, 0.32, 0.41, 0.29, 0.4, 0.32, 0.34, 0.37, 0.28, 0.33, 0.35, 0.34, 0.33, 0.32, 0.38, 0.33],
#     [0.29, 0.35, 0.31, 0.3, 0.28, 0.31, 0.35, 0.46, 0.35, 0.42, 0.38, 0.49, 0.35, 0.38, 0.33, 0.3, 0.55, 0.31, 0.46, 0.29, 0.4, 0.48, 0.33, 0.59, 0.33, 0.33, 0.31, 0.33, 0.32, 0.28, 0.41, 0.4],
#     [0.36, 0.28, 0.37, 0.34, 0.33, 0.31, 0.32, 0.37, 0.39, 0.62, 0.35, 0.3, 0.34, 0.46, 0.32, 0.31, 0.34, 0.35, 0.28, 0.37, 0.32, 0.58, 0.28, 0.73, 0.32, 0.35, 0.37, 0.34, 0.28, 0.35, 0.41, 0.36],
#     [0.27, 0.3, 0.35, 0.35, 0.35, 0.28, 0.37, 0.32, 0.38, 0.64, 0.3, 0.35, 0.35, 0.45, 0.38, 0.33, 0.37, 0.35, 0.33, 0.35, 0.32, 0.65, 0.35, 0.78, 0.35, 0.34, 0.37, 0.34, 0.34, 0.32, 0.44, 0.37],
#     [0.4, 0.37, 0.38, 0.43, 0.33, 0.37, 0.31, 0.35, 0.44, 0.64, 0.31, 0.37, 0.32, 0.49, 0.38, 0.29, 0.3, 0.25, 0.34, 0.33, 0.3, 0.59, 0.36, 0.71, 0.33, 0.38, 0.32, 0.37, 0.46, 0.29, 0.44, 0.37],
#     [0.32, 0.33, 0.36, 0.51, 0.44, 0.39, 0.37, 0.38, 0.63, 0.49, 0.31, 0.36, 0.32, 0.42, 0.31, 0.42, 0.31, 0.37, 0.31, 0.3, 0.31, 0.47, 0.4, 0.55, 0.35, 0.4, 0.35, 0.4, 0.55, 0.33, 0.42, 0.36],
#     [0.32, 0.42, 0.47, 0.62, 0.39, 0.36, 0.3, 0.39, 0.71, 0.41, 0.29, 0.36, 0.38, 0.39, 0.33, 0.36, 0.33, 0.32, 0.39, 0.37, 0.4, 0.4, 0.35, 0.39, 0.33, 0.46, 0.34, 0.36, 0.64, 0.33, 0.35, 0.42],
#     [0.38, 0.3, 0.48, 0.5, 0.4, 0.41, 0.41, 0.29, 0.6, 0.31, 0.34, 0.31, 0.38, 0.41, 0.47, 0.63, 0.35, 0.32, 0.33, 0.35, 0.37, 0.37, 0.32, 0.39, 0.38, 0.39, 0.36, 0.37, 0.55, 0.28, 0.38, 0.48],
#     [0.34, 0.39, 0.74, 0.49, 0.31, 0.35, 0.47, 0.39, 0.49, 0.33, 0.36, 0.29, 0.34, 0.42, 0.59, 0.35, 0.3, 0.29, 0.33, 0.35, 0.39, 0.29, 0.32, 0.32, 0.34, 0.4, 0.37, 0.29, 0.45, 0.34, 0.37, 0.65],
#     [0.37, 0.39, 0.82, 0.41, 0.37, 0.38, 0.44, 0.37, 0.42, 0.39, 0.29, 0.43, 0.36, 0.51, 0.75, 0.29, 0.42, 0.31, 0.32, 0.3, 0.32, 0.34, 0.39, 0.34, 0.33, 0.35, 0.37, 0.34, 0.32, 0.38, 0.35, 0.67],
#     [0.35, 0.55, 0.76, 0.39, 0.34, 0.49, 0.5, 0.32, 0.35, 0.43, 0.37, 0.35, 0.47, 0.51, 0.62, 0.34, 0.34, 0.33, 0.39, 0.35, 0.31, 0.37, 0.44, 0.42, 0.37, 0.35, 0.35, 0.35, 0.37, 0.35, 0.37, 0.63],
#     [0.31, 0.73, 0.57, 0.3, 0.34, 0.56, 0.4, 0.38, 0.39, 0.39, 0.39, 0.32, 0.63, 0.41, 0.5, 0.39, 0.37, 0.35, 0.3, 0.43, 0.32, 0.33, 0.41, 0.39, 0.3, 0.32, 0.44, 0.32, 0.35, 0.31, 0.39, 0.51],
#     [0.37, 0.75, 0.35, 0.35, 0.42, 0.7, 0.37, 0.35, 0.41, 0.41, 0.37, 0.35, 0.64, 0.37, 0.35, 0.4, 0.37, 0.32, 0.37, 0.49, 0.33, 0.5, 0.47, 0.45, 0.38, 0.42, 0.35, 0.38, 0.3, 0.4, 0.46, 0.35],
#     [0.33, 0.77, 0.44, 0.38, 0.34, 0.62, 0.35, 0.32, 0.37, 0.4, 0.38, 0.32, 0.63, 0.37, 0.38, 0.31, 0.41, 0.3, 0.34, 0.47, 0.37, 0.73, 0.4, 0.59, 0.42, 0.31, 0.63, 0.4, 0.31, 0.35, 0.58, 0.44],
#     [0.37, 0.59, 0.35, 0.41, 0.52, 0.47, 0.37, 0.39, 0.35, 0.47, 0.42, 0.32, 0.44, 0.4, 0.31, 0.34, 0.35, 0.35, 0.4, 0.39, 0.52, 0.85, 0.33, 0.7, 0.42, 0.35, 0.36, 0.38, 0.32, 0.34, 0.66, 0.38],
#     [0.41, 0.4, 0.37, 0.4, 0.58, 0.41, 0.34, 0.37, 0.38, 0.4, 0.5, 0.44, 0.34, 0.36, 0.28, 0.38, 0.3, 0.45, 0.36, 0.37, 0.71, 0.72, 0.39, 0.55, 0.43, 0.37, 0.37, 0.4, 0.35, 0.37, 0.55, 0.34],
#     [0.29, 0.41, 0.37, 0.46, 0.68, 0.29, 0.38, 0.35, 0.37, 0.38, 0.66, 0.49, 0.33, 0.37, 0.39, 0.35, 0.31, 0.32, 0.35, 0.35, 0.82, 0.44, 0.3, 0.47, 0.37, 0.31, 0.39, 0.31, 0.34, 0.38, 0.46, 0.4],
#     [0.39, 0.34, 0.32, 0.37, 0.54, 0.37, 0.36, 0.4, 0.37, 0.36, 0.59, 0.47, 0.4, 0.35, 0.35, 0.34, 0.35, 0.32, 0.39, 0.35, 0.56, 0.41, 0.56, 0.4, 0.34, 0.35, 0.33, 0.39, 0.33, 0.35, 0.37, 0.37],
# ])


def validate_hca(values: np.ndarray) -> np.ndarray:
    atlas = np.asarray(values, dtype=np.float64)
    if atlas.shape != SHAPE or not np.isfinite(atlas).all():
        raise ValueError(f"Expected a finite 32x32 {MODEL_NAME} HCA matrix")
    if np.any((atlas < 0) | (atlas > 1)):
        raise ValueError("HCA values must lie in [0, 1]; values are not rescaled")
    return atlas


def load_hca(path: Path) -> np.ndarray:
    """Load HCA without accepting pickle data or changing measured values."""
    model_ids = []
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        atlas = loaded
    else:
        with loaded as archive:
            if "hca" not in archive:
                raise ValueError("NPZ input must contain the matrix under key 'hca'")
            atlas = archive["hca"]
            if "model_id" in archive:
                model_ids.append(str(archive["model_id"].item()))
    metadata = path.with_name("metrics.json")
    if metadata.is_file():
        model_id = json.loads(metadata.read_text(encoding="utf-8")).get("model_id")
        if model_id is not None:
            model_ids.append(model_id)
    for model_id in model_ids:
        if model_id not in MODEL_IDS:
            raise ValueError(f"Expected {MODEL_ID} measurements, got {model_id!r}")
    return validate_hca(atlas)


def clustered_head_order(atlas: np.ndarray) -> np.ndarray:
    """One global permutation; Ward clusters use layers 16-32 only.

    Darker (lower-HCA) clusters are placed centrally, as in the original
    renderer. This arrangement does not imply intrinsic head symmetry.
    """
    from scipy.cluster.hierarchy import cut_tree, linkage

    profiles = atlas[15:32]
    tree = linkage(profiles.T, method="ward", metric="euclidean", optimal_ordering=True)
    labels = cut_tree(tree, n_clusters=[5]).ravel()
    means = profiles.mean(axis=0)
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    groups.sort(key=lambda heads: (float(means[heads].mean()), int(heads.min())))
    ranked = [head for group in groups for head in sorted(group, key=lambda h: (means[h], h))]
    positions = sorted(range(32), key=lambda i: (abs(i - 15.5), i))
    order = np.empty(32, dtype=int)
    order[positions] = ranked
    return order


def plot_atlas(values: np.ndarray, output: Path, head_order: str = "clustered-symmetric") -> np.ndarray:
    """Export PNG/PDF/SVG and a CSV mapping display columns to original heads."""
    atlas = validate_hca(values)
    if head_order not in {"original", "clustered-symmetric"}:
        raise ValueError("head_order must be original or clustered-symmetric")
    order = clustered_head_order(atlas) if head_order == "clustered-symmetric" else np.arange(32)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    style = {
        "font.family": "DejaVu Sans", "font.size": 8.0,
        "axes.labelsize": 8.2, "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(3.70, 2.85))
        try:
            # Keep the full theoretical HCA range; never rescale input values.
            image = ax.imshow(atlas[:, order], cmap="magma", origin="upper",
                              interpolation="nearest", aspect="equal", vmin=0, vmax=1)
            ticks = [0, 7, 15, 23, 31]
            ax.set_xticks(ticks, [str(order[i] + 1) for i in ticks])
            ax.set_yticks(ticks, ["1", "8", "16", "24", "32"])
            ax.set_xlabel("Attention head", labelpad=3.5)
            ax.set_ylabel("Decoder layer", labelpad=4.0)
            ax.tick_params(length=2.3, width=0.7, color="#555555")
            for spine in ax.spines.values():
                spine.set_color("#555555")
                spine.set_linewidth(0.75)
            for boundary in (10.5, 21.5):
                ax.axhline(boundary, color="white", linewidth=0.65,
                           alpha=0.65, linestyle=(0, (3, 2)))
            colorbar = fig.colorbar(image, ax=ax, fraction=0.047, pad=0.035)
            colorbar.set_label("Head-causal alignment", rotation=270, labelpad=11)
            colorbar.set_ticks([0, 0.5, 1])
            colorbar.ax.tick_params(length=2.0, labelsize=6.5)
            colorbar.outline.set_linewidth(0.65)
            for extension in ("png", "pdf", "svg"):
                fig.savefig(f"{output}.{extension}", dpi=360, bbox_inches="tight", pad_inches=0.035)
        finally:
            plt.close(fig)
    np.savetxt(f"{output}_head_order.csv", np.column_stack((np.arange(1, 33), order + 1)),
               fmt="%d", delimiter=",", header="display_column,original_head", comments="")
    return order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help=f"{MODEL_NAME} HCA .npy or .npz (key: hca)")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("llava_interleave_qwen7b_attention_atlas"),
                        help="Output path stem, without extension")
    parser.add_argument("--head-order", choices=("clustered-symmetric", "original"), default="clustered-symmetric")
    args = parser.parse_args()
    try:
        plot_atlas(load_hca(args.input), args.output, args.head_order)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Saved {args.output}.{{png,pdf,svg}} and {args.output}_head_order.csv")


if __name__ == "__main__":
    main()
