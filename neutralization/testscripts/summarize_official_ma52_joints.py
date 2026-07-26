#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize joint layout for one sample from official MA52 packed PKL.

Outputs:
- <frame_dir>_mean_pose.png         : average visible joint position with indices
- <frame_dir>_joint_stats.csv       : per-joint visibility + mean position
- <frame_dir>_joint_stats.txt       : joints sorted by vertical position

Use this to identify which joint indices belong to the lower body
before writing a new normalization script for the 44-joint dataset.
"""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_sample(pkl_path: Path, sample_index: int):
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    ann = data["annotations"][sample_index]
    xy = np.asarray(ann["keypoint"])[0]          # (T, V, 2)
    score = np.asarray(ann["keypoint_score"])[0] # (T, V)
    return ann, xy, score


def compute_stats(xy: np.ndarray, score: np.ndarray, thr: float):
    T, V, _ = xy.shape
    rows = []

    for j in range(V):
        keep = score[:, j] >= thr
        visible = int(np.sum(keep))
        ratio = float(visible / T)

        if visible > 0:
            mean_x = float(np.mean(xy[keep, j, 0]))
            mean_y = float(np.mean(xy[keep, j, 1]))
            min_x = float(np.min(xy[keep, j, 0]))
            max_x = float(np.max(xy[keep, j, 0]))
            min_y = float(np.min(xy[keep, j, 1]))
            max_y = float(np.max(xy[keep, j, 1]))
            mean_score = float(np.mean(score[keep, j]))
        else:
            mean_x = mean_y = min_x = max_x = min_y = max_y = mean_score = float("nan")

        rows.append({
            "joint_index": j,
            "visible_frames": visible,
            "visible_ratio": ratio,
            "mean_score": mean_score,
            "mean_x": mean_x,
            "mean_y": mean_y,
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
        })

    return rows


def save_mean_pose_png(rows, out_png: Path, title: str):
    visible_rows = [r for r in rows if r["visible_frames"] > 0]
    if not visible_rows:
        raise ValueError("No visible joints for this threshold")

    xs = np.array([r["mean_x"] for r in visible_rows], dtype=float)
    ys = np.array([r["mean_y"] for r in visible_rows], dtype=float)
    sizes = np.array([20 + 80 * r["visible_ratio"] for r in visible_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.scatter(xs, ys, s=sizes)

    for r in visible_rows:
        ax.text(r["mean_x"] + 2, r["mean_y"] + 2, str(r["joint_index"]), fontsize=9)

    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    xpad = max(10.0, 0.05 * (xmax - xmin + 1e-9))
    ypad = max(10.0, 0.05 * (ymax - ymin + 1e-9))

    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymax + ypad, ymin - ypad)  # invert y
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("mean x")
    ax.set_ylabel("mean y")
    ax.grid(True, alpha=0.2)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_csv(rows, out_csv: Path):
    fieldnames = [
        "joint_index", "visible_frames", "visible_ratio", "mean_score",
        "mean_x", "mean_y", "min_x", "max_x", "min_y", "max_y"
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def save_sorted_txt(rows, out_txt: Path):
    visible_rows = [r for r in rows if r["visible_frames"] > 0]
    visible_rows.sort(key=lambda r: r["mean_y"])

    with out_txt.open("w", encoding="utf-8") as f:
        f.write("Joints sorted by mean_y (top to bottom on image)\n")
        f.write("joint_index | visible_ratio | mean_score | mean_x | mean_y\n")
        for r in visible_rows:
            f.write(
                f"{r['joint_index']:>2} | "
                f"{r['visible_ratio']:.3f} | "
                f"{r['mean_score']:.3f} | "
                f"{r['mean_x']:.1f} | "
                f"{r['mean_y']:.1f}\n"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--score-thr", type=float, default=0.2)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ann, xy, score = load_sample(Path(args.pkl).expanduser().resolve(), args.sample_index)
    rows = compute_stats(xy, score, args.score_thr)

    stem = ann["frame_dir"]
    title = f"{stem} | label={ann['label']} | mean pose | score >= {args.score_thr:.2f}"

    out_png = out_dir / f"{stem}_mean_pose.png"
    out_csv = out_dir / f"{stem}_joint_stats.csv"
    out_txt = out_dir / f"{stem}_joint_stats.txt"

    save_mean_pose_png(rows, out_png, title)
    save_csv(rows, out_csv)
    save_sorted_txt(rows, out_txt)

    print("[OK] Saved mean pose PNG :", out_png)
    print("[OK] Saved joint stats CSV:", out_csv)
    print("[OK] Saved sorted TXT    :", out_txt)
    print("[INFO] frame_dir:", stem)
    print("[INFO] joints:", xy.shape[1])
    print("[INFO] frames:", xy.shape[0])
    print("[INFO] score threshold:", args.score_thr)


if __name__ == "__main__":
    main()
