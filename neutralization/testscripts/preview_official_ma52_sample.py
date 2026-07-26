#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preview one sample from the official MA52 packed PKL dataset and save:
- original_frame_<idx>.png
- original_sample.gif

Works with train_data.pkl / val_data.pkl / test_data.pkl where each annotation has:
  frame_dir, keypoint, keypoint_score, label, total_frames, ...
Expected keypoint shape: (1, T, V, 2)
Expected keypoint_score shape: (1, T, V)
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def load_sample(pkl_path: Path, sample_index: int):
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    anns = data["annotations"]
    ann = anns[sample_index]

    keypoint = np.asarray(ann["keypoint"])         # (1, T, V, 2)
    keypoint_score = np.asarray(ann["keypoint_score"])  # (1, T, V)

    if keypoint.ndim != 4 or keypoint.shape[0] != 1 or keypoint.shape[-1] != 2:
        raise ValueError(f"Unexpected keypoint shape: {keypoint.shape}")
    if keypoint_score.ndim != 3 or keypoint_score.shape[0] != 1:
        raise ValueError(f"Unexpected keypoint_score shape: {keypoint_score.shape}")

    xy = keypoint[0]          # (T, V, 2)
    score = keypoint_score[0] # (T, V)

    return ann, xy, score


def save_png(xy: np.ndarray, score: np.ndarray, frame_idx: int, out_png: Path, title: str):
    frame_idx = max(0, min(frame_idx, len(xy) - 1))
    pts = xy[frame_idx]
    sc = score[frame_idx]

    fig, ax = plt.subplots(figsize=(7, 7))
    sizes = 15 + 35 * np.clip(sc, 0, 1)
    ax.scatter(pts[:, 0], pts[:, 1], s=sizes)
    ax.set_title(title + f" | frame {frame_idx}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_gif(xy: np.ndarray, score: np.ndarray, out_gif: Path, title: str, fps: int = 12, max_frames: int = None):
    if max_frames is not None:
        xy = xy[:max_frames]
        score = score[:max_frames]

    T = len(xy)
    if T == 0:
        raise ValueError("Empty sequence")

    x_all = xy[..., 0]
    y_all = xy[..., 1]
    xmin, xmax = float(np.min(x_all)), float(np.max(x_all))
    ymin, ymax = float(np.min(y_all)), float(np.max(y_all))
    xpad = max(10.0, 0.05 * (xmax - xmin + 1e-9))
    ypad = max(10.0, 0.05 * (ymax - ymin + 1e-9))

    fig, ax = plt.subplots(figsize=(7, 7))
    pts0 = xy[0]
    sc0 = score[0]
    sizes0 = 15 + 35 * np.clip(sc0, 0, 1)
    scat = ax.scatter(pts0[:, 0], pts0[:, 1], s=sizes0)

    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymax + ypad, ymin - ypad)  # inverted y-axis
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)
    title_text = ax.set_title(title + " | frame 0")

    def update(t):
        pts = xy[t]
        sc = score[t]
        sizes = 15 + 35 * np.clip(sc, 0, 1)
        scat.set_offsets(pts)
        scat.set_sizes(sizes)
        title_text.set_text(title + f" | frame {t}")
        return scat, title_text

    anim = FuncAnimation(fig, update, frames=T, interval=max(1, int(1000 / max(1, fps))), blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Preview one MA52 sample from packed PKL.")
    ap.add_argument("--pkl", required=True, help="Path to train_data.pkl / val_data.pkl / test_data.pkl")
    ap.add_argument("--sample-index", type=int, default=0, help="Annotation index to preview")
    ap.add_argument("--frame-idx", type=int, default=12, help="Frame index for PNG")
    ap.add_argument("--fps", type=int, default=12, help="GIF FPS")
    ap.add_argument("--max-frames", type=int, default=120, help="Cap frames in GIF for speed")
    ap.add_argument("--output-dir", required=True, help="Where to save the PNG/GIF")
    args = ap.parse_args()

    pkl_path = Path(args.pkl).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ann, xy, score = load_sample(pkl_path, args.sample_index)

    stem = ann["frame_dir"]
    label = ann["label"]
    total_frames = ann["total_frames"]
    title = f"{stem} | label={label} | total_frames={total_frames}"

    out_png = out_dir / f"{stem}_original_frame_{args.frame_idx}.png"
    out_gif = out_dir / f"{stem}_original.gif"

    save_png(xy, score, args.frame_idx, out_png, title)
    save_gif(xy, score, out_gif, title, fps=args.fps, max_frames=args.max_frames)

    print("[OK] Saved PNG:", out_png)
    print("[OK] Saved GIF:", out_gif)
    print("[INFO] frame_dir:", stem)
    print("[INFO] label:", label)
    print("[INFO] keypoint shape:", xy.shape)
    print("[INFO] score shape:", score.shape)


if __name__ == "__main__":
    main()
