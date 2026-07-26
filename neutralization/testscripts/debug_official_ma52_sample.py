#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Debug-view one sample from official MA52 packed PKL with score filtering and joint indices.

Saves:
- <frame_dir>_frame_<idx>_raw.png
- <frame_dir>_frame_<idx>_filtered.png
- <frame_dir>_frame_<idx>_indexed.png
- <frame_dir>_filtered.gif
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

    ann = data["annotations"][sample_index]
    xy = np.asarray(ann["keypoint"])[0]          # (T, V, 2)
    score = np.asarray(ann["keypoint_score"])[0] # (T, V)
    return ann, xy, score


def _axis_limits(xy):
    x_all = xy[..., 0]
    y_all = xy[..., 1]
    xmin, xmax = float(np.min(x_all)), float(np.max(x_all))
    ymin, ymax = float(np.min(y_all)), float(np.max(y_all))
    xpad = max(10.0, 0.05 * (xmax - xmin + 1e-9))
    ypad = max(10.0, 0.05 * (ymax - ymin + 1e-9))
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


def save_raw_png(xy, score, frame_idx, out_png, title):
    pts = xy[frame_idx]
    sc = score[frame_idx]

    fig, ax = plt.subplots(figsize=(7, 9))
    sizes = 15 + 45 * np.clip(sc, 0, 1)
    ax.scatter(pts[:, 0], pts[:, 1], s=sizes)
    ax.set_title(title + f" | raw | frame {frame_idx}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    xmin, xmax, ymin, ymax = _axis_limits(xy)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)  # invert y
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_filtered_png(xy, score, frame_idx, out_png, title, thr):
    pts = xy[frame_idx]
    sc = score[frame_idx]
    keep = sc >= thr

    fig, ax = plt.subplots(figsize=(7, 9))
    sizes = 20 + 55 * np.clip(sc[keep], 0, 1)
    ax.scatter(pts[keep, 0], pts[keep, 1], s=sizes)
    ax.set_title(title + f" | score >= {thr:.2f} | frame {frame_idx}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    xmin, xmax, ymin, ymax = _axis_limits(xy)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_indexed_png(xy, score, frame_idx, out_png, title, thr):
    pts = xy[frame_idx]
    sc = score[frame_idx]
    keep = sc >= thr

    fig, ax = plt.subplots(figsize=(7, 9))
    sizes = 20 + 55 * np.clip(sc[keep], 0, 1)
    ax.scatter(pts[keep, 0], pts[keep, 1], s=sizes)

    for idx in np.where(keep)[0]:
        ax.text(pts[idx, 0] + 2, pts[idx, 1] + 2, str(idx), fontsize=8)

    ax.set_title(title + f" | indexed, score >= {thr:.2f} | frame {frame_idx}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    xmin, xmax, ymin, ymax = _axis_limits(xy)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_filtered_gif(xy, score, out_gif, title, thr, fps=12, max_frames=120):
    xy = xy[:max_frames]
    score = score[:max_frames]
    T = len(xy)

    fig, ax = plt.subplots(figsize=(7, 9))
    keep0 = score[0] >= thr
    scat = ax.scatter(xy[0, keep0, 0], xy[0, keep0, 1], s=20 + 55 * np.clip(score[0, keep0], 0, 1))
    xmin, xmax, ymin, ymax = _axis_limits(xy)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)
    title_text = ax.set_title(title + f" | score >= {thr:.2f} | frame 0")

    def update(t):
        keep = score[t] >= thr
        pts = xy[t, keep]
        scat.set_offsets(pts if len(pts) else np.empty((0, 2)))
        scat.set_sizes(20 + 55 * np.clip(score[t, keep], 0, 1))
        title_text.set_text(title + f" | score >= {thr:.2f} | frame {t}")
        return scat, title_text

    anim = FuncAnimation(fig, update, frames=T, interval=max(1, int(1000 / max(1, fps))), blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--frame-idx", type=int, default=12)
    ap.add_argument("--score-thr", type=float, default=0.2)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ann, xy, score = load_sample(Path(args.pkl).expanduser().resolve(), args.sample_index)
    stem = ann["frame_dir"]
    title = f"{stem} | label={ann['label']} | total_frames={ann['total_frames']}"

    raw_png = out_dir / f"{stem}_frame_{args.frame_idx}_raw.png"
    filtered_png = out_dir / f"{stem}_frame_{args.frame_idx}_filtered.png"
    indexed_png = out_dir / f"{stem}_frame_{args.frame_idx}_indexed.png"
    gif_path = out_dir / f"{stem}_filtered.gif"

    save_raw_png(xy, score, args.frame_idx, raw_png, title)
    save_filtered_png(xy, score, args.frame_idx, filtered_png, title, args.score_thr)
    save_indexed_png(xy, score, args.frame_idx, indexed_png, title, args.score_thr)
    save_filtered_gif(xy, score, gif_path, title, args.score_thr, fps=args.fps, max_frames=args.max_frames)

    print("[OK] Saved raw PNG     :", raw_png)
    print("[OK] Saved filtered PNG:", filtered_png)
    print("[OK] Saved indexed PNG :", indexed_png)
    print("[OK] Saved filtered GIF:", gif_path)
    print("[INFO] frame_dir:", stem)
    print("[INFO] keypoint shape:", xy.shape)
    print("[INFO] score shape:", score.shape)
    print("[INFO] score threshold:", args.score_thr)


if __name__ == "__main__":
    main()
