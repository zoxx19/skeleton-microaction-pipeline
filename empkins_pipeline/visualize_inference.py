"""
visualize_inference.py
======================
Visualize inferred micro-action segments with their predicted labels.

Produces three types of visualizations:
  1. Timeline — segments colored by label across the full recording
  2. Grid     — one skeleton frame per segment with label overlay
  3. GIFs     — animated skeleton for top N segments per condition

Run:
    python empkins_processing/visualize_inference.py

Change SUBJECT / CONDITION / PHASE at the top to visualize different data.
"""

import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa


# ============================================================
# CONFIG
# ============================================================
SUBJECT    = "VP_04"
CONDITIONS = ["tsst", "ftsst"]   # conditions to visualize
PHASES     = ["talk", "math"]    # phases to visualize (math skipped for ftsst)

NORMALIZED_BASE = Path("empkins_processing/normalized")
INFERENCE_BASE  = Path("empkins_processing/inference")
OUTPUT_DIR      = Path("empkins_processing/visualizations") / SUBJECT

MAX_GIFS_PER_PHASE = 5   # top N segments to animate per phase
GIF_STEP           = 2   # animate every Nth frame


# ============================================================
# REFINED 4-CLASS SCHEMA
# ============================================================
REMAP = {
    "tilting head":           "head movement",
    "turning head":           "head movement",
    "nodding":                "head movement",
    "shaking head":           "head movement",
    "head up":                "head movement",
    "bowing head":            "head movement",
    "illustrative gestures":  "arm gesturing",
    "spreading hands":        "arm gesturing",
    "retracting arms":        "arm gesturing",
    "waving":                 "arm gesturing",
    "stretching arms":        "arm gesturing",
    "playing objects":        "hand/object interaction",
    "other finger movements": "hand/object interaction",
    "hands touching fingers": "hand/object interaction",
    "rubbing hands":          "hand/object interaction",
    "scratching arms":        "hand/object interaction",
    "putting hands together": "hand/object interaction",
    "shaking body":           "body posture change",
    "sitting straightly":     "body posture change",
    "crossing arms":          "body posture change",
    "arms akimbo":            "body posture change",
    "shrugging":              "body posture change",
}

CLASS_COLORS = {
    "head movement":          "#E74C3C",
    "arm gesturing":          "#3498DB",
    "hand/object interaction":"#2ECC71",
    "body posture change":    "#F39C12",
    "other":                  "#95A5A6",
}

# Original 40-class colors (for original label mode)
ORIG_COLORS = {
    "tilting head":           "#E74C3C",
    "turning head":           "#C0392B",
    "nodding":                "#E91E63",
    "shaking head":           "#FF5722",
    "illustrative gestures":  "#3498DB",
    "spreading hands":        "#2980B9",
    "retracting arms":        "#1A5276",
    "waving":                 "#5DADE2",
    "shaking body":           "#F39C12",
    "sitting straightly":     "#D68910",
    "crossing arms":          "#E67E22",
    "arms akimbo":            "#CA6F1E",
    "playing objects":        "#2ECC71",
    "other finger movements": "#27AE60",
    "hands touching fingers": "#1E8449",
}


# ============================================================
# BlazePose edges for skeleton drawing
# ============================================================
EDGES = [
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
    (0, 7), (0, 8),
]


# ============================================================
# HELPERS
# ============================================================
def get_refined_label(orig_label):
    return REMAP.get(orig_label, "other")


def get_color(orig_label, use_refined=True):
    if use_refined:
        refined = get_refined_label(orig_label)
        return CLASS_COLORS.get(refined, "#95A5A6")
    return ORIG_COLORS.get(orig_label, "#95A5A6")


def load_results_for_subject(subject):
    pkl_path = INFERENCE_BASE / subject / "inference_results.pkl"
    if not pkl_path.exists():
        print(f"  No inference results found for {subject}")
        return []
    with open(str(pkl_path), "rb") as f:
        return pickle.load(f)


def load_segment_npy(subject, condition, phase, filename):
    path = NORMALIZED_BASE / subject / condition / phase / filename
    if not path.exists():
        return None
    return np.load(str(path))


# ============================================================
# VISUALIZATION 1: TIMELINE
# ============================================================
def plot_timeline(results, condition, phase, out_path):
    """
    Timeline showing each segment as a colored block.
    X axis = time (start frame), color = refined label.
    """
    cond_results = [r for r in results
                    if r["condition"] == condition and r["phase"] == phase]
    if not cond_results:
        return

    # Sort by start frame (extract from filename)
    def get_start(r):
        parts = r["file"].split("_")
        try:
            idx = parts.index("frames")
            return int(parts[idx + 1])
        except (ValueError, IndexError):
            return 0

    cond_results.sort(key=get_start)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6), gridspec_kw={"height_ratios": [3, 1]})

    # ── Top panel: colored timeline blocks ──
    ax1.set_xlim(0, len(cond_results))
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.set_xlabel("Segment index (sorted by time)", fontsize=11)
    ax1.set_title(f"{SUBJECT} — {condition}/{phase} — Micro-action timeline ({len(cond_results)} segments)",
                  fontsize=13, fontweight="bold")

    for i, r in enumerate(cond_results):
        color     = get_color(r["top1_name"], use_refined=True)
        refined   = get_refined_label(r["top1_name"])
        conf      = r["top1_prob"]
        alpha     = 0.4 + 0.6 * conf   # more opaque = more confident

        rect = mpatches.FancyBboxPatch(
            (i + 0.05, 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.02",
            facecolor=color, alpha=alpha, edgecolor="white", linewidth=0.5
        )
        ax1.add_patch(rect)

        # Confidence bar inside block
        ax1.add_patch(mpatches.Rectangle(
            (i + 0.05, 0.05), 0.9, conf * 0.9,
            facecolor=color, alpha=0.3, linewidth=0
        ))

    # ── Legend ──
    legend_patches = [
        mpatches.Patch(color=c, label=l)
        for l, c in CLASS_COLORS.items()
    ]
    ax1.legend(handles=legend_patches, loc="upper right",
               fontsize=9, framealpha=0.9, ncol=2)
    ax1.text(0.01, 0.92, "Block height = confidence", transform=ax1.transAxes,
             fontsize=8, color="gray", va="top")

    # ── Bottom panel: label distribution bar ──
    dist = Counter(get_refined_label(r["top1_name"]) for r in cond_results)
    labels = list(CLASS_COLORS.keys())
    counts = [dist.get(l, 0) for l in labels]
    colors = [CLASS_COLORS[l] for l in labels]
    pcts   = [100 * c / len(cond_results) for c in counts]

    bars = ax2.bar(range(len(labels)), pcts, color=colors, alpha=0.85, edgecolor="white")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("% of segments")
    ax2.set_ylim(0, 100)
    for bar, pct in zip(bars, pcts):
        if pct > 1:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{pct:.0f}%", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


# ============================================================
# VISUALIZATION 2: SKELETON GRID
# ============================================================
def draw_skeleton_on_ax(ax, frame, color, alpha=0.9):
    """Draw one skeleton frame on a 3D axis (MA-52 space, Y down -> flip for display)."""
    ax.scatter(frame[:, 0], frame[:, 2], -frame[:, 1],
               s=18, c=[color], alpha=alpha, depthshade=False)
    for i, j in EDGES:
        ax.plot([frame[i, 0], frame[j, 0]],
                [frame[i, 2], frame[j, 2]],
                [-frame[i, 1], -frame[j, 1]],
                lw=1.8, color=color, alpha=alpha)


def set_skeleton_limits(ax, frames_all):
    """Set equal axis limits for a skeleton axes."""
    all_pts = frames_all.reshape(-1, 3)
    for dim_idx, (get_vals, setter) in enumerate([
        (lambda p: p[:, 0], ax.set_xlim),
        (lambda p: p[:, 2], ax.set_ylim),
        (lambda p: -p[:, 1], ax.set_zlim),
    ]):
        vals = get_vals(all_pts)
        mid  = (vals.min() + vals.max()) / 2
        rng  = max(vals.max() - vals.min(), 0.2) * 0.65
        setter(mid - rng, mid + rng)


def plot_skeleton_grid(results, condition, phase, subject, out_path):
    """
    Grid of skeleton frames — one per segment, colored and labeled.
    Shows the middle frame of each segment.
    """
    cond_results = [r for r in results
                    if r["condition"] == condition and r["phase"] == phase]
    if not cond_results:
        return

    # Sort by confidence descending
    cond_results = sorted(cond_results, key=lambda r: -r["top1_prob"])

    n      = min(len(cond_results), 24)   # max 24 in grid
    ncols  = 6
    nrows  = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(ncols * 2.5, nrows * 2.8))
    fig.suptitle(f"{subject} — {condition}/{phase} — Segment grid (top {n} by confidence)",
                 fontsize=12, fontweight="bold", y=1.01)

    for idx in range(n):
        r = cond_results[idx]

        # Load segment
        poses = load_segment_npy(subject, condition, phase, r["file"])
        if poses is None:
            continue

        mid_frame = poses[len(poses) // 2]
        color     = get_color(r["top1_name"], use_refined=True)
        refined   = get_refined_label(r["top1_name"])
        conf      = r["top1_prob"]

        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        ax.view_init(elev=10, azim=-90)
        draw_skeleton_on_ax(ax, mid_frame, color)
        set_skeleton_limits(ax, poses)

        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")

        # Title: label + confidence
        short_label = refined.replace("hand/object interaction", "hand/obj")
        ax.set_title(f"{short_label}\n{r['top1_name'][:18]}\n({conf*100:.0f}%)",
                     fontsize=7, pad=2)

        # Colored border
        for spine in ax.spines.values():
            spine.set_edgecolor(color)

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=l) for l, c in CLASS_COLORS.items()]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=5, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


# ============================================================
# VISUALIZATION 3: ANIMATED GIFS
# ============================================================
def animate_segment_with_label(poses, label, refined, conf, fps, out_path, step=2):
    """Animate a segment as a GIF with label overlay."""
    frames = poses[::step]
    T      = len(frames)
    color  = CLASS_COLORS.get(refined, "#95A5A6")

    fig = plt.figure(figsize=(6, 5.5))
    fig.patch.set_facecolor("#1a1a2e")

    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#1a1a2e")
    ax.view_init(elev=12, azim=-90)

    set_skeleton_limits(ax, poses)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333355")
    ax.yaxis.pane.set_edgecolor("#333355")
    ax.zaxis.pane.set_edgecolor("#333355")
    ax.grid(True, alpha=0.15, color="#333355")

    scat,      = ax.plot([], [], [], "o", ms=5, color=color, alpha=0.9)
    bone_lines  = [ax.plot([], [], [], "-", lw=2, color=color, alpha=0.8)[0]
                   for _ in EDGES]

    # Label text
    label_txt = ax.text2D(0.5, 0.97, "", transform=ax.transAxes,
                          ha="center", va="top", fontsize=11,
                          color="white", fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor=color, alpha=0.8))
    conf_txt  = ax.text2D(0.5, 0.90, "", transform=ax.transAxes,
                          ha="center", va="top", fontsize=9, color="#CCCCCC")
    frame_txt = ax.text2D(0.02, 0.02, "", transform=ax.transAxes,
                          ha="left", va="bottom", fontsize=8, color="#888888")

    def init():
        scat.set_data([], [])
        scat.set_3d_properties([])
        for b in bone_lines:
            b.set_data([], [])
            b.set_3d_properties([])
        return [scat] + bone_lines

    def update(f):
        pts = frames[f]
        scat.set_data(pts[:, 0], pts[:, 2])
        scat.set_3d_properties(-pts[:, 1])
        for b, (i, j) in zip(bone_lines, EDGES):
            b.set_data([pts[i, 0], pts[j, 0]],
                       [pts[i, 2], pts[j, 2]])
            b.set_3d_properties([-pts[i, 1], -pts[j, 1]])
        label_txt.set_text(f"  {refined.upper()}  ")
        conf_txt.set_text(f"{label} — {conf*100:.1f}% confidence")
        frame_txt.set_text(f"frame {f * step + 1}/{len(poses)}")
        return [scat] + bone_lines + [label_txt, conf_txt, frame_txt]

    ani = animation.FuncAnimation(
        fig, update, frames=T, init_func=init,
        interval=max(20, int(1000 / (fps / step))), blit=False
    )
    ani.save(str(out_path), writer="pillow",
             fps=max(1, int(fps // step)),
             savefig_kwargs={"facecolor": "#1a1a2e"})
    plt.close()
    print(f"    Animated: {out_path.name}")


# ============================================================
# MAIN
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gif_dir = OUTPUT_DIR / "gifs"
    gif_dir.mkdir(exist_ok=True)

    print(f"{'='*55}")
    print(f"Visualizing inference results for {SUBJECT}")
    print(f"{'='*55}")
    print()

    # Load inference results
    results = load_results_for_subject(SUBJECT)
    if not results:
        print("No results found. Run batch_inference.py first.")
        return

    print(f"Loaded {len(results)} inference results")
    print()

    # Print summary
    print("Label summary (refined 4-class):")
    dist = Counter(get_refined_label(r["top1_name"]) for r in results)
    for label, count in dist.most_common():
        pct = 100 * count / len(results)
        bar = "█" * int(pct / 2)
        print(f"  {label:25s}: {count:3d} ({pct:5.1f}%) {bar}")
    print()

    # ── Process each condition/phase ──
    for condition in CONDITIONS:
        for phase in PHASES:
            cond_results = [r for r in results
                            if r["condition"] == condition and r["phase"] == phase]
            if not cond_results:
                continue

            print(f"--- {condition}/{phase} ({len(cond_results)} segments) ---")
            phase_dir = OUTPUT_DIR / condition / phase
            phase_dir.mkdir(parents=True, exist_ok=True)

            # 1. Timeline
            plot_timeline(
                results, condition, phase,
                out_path=phase_dir / f"timeline_{condition}_{phase}.png"
            )

            # 2. Skeleton grid
            plot_skeleton_grid(
                results, condition, phase, SUBJECT,
                out_path=phase_dir / f"skeleton_grid_{condition}_{phase}.png"
            )

            # 3. Animated GIFs — top N by confidence
            sorted_results = sorted(cond_results, key=lambda r: -r["top1_prob"])
            n_gifs = min(MAX_GIFS_PER_PHASE, len(sorted_results))
            print(f"  Animating top {n_gifs} segments...")

            for i, r in enumerate(sorted_results[:n_gifs]):
                poses = load_segment_npy(SUBJECT, condition, phase, r["file"])
                if poses is None:
                    print(f"    [SKIP] segment {i} — file not found")
                    continue

                refined  = get_refined_label(r["top1_name"])
                label    = r["top1_name"]
                conf     = r["top1_prob"]
                fps      = 59.0
                out_name = f"seg{i:02d}_{condition}_{phase}_{refined.replace('/', '_').replace(' ', '_')}.gif"

                animate_segment_with_label(
                    poses, label, refined, conf, fps,
                    out_path=gif_dir / out_name,
                    step=GIF_STEP
                )

            print()

    # ── Cross-condition comparison timeline ──
    print("Generating cross-condition comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (condition, phase) in zip(axes, [
        ("ftsst", "talk"), ("tsst", "talk"), ("tsst", "math")
    ]):
        cond_results = [r for r in results
                        if r["condition"] == condition and r["phase"] == phase]
        if not cond_results:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        # Sort by start frame
        def get_start(r):
            parts = r["file"].split("_")
            try:
                idx = parts.index("frames")
                return int(parts[idx + 1])
            except:
                return 0

        cond_results = sorted(cond_results, key=get_start)

        n = len(cond_results)
        ax.set_xlim(0, n)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title(f"{condition}/{phase}\n({n} segments)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Segment index (time →)", fontsize=9)

        for i, r in enumerate(cond_results):
            color   = get_color(r["top1_name"], use_refined=True)
            conf    = r["top1_prob"]
            alpha   = 0.3 + 0.7 * conf

            rect = mpatches.FancyBboxPatch(
                (i + 0.05, 0.1), 0.9, 0.8,
                boxstyle="square,pad=0",
                facecolor=color, alpha=alpha,
                edgecolor="none"
            )
            ax.add_patch(rect)

        # Mini distribution below
        dist = Counter(get_refined_label(r["top1_name"]) for r in cond_results)
        labels_list = list(CLASS_COLORS.keys())
        pcts = [100 * dist.get(l, 0) / n for l in labels_list]
        colors_list = [CLASS_COLORS[l] for l in labels_list]

        ax2 = ax.inset_axes([0, -0.35, 1, 0.25])
        ax2.bar(range(len(labels_list)), pcts, color=colors_list, alpha=0.85)
        ax2.set_xticks(range(len(labels_list)))
        ax2.set_xticklabels(
            [l.replace("hand/object interaction", "hand/obj").replace(" ", "\n")
             for l in labels_list],
            fontsize=6, rotation=0
        )
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("%", fontsize=7)
        for bar_i, pct in enumerate(pcts):
            if pct > 3:
                ax2.text(bar_i, pct + 1, f"{pct:.0f}", ha="center", fontsize=6)

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=l) for l, c in CLASS_COLORS.items()]
    fig.legend(handles=legend_patches, loc="upper center",
               ncol=5, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(f"{SUBJECT} — Cross-condition micro-action timeline",
                 fontsize=13, fontweight="bold", y=1.08)
    plt.tight_layout()
    comparison_path = OUTPUT_DIR / "cross_condition_comparison.png"
    fig.savefig(str(comparison_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {comparison_path.name}")

    print()
    print(f"{'='*55}")
    print(f"Done! All outputs in: {OUTPUT_DIR}")
    print(f"  cross_condition_comparison.png")
    print(f"  <condition>/<phase>/timeline_*.png")
    print(f"  <condition>/<phase>/skeleton_grid_*.png")
    print(f"  gifs/seg*.gif")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()