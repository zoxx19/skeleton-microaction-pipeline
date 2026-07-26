"""
verify_alignment.py
===================
Plot a normalized Empkins segment overlaid with a MA-52 sample
in the same 3D space to verify alignment quality.

Run:
    python empkins_processing/verify_alignment.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa


# ============================================================
# CONFIG
# ============================================================
SUBJECT      = "VP_04"
NORMALIZED_DIR = Path("empkins_processing/normalized_v2") / SUBJECT
MA52_DIR       = Path("npy_out_ma_neutralized/train")

ANIMATE_STEP   = 2
OUTPUT_DIR     = Path("empkins_processing/verification")


# ============================================================
# BlazePose edges
# ============================================================
EDGES = [
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
    (0, 7), (0, 8), (0, 9), (0, 10),
]


# ============================================================
# STATIC COMPARISON: 3 views side by side
# ============================================================
def plot_static_comparison(empkins_seq, ma52_seq, out_path):
    """
    Plot one frame from each skeleton from 3 angles,
    then overlay both in the same plot.
    """
    # Use middle frame
    emp_frame = empkins_seq[len(empkins_seq)//2]
    ma52_frame = ma52_seq[len(ma52_seq)//2, :, :3]  # drop 4th channel if present

    fig = plt.figure(figsize=(18, 5))

    views = [
        (10, -90, "Front view"),
        (10,   0, "Side view"),
        (60, -90, "Top view"),
    ]

    for col, (elev, azim, label) in enumerate(views):
        ax = fig.add_subplot(1, 3, col+1, projection="3d")
        ax.view_init(elev=elev, azim=azim)

        # Plot MA-52 (blue)
        _plot_skeleton_on_ax(ax, ma52_frame, color="steelblue",
                             label="MA-52", alpha=0.7)

        # Plot Empkins normalized (red)
        _plot_skeleton_on_ax(ax, emp_frame, color="tomato",
                             label="Empkins (normalized)", alpha=0.7)

        # Set equal limits from both
        all_pts = np.vstack([emp_frame, ma52_frame])
        _set_equal_limits(ax, all_pts)

        ax.set_title(label)
        ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("-Y(up)")
        if col == 0:
            ax.legend(loc="upper left", fontsize=8)

    plt.suptitle("Empkins (red) vs MA-52 (blue) — normalized to same space",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def _plot_skeleton_on_ax(ax, frame, color, label=None, alpha=1.0):
    """Plot one skeleton frame on a 3D axis. Y is down so we flip for display."""
    # display: X=X, Y=Z(depth), Z=-Y(height)
    ax.scatter(frame[:, 0], frame[:, 2], -frame[:, 1],
               s=20, c=[color], alpha=alpha,
               label=label)
    for i, j in EDGES:
        ax.plot([frame[i, 0], frame[j, 0]],
                [frame[i, 2], frame[j, 2]],
                [-frame[i, 1], -frame[j, 1]],
                lw=1.5, color=color, alpha=alpha)


def _set_equal_limits(ax, pts):
    xs = pts[:, 0]; ys = pts[:, 2]; zs = -pts[:, 1]
    def lim(v):
        mid = (v.min() + v.max()) / 2
        rng = max(v.max() - v.min(), 0.1) * 0.6
        return mid - rng, mid + rng
    ax.set_xlim(*lim(xs))
    ax.set_ylim(*lim(ys))
    ax.set_zlim(*lim(zs))


# ============================================================
# ANIMATED OVERLAY
# ============================================================
def animate_overlay(empkins_seq, ma52_seq, fps, title, step, out_path):
    """
    Animate both skeletons overlaid.
    Empkins = red, MA-52 = blue (static reference frame).
    """
    emp_frames  = empkins_seq[::step]
    T           = emp_frames.shape[0]
    ma52_frame  = ma52_seq[len(ma52_seq)//2, :, :3]  # static MA-52 reference

    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=10, azim=-90)

    # Compute limits from both
    all_pts = np.vstack([emp_frames.reshape(-1, 3), ma52_frame])
    def lim(v):
        mid = (v.min() + v.max()) / 2
        rng = max(v.max() - v.min(), 0.1) * 0.65
        return mid - rng, mid + rng

    ax.set_xlim(*lim(all_pts[:, 0]))
    ax.set_ylim(*lim(all_pts[:, 2]))
    ax.set_zlim(*lim(-all_pts[:, 1]))
    ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("-Y(up)")

    # Static MA-52 reference (blue, transparent)
    _plot_skeleton_on_ax(ax, ma52_frame, color="steelblue", alpha=0.3)

    # Animated Empkins (red)
    scat_e,    = ax.plot([], [], [], "o", ms=4, color="tomato", label="Empkins")
    bones_e     = [ax.plot([], [], [], "-", lw=1.5, color="tomato")[0]
                   for _ in EDGES]
    title_txt   = ax.set_title("")
    ax.legend(loc="upper left", fontsize=8)

    def init():
        scat_e.set_data([], [])
        scat_e.set_3d_properties([])
        for b in bones_e:
            b.set_data([], [])
            b.set_3d_properties([])
        return [scat_e] + bones_e

    def update(f):
        pts = emp_frames[f]
        scat_e.set_data(pts[:, 0], pts[:, 2])
        scat_e.set_3d_properties(-pts[:, 1])
        for b, (i, j) in zip(bones_e, EDGES):
            b.set_data([pts[i, 0], pts[j, 0]],
                       [pts[i, 2], pts[j, 2]])
            b.set_3d_properties([-pts[i, 1], -pts[j, 1]])
        title_txt.set_text(f"{title} | f{f*step}")
        return [scat_e] + bones_e + [title_txt]

    ani = animation.FuncAnimation(
        fig, update, frames=T, init_func=init,
        interval=max(20, int(1000/(fps/step))), blit=False
    )
    ani.save(str(out_path), writer="pillow",
             fps=max(1, int(fps//step)))
    plt.close()
    print(f"Saved: {out_path}")


# ============================================================
# PRINT STATS COMPARISON
# ============================================================
def print_stats_comparison(empkins_seq, ma52_seq):
    ma52_f = ma52_seq[:, :, :3]

    print("=== Coordinate space comparison ===")
    print(f"{'':20s}  {'Empkins':>20s}  {'MA-52':>20s}")
    print("-" * 65)

    for label, idx in [("l_hip (23)", 23), ("r_hip (24)", 24),
                        ("nose (0)", 0), ("l_shoulder (11)", 11),
                        ("l_knee (25)", 25), ("l_ankle (27)", 27)]:
        e = empkins_seq[len(empkins_seq)//2, idx]
        m = ma52_f[len(ma52_f)//2, idx]
        print(f"  {label:18s}  "
              f"[{e[0]:+.3f},{e[1]:+.3f},{e[2]:+.3f}]  "
              f"[{m[0]:+.3f},{m[1]:+.3f},{m[2]:+.3f}]")

    print()
    print(f"  {'X range':15s}  "
          f"[{empkins_seq[:,:,0].min():.3f}, {empkins_seq[:,:,0].max():.3f}]  "
          f"[{ma52_f[:,:,0].min():.3f}, {ma52_f[:,:,0].max():.3f}]")
    print(f"  {'Y range':15s}  "
          f"[{empkins_seq[:,:,1].min():.3f}, {empkins_seq[:,:,1].max():.3f}]  "
          f"[{ma52_f[:,:,1].min():.3f}, {ma52_f[:,:,1].max():.3f}]")
    print(f"  {'Z range':15s}  "
          f"[{empkins_seq[:,:,2].min():.3f}, {empkins_seq[:,:,2].max():.3f}]  "
          f"[{ma52_f[:,:,2].min():.3f}, {ma52_f[:,:,2].max():.3f}]")

    # Torso length
    def torso(seq):
        sho = 0.5 * (seq[0, 11] + seq[0, 12])
        hip = 0.5 * (seq[0, 23] + seq[0, 24])
        return np.linalg.norm(sho - hip)

    print(f"\n  Torso length: Empkins={torso(empkins_seq):.4f}  "
          f"MA-52={torso(ma52_f):.4f}")
    print()


# ============================================================
# MAIN
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load first normalized Empkins segment
    emp_files = sorted(NORMALIZED_DIR.rglob("*.npy"))
    if not emp_files:
        print("No normalized segments found! Run step3_normalize.py first.")
        return

    # Load MA-52 sample
    ma52_files = sorted(MA52_DIR.glob("*.npy"))
    if not ma52_files:
        print(f"No MA-52 files found in {MA52_DIR}")
        return

    # Pick highest-scoring Empkins segment (first = highest score)
    emp_path  = emp_files[0]
    ma52_path = ma52_files[0]

    emp_seq  = np.load(str(emp_path))
    ma52_seq = np.load(str(ma52_path))

    print(f"Empkins segment : {emp_path.name}  shape={emp_seq.shape}")
    print(f"MA-52 sample    : {ma52_path.name}  shape={ma52_seq.shape}")
    print()

    # Print stats comparison
    print_stats_comparison(emp_seq, ma52_seq)

    # Static 3-view comparison
    plot_static_comparison(
        emp_seq, ma52_seq,
        out_path=OUTPUT_DIR / "static_comparison.png"
    )

    # Animated overlay
    animate_overlay(
        emp_seq, ma52_seq,
        fps=59.0,
        title=f"Empkins(red) vs MA-52(blue)",
        step=ANIMATE_STEP,
        out_path=OUTPUT_DIR / "animated_overlay.gif"
    )

    print(f"\nVerification files saved to: {OUTPUT_DIR}")
    print("  static_comparison.png  — 3 views side by side")
    print("  animated_overlay.gif   — Empkins animating over static MA-52")


if __name__ == "__main__":
    main()