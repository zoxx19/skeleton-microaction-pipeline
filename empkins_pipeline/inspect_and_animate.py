"""
inspect_and_animate.py
======================
Step 1: Load one Empkins subject, extract the relevant TSST phase
using the times JSON, center the skeleton by hip, and animate.

Run:
    python empkins_processing/inspect_and_animate.py
"""

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed on HPC
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa


# ============================================================
# CONFIG — change these to test different subjects/conditions
# ============================================================
# Set <PROJECT_ROOT> to your local checkout of the Micro-Action repository.
DATA_ROOT    = Path("<PROJECT_ROOT>/data/empkins_pilot/data_per_subject")
SUBJECT      = "VP_04"
CONDITION    = "tsst"       # "tsst" or "ftsst"
PHASE        = "talk"       # "talk", "math", or "total"

ANIMATE_STEP = 3            # show every Nth frame (speeds up gif)
MAX_FRAMES   = 600          # max frames to animate (None = all)
OUTPUT_DIR   = Path(".")    # where to save the gif


# ============================================================
# LOADING
# ============================================================
def parse_bvh_edges(bvh_gz_path: str):
    """Extract joint names and parent-child edges from BVH hierarchy."""
    with gzip.open(bvh_gz_path, "rt", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    lines    = text.split("MOTION")[0].splitlines()
    names    = []
    edges    = []
    stack    = []
    last_idx = None

    for line in lines:
        line = line.strip()
        m = re.match(r"^(ROOT|JOINT)\s+(.+)$", line)
        if m:
            names.append(m.group(2).strip())
            last_idx = len(names) - 1
            if stack:
                edges.append((stack[-1], last_idx))
            continue
        if line == "{":
            if last_idx is not None:
                stack.append(last_idx)
                last_idx = None
        elif line == "}":
            if stack:
                stack.pop()

    return names, edges


def parse_bvh_fps(bvh_gz_path: str) -> float:
    """Extract FPS from BVH motion header."""
    with gzip.open(bvh_gz_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip().startswith("Frame Time:"):
                ft = float(line.split(":")[1].strip())
                return round(1.0 / ft) if ft > 0 else 58.8
    return 58.8


def load_global_pose(csv_gz_path: str, joint_names: list) -> np.ndarray:
    """
    Load global pose CSV -> (T, V, 3) float32.
    Each joint has columns: joint=X, joint.1=Y, joint.2=Z
    NaNs are filled via interpolation.
    """
    df = pd.read_csv(csv_gz_path, compression="gzip", low_memory=False)
    for c in df.columns:
        if c != "body_part":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    T = len(df)
    V = len(joint_names)
    P = np.zeros((T, V, 3), dtype=np.float32)

    for vi, jname in enumerate(joint_names):
        cols = [jname, f"{jname}.1", f"{jname}.2"]
        if any(c not in df.columns for c in cols):
            print(f"  [WARN] Joint '{jname}' columns not found - zeroing")
            continue
        P[:, vi, :] = df[cols].to_numpy()

    # Fill NaNs via linear interpolation per joint per axis
    for j in range(V):
        for c in range(3):
            x = P[:, j, c]
            mask = np.isnan(x)
            if not mask.any():
                continue
            valid = np.where(~mask)[0]
            if len(valid) == 0:
                x[:] = 0.0
            else:
                x[:valid[0]]    = x[valid[0]]
                x[valid[-1]+1:] = x[valid[-1]]
                miss = np.where(np.isnan(x))[0]
                good = np.where(~np.isnan(x))[0]
                x[miss] = np.interp(miss, good, x[good])
            P[:, j, c] = x

    return P


def load_phase_frames(times_json_path: str, fps: float) -> dict:
    """
    Parse the times JSON and return frame indices for each phase.
    Uses the mocap clap_frame as the sync point between video time and mocap frames.
    """
    with open(times_json_path) as f:
        times = json.load(f)

    clap_frame = times["mocap"]["clap_frame"]
    clap_sec   = times["video"]["clap_sec"]

    def sec_to_frame(sec):
        return clap_frame + int((sec - clap_sec) * fps)

    phases = {
        "total": (
            sec_to_frame(times["video"]["total"]["begin_sec"]),
            sec_to_frame(times["video"]["total"]["end_sec"]),
        ),
        "talk": (
            sec_to_frame(times["video"]["talk"]["begin_sec"]),
            sec_to_frame(times["video"]["talk"]["end_sec"]),
        ),
        "math": (
            sec_to_frame(times["video"]["math"]["begin_sec"]),
            sec_to_frame(times["video"]["math"]["end_sec"]),
        ),
    }

    print("=== Phase frame ranges ===")
    for name, (s, e) in phases.items():
        duration = (e - s) / fps
        print(f"  {name:6s}: frames {s} -> {e}  ({duration:.1f} sec)")
    print()

    return phases


# ============================================================
# SKELETON PROCESSING
# ============================================================
def center_by_hip(P: np.ndarray, hip_idx: int = 0) -> np.ndarray:
    """
    Center skeleton by subtracting the hip position at each frame.
    P: (T, V, 3)
    """
    hip = P[:, hip_idx:hip_idx+1, :]   # (T, 1, 3)
    return P - hip


def get_body_joints_only(joint_names: list, edges: list):
    """
    Filter to only the main body joints — no fingers.
    Returns (new_names, new_edges, old_indices).
    """
    body_joints = [
        "Hips",
        "RightUpLeg", "RightLeg", "RightFoot",
        "LeftUpLeg",  "LeftLeg",  "LeftFoot",
        "Spine", "Spine1", "Spine2", "Spine3",
        "Neck", "Head",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        "LeftShoulder",  "LeftArm",  "LeftForeArm",  "LeftHand",
    ]

    old_to_new = {}
    new_names  = []
    for old_idx, name in enumerate(joint_names):
        if name in body_joints:
            old_to_new[old_idx] = len(new_names)
            new_names.append(name)

    new_edges = []
    for i, j in edges:
        if i in old_to_new and j in old_to_new:
            new_edges.append((old_to_new[i], old_to_new[j]))

    old_indices = [old_idx for old_idx, name in enumerate(joint_names)
                   if name in body_joints]

    return new_names, new_edges, old_indices


# ============================================================
# ANIMATION
# ============================================================
def animate_skeleton(P: np.ndarray,
                     edges: list,
                     joint_names: list,
                     fps: float,
                     title: str = "Skeleton",
                     step: int = 3,
                     max_frames: int = None,
                     out_dir: Path = Path(".")):
    """
    Animate 3D skeleton and save as gif.

    Empkins data: Y is up, X is left/right, Z is depth.
    Matplotlib 3D: Z is the vertical axis.
    We swap Y and Z for display so the person appears upright.

    P: (T, V, 3)  columns = [X, Y_up, Z_depth]
    """
    frames = P[::step]
    if max_frames is not None:
        frames = frames[:max_frames]

    T = frames.shape[0]
    print(f"Animating {T} frames (every {step} frames, fps={fps:.1f})")

    fig = plt.figure(figsize=(8, 7))
    ax  = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=10, azim=90)


    # Compute display coords: swap Y<->Z
    dx = frames[:, :, 0]          # X stays
    dy = frames[:, :, 2]          # Z_depth -> display Y
    dz = frames[:, :, 1]          # Y_up    -> display Z (vertical)

    def axis_limits(vals):
        lo, hi = vals.min(), vals.max()
        mid    = (lo + hi) / 2
        rng    = max(hi - lo, 10.0) * 1.15 / 2
        return mid - rng, mid + rng

    ax.set_xlim(*axis_limits(dx.flatten()))
    ax.set_ylim(*axis_limits(dy.flatten()))
    ax.set_zlim(*axis_limits(dz.flatten()))
    ax.set_xlabel("X (left/right)")
    ax.set_ylabel("Z (depth)")
    ax.set_zlabel("Y (up)")

    scat,     = ax.plot([], [], [], "o", ms=5, color="steelblue")
    bone_lines = [ax.plot([], [], [], "-", lw=2, color="steelblue")[0]
                  for _ in edges]
    title_txt  = ax.set_title("")

    def init():
        scat.set_data([], [])
        scat.set_3d_properties([])
        for b in bone_lines:
            b.set_data([], [])
            b.set_3d_properties([])
        return [scat] + bone_lines

    def update(f):
        pts = frames[f]                      # (V, 3)

        px = pts[:, 0]                       # X
        py = pts[:, 2]                       # Z -> depth
        pz = pts[:, 1]                       # Y -> height

        scat.set_data(px, py)
        scat.set_3d_properties(pz)

        for b, (i, j) in zip(bone_lines, edges):
            b.set_data([pts[i, 0], pts[j, 0]],
                       [pts[i, 2], pts[j, 2]])
            b.set_3d_properties([pts[i, 1], pts[j, 1]])

        title_txt.set_text(f"{title} | frame {f * step}")
        return [scat] + bone_lines + [title_txt]

    interval_ms = max(20, int(1000 / (fps / step)))
    ani = animation.FuncAnimation(
        fig, update, frames=T, init_func=init,
        interval=interval_ms, blit=False
    )

    plt.tight_layout()
    safe_title = title.replace(" ", "_").replace("|", "").replace("/", "_")
    out_path   = out_dir / f"{safe_title}.gif"
    print(f"Saving: {out_path}")
    ani.save(str(out_path), writer="pillow", fps=max(1, int(fps // step)))
    plt.close()
    print("Done.")


# ============================================================
# MAIN
# ============================================================
def main():
    subject_dir  = DATA_ROOT / SUBJECT
    filtered_dir = subject_dir / CONDITION / "mocap" / "filtered"
    times_json   = subject_dir / CONDITION / f"{SUBJECT}_times_tsst.json"

    bvh_files = sorted(filtered_dir.glob("*.bvh.gz"))
    gp_files  = sorted(filtered_dir.glob("*_global_pose.csv.gz"))

    if not bvh_files:
        raise FileNotFoundError(f"No .bvh.gz in {filtered_dir}")
    if not gp_files:
        raise FileNotFoundError(f"No global_pose.csv.gz in {filtered_dir}")
    if not times_json.exists():
        raise FileNotFoundError(f"No times JSON at {times_json}")

    bvh_path = str(bvh_files[0])
    gp_path  = str(gp_files[0])

    print(f"Subject   : {SUBJECT}")
    print(f"Condition : {CONDITION}")
    print(f"Phase     : {PHASE}")
    print()

    # Parse skeleton
    joint_names, edges = parse_bvh_edges(bvh_path)
    fps = parse_bvh_fps(bvh_path)
    print(f"Total joints : {len(joint_names)}  |  FPS : {fps:.1f}")
    print()

    # Keep only body joints (drop fingers)
    body_names, body_edges, body_indices = get_body_joints_only(joint_names, edges)
    print(f"Body joints  : {len(body_names)}")
    print(f"Names        : {body_names}")
    print()

    # Load full pose
    P_full = load_global_pose(gp_path, joint_names)
    print(f"Full pose  : {P_full.shape}  ({P_full.shape[0]/fps:.1f} sec)")
    print()

    # Keep body joints only
    P_body = P_full[:, body_indices, :]

    # Extract phase using times JSON
    phases    = load_phase_frames(str(times_json), fps)
    start, end = phases[PHASE]
    start      = max(0, start)
    end        = min(P_body.shape[0] - 1, end)
    P_phase    = P_body[start:end]
    print(f"Phase '{PHASE}': frames {start}-{end}  ({P_phase.shape[0]/fps:.1f} sec)")
    print()

    # Center by hip
    hip_idx    = body_names.index("Hips")
    P_centered = center_by_hip(P_phase, hip_idx=hip_idx)
    print(f"Centered by '{body_names[hip_idx]}' (idx={hip_idx})")
    print(f"Y range after centering: [{P_centered[:,:,1].min():.1f}, {P_centered[:,:,1].max():.1f}] cm")
    print()


     # Center by hip
    hip_idx    = body_names.index("Hips")
    P_centered = center_by_hip(P_phase, hip_idx=hip_idx)
    print(f"Centered by '{body_names[hip_idx]}' (idx={hip_idx})")
    print(f"Y range after centering: [{P_centered[:,:,1].min():.1f}, {P_centered[:,:,1].max():.1f}] cm")
    print()

    # ---- ADD THIS BLOCK HERE ----
    fig2, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})
    frame_to_show = P_centered[len(P_centered)//2]  # middle frame

    for ax2, (elev, azim, label) in zip(axes, [
        (10,  90,  "Front view"),
        (10,   0,  "Side view"),
        (60, -90,  "Top view"),
    ]):
        ax2.view_init(elev=elev, azim=azim)
        pts = frame_to_show
        ax2.scatter(pts[:,0], pts[:,2], pts[:,1], s=40, color="steelblue")
        for i, j in body_edges:
            ax2.plot([pts[i,0], pts[j,0]], [pts[i,2], pts[j,2]], [pts[i,1], pts[j,1]], lw=2, color="steelblue")
        ax2.set_xlabel("X"); ax2.set_ylabel("Z"); ax2.set_zlabel("Y(up)")
        ax2.set_title(label)

    plt.suptitle(f"{SUBJECT} {CONDITION} - Static pose check", fontsize=13)
    plt.tight_layout()
    fig2.savefig(f"{SUBJECT}_{CONDITION}_pose_check.png", dpi=150)
    plt.close(fig2)
    print("Saved pose_check.png")
    # ---- END OF BLOCK ----


    # Animate
    animate_skeleton(
        P=P_centered,
        edges=body_edges,
        joint_names=body_names,
        fps=fps,
        title=f"{SUBJECT}_{CONDITION}_{PHASE}_centered",
        step=ANIMATE_STEP,
        max_frames=MAX_FRAMES,
        out_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()