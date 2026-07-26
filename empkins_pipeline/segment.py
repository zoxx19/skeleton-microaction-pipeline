"""
step2_segment.py
================
Segment movement clips from all phases of one Empkins subject.

Global per-subject threshold:
  1. Load ALL available phases (tsst/talk, tsst/math, ftsst/talk)
  2. Compute activity signal for ALL combined
  3. Compute ONE global p85 threshold from the combined signal
  4. Apply that SAME threshold to each phase individually
  5. Segment and save organized by condition/phase

This makes the threshold comparable across conditions — if tsst/math
is genuinely calmer it gets fewer segments, not artificially the same %.

Run:
    python empkins_processing/step2_segment.py
"""

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa


# ============================================================
# CONFIG
# ============================================================
DATA_ROOT  = Path("/home/hpc/iwso/REDACTED_ACCOUNT/repos/Micro-Action/data/empkins_pilot/data_per_subject")
SUBJECT    = "VP_04"

# Global threshold percentile — computed across ALL phases of this subject
ACTIVITY_PERCENTILE = 85

# Segmentation parameters
MIN_VEL_FRAMES  = 5    # min consecutive active frames
PAD_FRAMES      = 25   # frames to pad around each burst
MIN_SEG_LEN     = 60   # min segment length (~1 sec at 59fps)
MAX_SEG_LEN     = 300  # max segment length (~5 sec at 59fps)
MIN_TOUCH_CHAIN = 2    # min touching spans to merge

# Upper body joints for activity detection
UPPER_JOINTS = [
    "Head", "Neck",
    "RightShoulder", "LeftShoulder",
    "RightArm",      "LeftArm",
    "RightForeArm",  "LeftForeArm",
    "RightHand",     "LeftHand",
]

# Animation
ANIMATE_STEP  = 2
MAX_ANIM_SEGS = 5

OUTPUT_BASE = Path("empkins_processing/segments") / SUBJECT


# ============================================================
# LOADING
# ============================================================
def parse_bvh_edges(bvh_gz_path):
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


def parse_bvh_fps(bvh_gz_path):
    with gzip.open(bvh_gz_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip().startswith("Frame Time:"):
                ft = float(line.split(":")[1].strip())
                return round(1.0 / ft) if ft > 0 else 58.8
    return 58.8


def load_global_pose(csv_gz_path, joint_names):
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
            continue
        P[:, vi, :] = df[cols].to_numpy()
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


def load_phase_frames(times_json_path, fps):
    with open(times_json_path) as f:
        times = json.load(f)
    clap_frame = times["mocap"]["clap_frame"]
    clap_sec   = times["video"]["clap_sec"]

    def sec_to_frame(sec):
        return clap_frame + int((sec - clap_sec) * fps)

    phases = {}
    if times["video"].get("total") and times["video"]["total"]:
        phases["total"] = (
            sec_to_frame(times["video"]["total"]["begin_sec"]),
            sec_to_frame(times["video"]["total"]["end_sec"]),
        )
    if times["video"].get("talk") and times["video"]["talk"]:
        phases["talk"] = (
            sec_to_frame(times["video"]["talk"]["begin_sec"]),
            sec_to_frame(times["video"]["talk"]["end_sec"]),
        )
    if times["video"].get("math") and times["video"]["math"]:
        phases["math"] = (
            sec_to_frame(times["video"]["math"]["begin_sec"]),
            sec_to_frame(times["video"]["math"]["end_sec"]),
        )
    return phases


def get_body_joints_only(joint_names, edges):
    body_joints = [
        "Hips",
        "RightUpLeg", "RightLeg", "RightFoot",
        "LeftUpLeg",  "LeftLeg",  "LeftFoot",
        "Spine", "Spine1", "Spine2", "Spine3",
        "Neck", "Head",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        "LeftShoulder",  "LeftArm",  "LeftForeArm",  "LeftHand",
    ]
    old_to_new  = {}
    new_names   = []
    for old_idx, name in enumerate(joint_names):
        if name in body_joints:
            old_to_new[old_idx] = len(new_names)
            new_names.append(name)
    new_edges   = [(old_to_new[i], old_to_new[j])
                   for i, j in edges
                   if i in old_to_new and j in old_to_new]
    old_indices = [old_idx for old_idx, name in enumerate(joint_names)
                   if name in body_joints]
    return new_names, new_edges, old_indices


def center_by_hip(P, hip_idx=0):
    return P - P[:, hip_idx:hip_idx+1, :]


# ============================================================
# ACTIVITY SIGNAL
# ============================================================
def compute_activity_signal(P, joint_names):
    """Mean upper-body velocity per frame. P: (T,V,3) -> (T-1,)"""
    upper_idx = [joint_names.index(j)
                 for j in UPPER_JOINTS if j in joint_names]
    vel       = np.linalg.norm(np.diff(P, axis=0), axis=2)  # (T-1, V)
    return vel[:, upper_idx].mean(axis=1)                    # (T-1,)


# ============================================================
# MOVEMENT DETECTION
# ============================================================
def _mark_runs_at_least_k(bools, k):
    if k <= 1:
        return bools.copy()
    out = np.zeros_like(bools, dtype=bool)
    n   = len(bools)
    i   = 0
    while i < n:
        if not bools[i]:
            i += 1
            continue
        j = i + 1
        while j < n and bools[j]:
            j += 1
        if (j - i) >= k:
            out[i:j] = True
        i = j
    return out


def detect_activity_frames(activity, threshold, min_vel_frames):
    """Flag frames above threshold. Returns mask (T,)"""
    high  = activity > threshold
    high  = _mark_runs_at_least_k(high, min_vel_frames)
    T     = len(activity) + 1
    mask  = np.zeros(T, dtype=bool)
    mask[1:] = high
    return mask


# ============================================================
# SEGMENTATION
# ============================================================
def segment_by_movement(P, movement, pad, min_seg, max_seg, min_touch_chain):
    n = P.shape[0]
    if not movement.any():
        return [], []

    mm     = movement.astype(int)
    starts = np.flatnonzero(np.diff(np.r_[0, mm]) == 1)
    ends   = np.flatnonzero(np.diff(np.r_[mm, 0]) == -1)

    spans = [(max(0, s - pad), min(n - 1, e + pad))
             for s, e in zip(starts, ends)]
    spans.sort(key=lambda x: x[0])

    merged = []
    cs, ce = spans[0]
    for s, e in spans[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append([cs, ce])
            cs, ce = s, e
    merged.append([cs, ce])

    final = []
    chain = [merged[0]]

    def flush(ch):
        if len(ch) >= min_touch_chain:
            final.append((ch[0][0], ch[-1][1]))
        else:
            final.extend((x[0], x[1]) for x in ch)

    for span in merged[1:]:
        prev = chain[-1]
        if span[0] == prev[1] + 1:
            chain.append(span)
        else:
            flush(chain)
            chain = [span]
    flush(chain)

    filtered = [(s, e) for s, e in final
                if min_seg <= (e - s + 1) <= max_seg]
    segments = [P[s:e+1] for s, e in filtered]
    return segments, filtered


def score_segment(seg, joint_names):
    if seg.shape[0] < 2:
        return 0.0
    upper_idx = [joint_names.index(j)
                 for j in UPPER_JOINTS if j in joint_names]
    vel = np.linalg.norm(np.diff(seg, axis=0), axis=2)
    return float(np.mean(vel[:, upper_idx]))


# ============================================================
# VISUALIZATION
# ============================================================
def plot_activity_and_segments(activity, movement, spans, threshold,
                                fps, title, out_path):
    T  = len(activity)
    t  = np.arange(T) / fps

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)

    ax1.plot(t, activity, lw=0.8, color="steelblue", alpha=0.8)
    ax1.axhline(threshold, color="red", lw=1.5, linestyle="--",
                label=f"Global p{ACTIVITY_PERCENTILE} = {threshold:.3f}")
    ax1.set_ylabel("Activity")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title(title)

    t2 = np.arange(len(movement)) / fps
    ax2.fill_between(t2, movement.astype(int), alpha=0.3,
                     color="steelblue", label="active frames")
    for i, (s, e) in enumerate(spans):
        ax2.axvspan(s/fps, e/fps, alpha=0.4, color="red")
        ax2.text((s+e)/2/fps, 1.1, str(i), ha="center",
                 fontsize=7, color="darkred")
    ax2.set_ylabel("Movement")
    ax2.set_xlabel("Time (sec)")
    ax2.set_ylim(-0.1, 1.4)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def animate_segment(seg, edges, joint_names, fps, title, step, out_path):
    frames = seg[::step]
    T      = frames.shape[0]

    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=10, azim=90)

    def axis_lim(vals):
        lo, hi = vals.min(), vals.max()
        mid = (lo + hi) / 2
        rng = max(hi - lo, 10.0) * 1.2 / 2
        return mid - rng, mid + rng

    ax.set_xlim(*axis_lim(frames[:, :, 0].flatten()))
    ax.set_ylim(*axis_lim(frames[:, :, 2].flatten()))
    ax.set_zlim(*axis_lim(frames[:, :, 1].flatten()))
    ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("Y(up)")

    scat,      = ax.plot([], [], [], "o", ms=5, color="steelblue")
    bone_lines  = [ax.plot([], [], [], "-", lw=2, color="steelblue")[0]
                   for _ in edges]
    title_txt   = ax.set_title("")

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
        scat.set_3d_properties(pts[:, 1])
        for b, (i, j) in zip(bone_lines, edges):
            b.set_data([pts[i, 0], pts[j, 0]],
                       [pts[i, 2], pts[j, 2]])
            b.set_3d_properties([pts[i, 1], pts[j, 1]])
        title_txt.set_text(f"{title} | f{f*step}")
        return [scat] + bone_lines + [title_txt]

    ani = animation.FuncAnimation(
        fig, update, frames=T, init_func=init,
        interval=max(20, int(1000/(fps/step))), blit=False
    )
    ani.save(str(out_path), writer="pillow", fps=max(1, int(fps//step)))
    plt.close()


# ============================================================
# PROCESS ONE PHASE
# ============================================================
def process_phase(P_body, body_names, body_edges, fps,
                  global_threshold, phase_name, condition,
                  out_dir):
    """Segment one phase using the global threshold."""
    out_dir.mkdir(parents=True, exist_ok=True)

    hip_idx    = body_names.index("Hips")
    P_centered = center_by_hip(P_body, hip_idx)

    activity = compute_activity_signal(P_centered, body_names)
    movement = detect_activity_frames(
        activity, global_threshold, MIN_VEL_FRAMES
    )

    n_active = movement.sum()
    pct      = 100 * n_active / len(movement)
    print(f"  Active frames: {n_active}/{len(movement)} ({pct:.1f}%)")

    segments, spans = segment_by_movement(
        P_centered, movement,
        pad=PAD_FRAMES,
        min_seg=MIN_SEG_LEN,
        max_seg=MAX_SEG_LEN,
        min_touch_chain=MIN_TOUCH_CHAIN,
    )

    print(f"  Segments found: {len(segments)}")

    # Score + sort
    scores   = [score_segment(s, body_names) for s in segments]
    order    = np.argsort(-np.array(scores))
    segments = [segments[i] for i in order]
    spans    = [spans[i]    for i in order]
    scores   = [scores[i]   for i in order]

    # Plot
    plot_activity_and_segments(
        activity, movement, spans, global_threshold,
        fps=fps,
        title=f"{SUBJECT} {condition} {phase_name} — global p{ACTIVITY_PERCENTILE} = {global_threshold:.3f}",
        out_path=out_dir / "activity_and_segments.png",
    )

    # Log
    log_lines = []
    log_lines.append(f"Subject: {SUBJECT}  Condition: {condition}  Phase: {phase_name}")
    log_lines.append(f"FPS: {fps:.1f}  |  Phase frames: {len(movement)}")
    log_lines.append(f"Global threshold (p{ACTIVITY_PERCENTILE}): {global_threshold:.4f}")
    log_lines.append(f"Active frames: {pct:.1f}%")
    log_lines.append(f"Segments: {len(segments)}")
    log_lines.append("")
    log_lines.append(f"{'#':>3}  {'start':>6}  {'end':>6}  {'len':>5}  {'sec':>5}  {'score':>8}")
    log_lines.append("-" * 45)

    for i, (seg, (s, e), sc) in enumerate(zip(segments, spans, scores)):
        L    = e - s + 1
        line = f"{i:>3}  {s:>6}  {e:>6}  {L:>5}  {L/fps:>5.1f}  {sc:>8.4f}"
        log_lines.append(line)
        print(f"  {line}")

    with open(out_dir / "segmentation_log.txt", "w") as f:
        for line in log_lines:
            f.write(line + "\n")

    # Save .npz
    for i, (seg, (s, e), sc) in enumerate(zip(segments, spans, scores)):
        np.savez_compressed(
            str(out_dir / f"seg_{i:03d}_frames_{s}_{e}.npz"),
            poses=seg,
            start_frame=s,
            end_frame=e,
            score=sc,
            fps=fps,
            joint_names=np.array(body_names, dtype=object),
            condition=condition,
            phase=phase_name,
            subject=SUBJECT,
        )

    # Animate top N
    n_anim = min(MAX_ANIM_SEGS, len(segments))
    for i in range(n_anim):
        seg  = segments[i]
        s, e = spans[i]
        sc   = scores[i]
        L    = e - s + 1
        title    = f"seg{i:03d}_f{s}-{e}_len{L}_sc{sc:.3f}"
        out_path = out_dir / f"{title}.gif"
        animate_segment(seg, body_edges, body_names,
                        fps, title, ANIMATE_STEP, out_path)

    return len(segments)


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"{'='*60}")
    print(f"Subject: {SUBJECT}")
    print(f"Global threshold: p{ACTIVITY_PERCENTILE} across all phases")
    print(f"{'='*60}")
    print()

    subject_dir = DATA_ROOT / SUBJECT

    # Discover all available condition/phase combinations
    all_phases = []
    for condition in ["tsst", "ftsst"]:
        filtered_dir = subject_dir / condition / "mocap" / "filtered"
        times_json   = subject_dir / condition / f"{SUBJECT}_times_{condition}.json"

        if not filtered_dir.exists() or not times_json.exists():
            print(f"[SKIP] {condition} — missing data")
            continue

        bvh_files = sorted(filtered_dir.glob("*.bvh.gz"))
        gp_files  = sorted(filtered_dir.glob("*_global_pose.csv.gz"))
        if not bvh_files or not gp_files:
            continue

        bvh_path = str(bvh_files[0])
        gp_path  = str(gp_files[0])

        joint_names, edges = parse_bvh_edges(bvh_path)
        fps = parse_bvh_fps(bvh_path)
        body_names, body_edges, body_indices = get_body_joints_only(
            joint_names, edges)

        P_full = load_global_pose(gp_path, joint_names)
        P_body = P_full[:, body_indices, :]

        phases = load_phase_frames(str(times_json), fps)

        for phase_name, (start, end) in phases.items():
            if phase_name == "total":
                continue  # skip total, use talk+math separately
            start = max(0, start)
            end   = min(P_body.shape[0] - 1, end)
            P_phase = P_body[start:end]

            hip_idx    = body_names.index("Hips")
            P_centered = center_by_hip(P_phase, hip_idx)
            activity   = compute_activity_signal(P_centered, body_names)

            all_phases.append({
                "condition":   condition,
                "phase":       phase_name,
                "P_centered":  P_centered,
                "activity":    activity,
                "body_names":  body_names,
                "body_edges":  body_edges,
                "fps":         fps,
                "start":       start,
                "end":         end,
            })

            dur = P_phase.shape[0] / fps
            print(f"  Loaded {condition}/{phase_name}: "
                  f"{P_phase.shape[0]} frames ({dur:.0f} sec)")

    if not all_phases:
        print("No phases found!")
        return

    # Compute GLOBAL threshold from ALL phases combined
    all_activity = np.concatenate([p["activity"] for p in all_phases])
    global_threshold = float(np.percentile(all_activity, ACTIVITY_PERCENTILE))

    print()
    print(f"{'='*60}")
    print(f"Global activity stats (all phases combined):")
    print(f"  Total frames : {len(all_activity)}")
    print(f"  Mean         : {all_activity.mean():.4f}")
    print(f"  Std          : {all_activity.std():.4f}")
    print(f"  p50          : {np.percentile(all_activity, 50):.4f}")
    print(f"  p85          : {np.percentile(all_activity, 85):.4f}")
    print(f"  p90          : {np.percentile(all_activity, 90):.4f}")
    print(f"  p95          : {np.percentile(all_activity, 95):.4f}")
    print(f"  Global threshold (p{ACTIVITY_PERCENTILE}): {global_threshold:.4f}")
    print(f"{'='*60}")
    print()

    # Process each phase with the global threshold
    summary = []
    for p in all_phases:
        condition  = p["condition"]
        phase_name = p["phase"]
        out_dir    = OUTPUT_BASE / condition / phase_name

        print(f"\n--- {condition} / {phase_name} ---")

        n_segs = process_phase(
            P_body       = p["P_centered"],
            body_names   = p["body_names"],
            body_edges   = p["body_edges"],
            fps          = p["fps"],
            global_threshold = global_threshold,
            phase_name   = phase_name,
            condition    = condition,
            out_dir      = out_dir,
        )

        dur = (p["end"] - p["start"]) / p["fps"]
        summary.append((condition, phase_name, n_segs, dur))

    # Final summary
    print()
    print(f"{'='*60}")
    print(f"SUMMARY — {SUBJECT} (global threshold = {global_threshold:.4f})")
    print(f"{'='*60}")
    print(f"{'Condition':>8}  {'Phase':>6}  {'Segs':>5}  {'Duration':>10}  {'Segs/min':>9}")
    print("-" * 50)
    total_segs = 0
    for condition, phase, n_segs, dur in summary:
        segs_per_min = n_segs / (dur / 60)
        print(f"{condition:>8}  {phase:>6}  {n_segs:>5}  "
              f"{dur:>8.0f}s  {segs_per_min:>9.1f}")
        total_segs += n_segs
    print("-" * 50)
    print(f"{'TOTAL':>8}  {'':>6}  {total_segs:>5}")
    print()
    print(f"Output: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()