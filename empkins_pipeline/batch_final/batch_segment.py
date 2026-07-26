"""
batch_segment.py
================
Run segmentation on ALL subjects, ALL conditions, ALL phases.
Uses global per-subject threshold (p85 across all phases of each subject).

Run:
    python empkins_processing/batch_segment.py
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

# ============================================================
# CONFIG
# ============================================================
# Set <PROJECT_ROOT> to your local checkout of the Micro-Action repository.
DATA_ROOT  = Path("<PROJECT_ROOT>/data/empkins_pilot/data_per_subject")
OUTPUT_BASE = Path("empkins_processing/segments")

SUBJECTS = [
    "VP_01","VP_02","VP_03","VP_04","VP_05","VP_06","VP_07",
    "VP_08","VP_09","VP_10","VP_11","VP_12","VP_13","VP_14",
    "VP_15","VP_16","VP_17","VP_18","VP_19","VP_20","VP_21",
]

ACTIVITY_PERCENTILE = 85
MIN_VEL_FRAMES  = 5
PAD_FRAMES      = 25
MIN_SEG_LEN     = 60
MAX_SEG_LEN     = 300
MIN_TOUCH_CHAIN = 2

UPPER_JOINTS = [
    "Head", "Neck",
    "RightShoulder", "LeftShoulder",
    "RightArm",      "LeftArm",
    "RightForeArm",  "LeftForeArm",
    "RightHand",     "LeftHand",
]

# No animations in batch mode — saves time
ANIMATE = False


# ============================================================
# LOADING
# ============================================================
def parse_bvh_edges(bvh_gz_path):
    with gzip.open(bvh_gz_path, "rt", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    lines    = text.split("MOTION")[0].splitlines()
    names, edges, stack, last_idx = [], [], [], None
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
    T, V = len(df), len(joint_names)
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
                x[:valid[0]] = x[valid[0]]
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
    def s2f(sec):
        return clap_frame + int((sec - clap_sec) * fps)
    phases = {}
    if times["video"].get("total") and times["video"]["total"]:
        phases["total"] = (s2f(times["video"]["total"]["begin_sec"]),
                           s2f(times["video"]["total"]["end_sec"]))
    if times["video"].get("talk") and times["video"]["talk"]:
        phases["talk"] = (s2f(times["video"]["talk"]["begin_sec"]),
                          s2f(times["video"]["talk"]["end_sec"]))
    if times["video"].get("math") and times["video"]["math"]:
        phases["math"] = (s2f(times["video"]["math"]["begin_sec"]),
                          s2f(times["video"]["math"]["end_sec"]))
    return phases


def get_body_joints_only(joint_names, edges):
    body_joints = [
        "Hips","RightUpLeg","RightLeg","RightFoot",
        "LeftUpLeg","LeftLeg","LeftFoot",
        "Spine","Spine1","Spine2","Spine3","Neck","Head",
        "RightShoulder","RightArm","RightForeArm","RightHand",
        "LeftShoulder","LeftArm","LeftForeArm","LeftHand",
    ]
    old_to_new, new_names = {}, []
    for old_idx, name in enumerate(joint_names):
        if name in body_joints:
            old_to_new[old_idx] = len(new_names)
            new_names.append(name)
    new_edges   = [(old_to_new[i], old_to_new[j])
                   for i, j in edges if i in old_to_new and j in old_to_new]
    old_indices = [old_idx for old_idx, name in enumerate(joint_names)
                   if name in body_joints]
    return new_names, new_edges, old_indices


def center_by_hip(P, hip_idx=0):
    return P - P[:, hip_idx:hip_idx+1, :]


# ============================================================
# ACTIVITY + DETECTION
# ============================================================
def compute_activity_signal(P, joint_names):
    upper_idx = [joint_names.index(j) for j in UPPER_JOINTS if j in joint_names]
    vel = np.linalg.norm(np.diff(P, axis=0), axis=2)
    return vel[:, upper_idx].mean(axis=1)


def _mark_runs_at_least_k(bools, k):
    if k <= 1:
        return bools.copy()
    out = np.zeros_like(bools, dtype=bool)
    n, i = len(bools), 0
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
    high = _mark_runs_at_least_k(activity > threshold, min_vel_frames)
    mask = np.zeros(len(activity) + 1, dtype=bool)
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
    spans  = [(max(0, s-pad), min(n-1, e+pad)) for s, e in zip(starts, ends)]
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

    final, chain = [], [merged[0]]
    def flush(ch):
        if len(ch) >= min_touch_chain:
            final.append((ch[0][0], ch[-1][1]))
        else:
            final.extend((x[0], x[1]) for x in ch)
    for span in merged[1:]:
        if span[0] == chain[-1][1] + 1:
            chain.append(span)
        else:
            flush(chain)
            chain = [span]
    flush(chain)

    filtered = [(s, e) for s, e in final if min_seg <= (e-s+1) <= max_seg]
    return [P[s:e+1] for s, e in filtered], filtered


def score_segment(seg, joint_names):
    if seg.shape[0] < 2:
        return 0.0
    upper_idx = [joint_names.index(j) for j in UPPER_JOINTS if j in joint_names]
    vel = np.linalg.norm(np.diff(seg, axis=0), axis=2)
    return float(np.mean(vel[:, upper_idx]))


# ============================================================
# PLOT
# ============================================================
def plot_activity(activity, movement, spans, threshold, fps, title, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    t = np.arange(len(activity)) / fps
    ax1.plot(t, activity, lw=0.8, color="steelblue", alpha=0.8)
    ax1.axhline(threshold, color="red", lw=1.5, ls="--",
                label=f"Global p{ACTIVITY_PERCENTILE}={threshold:.3f}")
    ax1.set_ylabel("Activity"); ax1.legend(fontsize=8); ax1.set_title(title)
    t2 = np.arange(len(movement)) / fps
    ax2.fill_between(t2, movement.astype(int), alpha=0.3, color="steelblue")
    for i, (s, e) in enumerate(spans):
        ax2.axvspan(s/fps, e/fps, alpha=0.4, color="red")
        ax2.text((s+e)/2/fps, 1.1, str(i), ha="center", fontsize=6, color="darkred")
    ax2.set_xlabel("Time (sec)"); ax2.set_ylabel("Movement"); ax2.set_ylim(-0.1, 1.4)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=100)
    plt.close()


# ============================================================
# PROCESS ONE SUBJECT
# ============================================================
def process_subject(subject):
    subject_dir = DATA_ROOT / subject
    print(f"\n{'='*55}")
    print(f"  Subject: {subject}")
    print(f"{'='*55}")

    # Discover all condition/phase data
    all_phases = []
    for condition in ["tsst", "ftsst"]:
        filtered_dir = subject_dir / condition / "mocap" / "filtered"
        times_json   = subject_dir / condition / f"{subject}_times_{condition}.json"
        if not filtered_dir.exists() or not times_json.exists():
            continue
        bvh_files = sorted(filtered_dir.glob("*.bvh.gz"))
        gp_files  = sorted(filtered_dir.glob("*_global_pose.csv.gz"))
        if not bvh_files or not gp_files:
            continue

        joint_names, edges = parse_bvh_edges(str(bvh_files[0]))
        fps = parse_bvh_fps(str(bvh_files[0]))
        body_names, body_edges, body_indices = get_body_joints_only(joint_names, edges)

        P_full = load_global_pose(str(gp_files[0]), joint_names)
        P_body = P_full[:, body_indices, :]

        try:
            phases = load_phase_frames(str(times_json), fps)
        except Exception as e:
            print(f"  [SKIP] {condition}: {e}")
            continue

        for phase_name, (start, end) in phases.items():
            if phase_name == "total":
                continue
            start = max(0, start)
            end   = min(P_body.shape[0] - 1, end)
            if end <= start:
                continue
            P_phase    = P_body[start:end]
            hip_idx    = body_names.index("Hips")
            P_centered = P_phase - P_phase[:, hip_idx:hip_idx+1, :]
            activity   = compute_activity_signal(P_centered, body_names)
            all_phases.append({
                "condition": condition, "phase": phase_name,
                "P_centered": P_centered, "activity": activity,
                "body_names": body_names, "body_edges": body_edges,
                "fps": fps, "start": start, "end": end,
            })
            print(f"  Loaded {condition}/{phase_name}: "
                  f"{P_phase.shape[0]} frames ({P_phase.shape[0]/fps:.0f}s)")

    if not all_phases:
        print(f"  [SKIP] No data found for {subject}")
        return {}

    # Global threshold across all phases
    all_activity  = np.concatenate([p["activity"] for p in all_phases])
    global_thresh = float(np.percentile(all_activity, ACTIVITY_PERCENTILE))
    print(f"  Global threshold (p{ACTIVITY_PERCENTILE}): {global_thresh:.4f}")

    summary = {}
    for p in all_phases:
        condition, phase_name = p["condition"], p["phase"]
        out_dir = OUTPUT_BASE / subject / condition / phase_name
        out_dir.mkdir(parents=True, exist_ok=True)

        movement = detect_activity_frames(p["activity"], global_thresh, MIN_VEL_FRAMES)
        segments, spans = segment_by_movement(
            p["P_centered"], movement,
            PAD_FRAMES, MIN_SEG_LEN, MAX_SEG_LEN, MIN_TOUCH_CHAIN)

        # Score + sort
        scores   = [score_segment(s, p["body_names"]) for s in segments]
        order    = np.argsort(-np.array(scores)) if scores else []
        segments = [segments[i] for i in order]
        spans    = [spans[i]    for i in order]
        scores   = [scores[i]   for i in order]

        # Save plot
        plot_activity(
            p["activity"], movement, spans, global_thresh,
            p["fps"], f"{subject} {condition} {phase_name}",
            out_dir / "activity_and_segments.png")

        # Save log
        log_lines = [
            f"Subject: {subject}  Condition: {condition}  Phase: {phase_name}",
            f"FPS: {p['fps']:.1f}  |  Frames: {len(movement)}",
            f"Global threshold (p{ACTIVITY_PERCENTILE}): {global_thresh:.4f}",
            f"Segments: {len(segments)}", "",
            f"{'#':>3}  {'start':>6}  {'end':>6}  {'len':>5}  {'sec':>5}  {'score':>8}",
            "-" * 45,
        ]
        for i, (seg, (s, e), sc) in enumerate(zip(segments, spans, scores)):
            L = e - s + 1
            log_lines.append(f"{i:>3}  {s:>6}  {e:>6}  {L:>5}  {L/p['fps']:>5.1f}  {sc:>8.4f}")
        with open(out_dir / "segmentation_log.txt", "w") as f:
            f.write("\n".join(log_lines))

        # Save .npz files
        for i, (seg, (s, e), sc) in enumerate(zip(segments, spans, scores)):
            np.savez_compressed(
                str(out_dir / f"seg_{i:03d}_frames_{s}_{e}.npz"),
                poses=seg, start_frame=s, end_frame=e,
                score=sc, fps=p["fps"],
                joint_names=np.array(p["body_names"], dtype=object),
                condition=condition, phase=phase_name, subject=subject,
            )

        dur = (p["end"] - p["start"]) / p["fps"]
        segs_per_min = len(segments) / (dur / 60) if dur > 0 else 0
        print(f"  {condition}/{phase_name}: {len(segments)} segs  "
              f"({dur:.0f}s, {segs_per_min:.1f}/min)")
        summary[f"{condition}/{phase_name}"] = len(segments)

    return summary


# ============================================================
# MAIN
# ============================================================
def main():
    print("="*55)
    print("BATCH SEGMENTATION — ALL SUBJECTS")
    print(f"Subjects: {len(SUBJECTS)}")
    print("="*55)

    grand_total = 0
    all_summary = {}

    for subject in SUBJECTS:
        try:
            summary = process_subject(subject)
            all_summary[subject] = summary
            total = sum(summary.values())
            grand_total += total
        except Exception as e:
            print(f"  [ERROR] {subject}: {e}")
            import traceback; traceback.print_exc()

    # Final summary
    print()
    print("="*55)
    print("FINAL SUMMARY")
    print("="*55)
    print(f"{'Subject':>8}  {'tsst/talk':>10}  {'tsst/math':>10}  {'ftsst/talk':>11}  {'Total':>6}")
    print("-"*55)
    for subject in SUBJECTS:
        s = all_summary.get(subject, {})
        tt = s.get("tsst/talk", 0)
        tm = s.get("tsst/math", 0)
        ft = s.get("ftsst/talk", 0)
        tot = tt + tm + ft
        print(f"{subject:>8}  {tt:>10}  {tm:>10}  {ft:>11}  {tot:>6}")
    print("-"*55)
    print(f"{'TOTAL':>8}  {'':>10}  {'':>10}  {'':>11}  {grand_total:>6}")


if __name__ == "__main__":
    main()