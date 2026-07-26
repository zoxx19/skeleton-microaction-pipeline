import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = os.path.expanduser("~/repos/Micro-Action/Micro-action skeleton/MMN")

DATASET_SPLITS = {
    "train": {
        "input_dir": os.path.join(BASE_DIR, "npy_out_ma", "train"),
        "output_dir": os.path.join(BASE_DIR, "npy_out_ma_neutralized", "train"),
    },
    "val": {
        "input_dir": os.path.join(BASE_DIR, "npy_out_ma", "val"),
        "output_dir": os.path.join(BASE_DIR, "npy_out_ma_neutralized", "val"),
    },
    "test": {
        "input_dir": os.path.join(BASE_DIR, "npy_out_ma", "test"),
        "output_dir": os.path.join(BASE_DIR, "npy_out_ma_neutralized", "test"),
    },
}

SPLITS_TO_PROCESS = ["train", "val", "test"]

SKIP_CROPPED_FILES = False

SAVE_VISUALIZATION_SAMPLE = True
VIS_SAMPLE_INDEX = 0
VIS_FRAME_IDX = 12
VIS_OUTPUT_DIR = os.path.join(BASE_DIR, "neutralization_vis")

# =========================================================
# MA52 / MediaPipe-33 indices
# =========================================================
MA52 = {
    "nose": 0,
    "l_shoulder": 11,
    "r_shoulder": 12,
    "l_elbow": 13,
    "r_elbow": 14,
    "l_wrist": 15,
    "r_wrist": 16,
    "l_pinky": 17,
    "r_pinky": 18,
    "l_index": 19,
    "r_index": 20,
    "l_thumb": 21,
    "r_thumb": 22,
    "l_hip": 23,
    "r_hip": 24,
    "l_knee": 25,
    "r_knee": 26,
    "l_ankle": 27,
    "r_ankle": 28,
    "l_heel": 29,
    "r_heel": 30,
    "l_foot": 31,
    "r_foot": 32,
}

MEDIAPIPE33_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (3, 7), (6, 8),
    (9, 10),

    (11, 12),
    (11, 23), (12, 24),
    (23, 24),

    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21),
    (17, 19),

    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),
    (18, 20),

    (23, 25), (25, 27),
    (27, 29), (29, 31),

    (24, 26), (26, 28),
    (28, 30), (30, 32),
]

# =========================================================
# HELPERS
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def normalize(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n

def set_axes_equal_3d(ax):
    xs = ax.get_xlim3d()
    ys = ax.get_ylim3d()
    zs = ax.get_zlim3d()

    x_mid = np.mean(xs)
    y_mid = np.mean(ys)
    z_mid = np.mean(zs)

    max_range = max(xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]) / 2.0

    ax.set_xlim3d([x_mid - max_range, x_mid + max_range])
    ax.set_ylim3d([y_mid - max_range, y_mid + max_range])
    ax.set_zlim3d([z_mid - max_range, z_mid + max_range])

def get_npy_files(input_dir, skip_cropped=True):
    if not os.path.isdir(input_dir):
        return []

    files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
    if skip_cropped:
        files = [f for f in files if not re.search(r"_\d+_\d+\.npy$", f)]
    return sorted(files)

# =========================================================
# LOWER-BODY NEUTRALIZATION
# =========================================================
def build_body_axes(frame33_xyz):
    lhip = frame33_xyz[MA52["l_hip"]]
    rhip = frame33_xyz[MA52["r_hip"]]
    lsho = frame33_xyz[MA52["l_shoulder"]]
    rsho = frame33_xyz[MA52["r_shoulder"]]

    pelvis = 0.5 * (lhip + rhip)
    shoulder_center = 0.5 * (lsho + rsho)

    right_axis = normalize(rhip - lhip)
    up_axis = normalize(shoulder_center - pelvis)
    forward_axis = normalize(np.cross(right_axis, up_axis))

    right_axis = normalize(np.cross(up_axis, forward_axis))

    return pelvis, right_axis, up_axis, forward_axis

def make_neutral_lower_body_from_frame(frame33):
    P = frame33[:, :3].astype(np.float32)

    pelvis, right_axis, up_axis, forward_axis = build_body_axes(P)
    down_axis = -up_axis

    lhip = P[MA52["l_hip"]].copy()
    rhip = P[MA52["r_hip"]].copy()

    l_thigh = np.linalg.norm(P[MA52["l_knee"]] - P[MA52["l_hip"]])
    r_thigh = np.linalg.norm(P[MA52["r_knee"]] - P[MA52["r_hip"]])
    l_shin = np.linalg.norm(P[MA52["l_ankle"]] - P[MA52["l_knee"]])
    r_shin = np.linalg.norm(P[MA52["r_ankle"]] - P[MA52["r_knee"]])

    thigh = 0.5 * (l_thigh + r_thigh)
    shin = 0.5 * (l_shin + r_shin)

    orig_left_toe_dir = P[MA52["l_foot"]] - P[MA52["l_ankle"]]
    orig_right_toe_dir = P[MA52["r_foot"]] - P[MA52["r_ankle"]]
    toe_dir = orig_left_toe_dir + orig_right_toe_dir

    if np.dot(toe_dir, forward_axis) < 0:
        forward_axis = -forward_axis

    knee_forward = 0.02 * thigh
    toe_forward = 0.10 * shin
    heel_back = 0.03 * shin

    lknee = lhip + thigh * down_axis + knee_forward * forward_axis
    rknee = rhip + thigh * down_axis + knee_forward * forward_axis

    lankle = lknee + shin * down_axis
    rankle = rknee + shin * down_axis

    lheel = lankle - heel_back * forward_axis
    rheel = rankle - heel_back * forward_axis

    lfoot = lankle + toe_forward * forward_axis
    rfoot = rankle + toe_forward * forward_axis

    return {
        25: lknee,
        26: rknee,
        27: lankle,
        28: rankle,
        29: lheel,
        30: rheel,
        31: lfoot,
        32: rfoot,
    }

def neutralize_sequence_lower_body(seq):
    out = seq.copy().astype(np.float32)

    if out.ndim != 3 or out.shape[1] != 33 or out.shape[2] not in (3, 4):
        raise ValueError("Expected (T,33,3) or (T,33,4), got {}".format(out.shape))

    ref_frame = out[0]
    neutral = make_neutral_lower_body_from_frame(ref_frame)

    for t in range(out.shape[0]):
        for j, xyz in neutral.items():
            out[t, j, :3] = xyz

    return out

# =========================================================
# VISUALIZATION
# =========================================================
def plot_frame(ax, frame33, title="", annotate=False):
    P = frame33[:, :3].astype(np.float32)

    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=28)

    for i, j in MEDIAPIPE33_EDGES:
        ax.plot(
            [P[i, 0], P[j, 0]],
            [P[i, 1], P[j, 1]],
            [P[i, 2], P[j, 2]],
            linewidth=2
        )

    if annotate:
        for idx in range(P.shape[0]):
            ax.text(P[idx, 0], P[idx, 1], P[idx, 2], str(idx), fontsize=7)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

def save_compare_frames(before_seq, after_seq, out_path, frame_idx=0, annotate=False):
    before = before_seq[frame_idx]
    after = after_seq[frame_idx]

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    plot_frame(ax1, before, title="Before - frame {}".format(frame_idx), annotate=annotate)

    ax2 = fig.add_subplot(122, projection="3d")
    plot_frame(ax2, after, title="After neutralization - frame {}".format(frame_idx), annotate=annotate)

    all_pts = np.concatenate([before[:, :3], after[:, :3]], axis=0).astype(np.float32)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    pad = 0.05 * (maxs - mins + 1e-9)

    for ax in [ax1, ax2]:
        ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
        ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
        ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
        set_axes_equal_3d(ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# =========================================================
# DATASET PROCESSING
# =========================================================
def process_dataset(input_dir, output_dir, skip_cropped_files=True):
    ensure_dir(output_dir)

    kept = 0
    skipped = 0
    failed = 0

    npy_files = get_npy_files(input_dir, skip_cropped=skip_cropped_files)
    print("[INFO] Found {} .npy files in {}".format(len(npy_files), input_dir))

    for fname in npy_files:
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        try:
            seq = np.load(in_path)

            if not (seq.ndim == 3 and seq.shape[1] == 33 and seq.shape[2] in (3, 4)):
                print("[SKIP wrong shape] {}: {}".format(fname, seq.shape))
                skipped += 1
                continue

            seq_out = neutralize_sequence_lower_body(seq)
            np.save(out_path, seq_out.astype(np.float32))

            kept += 1
            if kept % 100 == 0:
                print("[INFO] processed {} files...".format(kept))

        except Exception as e:
            print("[FAIL] {}: {}".format(fname, e))
            failed += 1

    print("\nDone.")
    print("Kept:    {}".format(kept))
    print("Skipped: {}".format(skipped))
    print("Failed:  {}".format(failed))

def save_one_visualization_per_split(split_name, input_dir, output_dir):
    ensure_dir(VIS_OUTPUT_DIR)

    npy_files = get_npy_files(input_dir, skip_cropped=SKIP_CROPPED_FILES)
    if not npy_files:
        print("[WARN] No files found for visualization in split '{}'".format(split_name))
        return

    idx = min(VIS_SAMPLE_INDEX, len(npy_files) - 1)
    fname = npy_files[idx]

    original_path = os.path.join(input_dir, fname)
    neutral_path = os.path.join(output_dir, fname)

    if not (os.path.exists(original_path) and os.path.exists(neutral_path)):
        print("[WARN] Visualization sample missing for split '{}': {}".format(split_name, fname))
        return

    original_seq = np.load(original_path)
    neutral_seq = np.load(neutral_path)

    frame_idx = min(VIS_FRAME_IDX, original_seq.shape[0] - 1)

    out_png = os.path.join(
        VIS_OUTPUT_DIR,
        "{}_{}_compare.png".format(split_name, os.path.splitext(fname)[0])
    )
    save_compare_frames(
        original_seq,
        neutral_seq,
        out_path=out_png,
        frame_idx=frame_idx,
        annotate=True,
    )
    print("[INFO] Saved visualization: {}".format(out_png))

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    print("[INFO] BASE_DIR: {}".format(BASE_DIR))

    for split_name in SPLITS_TO_PROCESS:
        cfg = DATASET_SPLITS[split_name]

        input_dir = cfg["input_dir"]
        output_dir = cfg["output_dir"]

        print("\n" + "=" * 70)
        print("Processing split: {}".format(split_name))
        print("Input : {}".format(input_dir))
        print("Output: {}".format(output_dir))
        print("=" * 70)

        if not os.path.isdir(input_dir):
            print("[WARN] Input directory does not exist, skipping: {}".format(input_dir))
            continue

        process_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            skip_cropped_files=SKIP_CROPPED_FILES,
        )

        if SAVE_VISUALIZATION_SAMPLE:
            save_one_visualization_per_split(
                split_name=split_name,
                input_dir=input_dir,
                output_dir=output_dir,
            )

    print("\n[INFO] All requested splits finished.")