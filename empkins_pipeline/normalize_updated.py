
"""
step3_normalize_updated.py
=========================
Normalize Empkins segments (21 joints, 3D) to BlazePose 33-joint format
matching the MA-52 standing model training data coordinate space.

Main change vs the old version:
  - The old align step only centered + flipped Y + scaled torso.
  - This version also aligns BODY ORIENTATION.
  - If you provide one MA-52 reference skeleton, it additionally fits a
    similarity transform to that MA-52 frame using matched control joints.

Run:
    python empkins_processing/step3_normalize_updated.py
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
SEGMENTS_DIR = Path("empkins_processing/segments") / SUBJECT
OUTPUT_DIR   = Path("empkins_processing/normalized_v2") / SUBJECT

# Mean MA-52 torso length (hip midpoint -> shoulder midpoint)
MA52_TORSO_LENGTH = 0.42

# Optional:
# Set this to ONE MA-52 standing skeleton already in the MA-52 space.
# It can be:
#   - a single frame: (33, 3)
#   - or a sequence:  (T, 33, 3) and frame 0 will be used
MA52_TEMPLATE_PATH = Path("empkins_processing/ma52_mean_pose.npy")
# Example:
# MA52_TEMPLATE_PATH = Path("ma52_reference/standing_ref.npy")

MAX_ANIM_SEGS = 3
ANIMATE_STEP  = 2


# ============================================================
# BlazePose joint indices
# ============================================================
BP = {
    "nose": 0,
    "l_eye_inner": 1, "l_eye": 2, "l_eye_outer": 3,
    "r_eye_inner": 4, "r_eye": 5, "r_eye_outer": 6,
    "l_ear": 7, "r_ear": 8,
    "mouth_l": 9, "mouth_r": 10,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13,    "r_elbow": 14,
    "l_wrist": 15,    "r_wrist": 16,
    "l_pinky": 17,    "r_pinky": 18,
    "l_index": 19,    "r_index": 20,
    "l_thumb": 21,    "r_thumb": 22,
    "l_hip": 23,      "r_hip": 24,
    "l_knee": 25,     "r_knee": 26,
    "l_ankle": 27,    "r_ankle": 28,
    "l_heel": 29,     "r_heel": 30,
    "l_foot": 31,     "r_foot": 32,
}

BLAZEPOSE_EDGES = [
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
    (0, 7), (0, 8), (0, 9), (0, 10),
]

# Stable joints shared by your mapped skeleton and MA-52-like body layout
CONTROL_JOINTS = [
    BP["l_shoulder"], BP["r_shoulder"],
    BP["l_elbow"],    BP["r_elbow"],
    BP["l_wrist"],    BP["r_wrist"],
    BP["l_hip"],      BP["r_hip"],
    BP["l_knee"],     BP["r_knee"],
    BP["l_ankle"],    BP["r_ankle"],
]


# ============================================================
# STEP 1: Map Empkins 21 joints -> BlazePose 33 joints
# ============================================================
def empkins_to_blazepose(poses_21, joint_names_21):
    """poses_21: (T, 21, 3) -> poses_33: (T, 33, 3)"""
    T   = poses_21.shape[0]
    P   = np.zeros((T, 33, 3), dtype=np.float32)
    n2i = {n: i for i, n in enumerate(joint_names_21)}

    def get(name):
        if name in n2i:
            return poses_21[:, n2i[name], :]
        return np.zeros((T, 3), dtype=np.float32)

    # Direct body joint mappings
    P[:, BP["l_shoulder"]] = get("LeftShoulder")
    P[:, BP["r_shoulder"]] = get("RightShoulder")
    P[:, BP["l_elbow"]]    = get("LeftArm")
    P[:, BP["r_elbow"]]    = get("RightArm")
    P[:, BP["l_wrist"]]    = get("LeftForeArm")
    P[:, BP["r_wrist"]]    = get("RightForeArm")
    P[:, BP["l_hip"]]      = get("LeftUpLeg")
    P[:, BP["r_hip"]]      = get("RightUpLeg")
    P[:, BP["l_knee"]]     = get("LeftLeg")
    P[:, BP["r_knee"]]     = get("RightLeg")
    P[:, BP["l_ankle"]]    = get("LeftFoot")
    P[:, BP["r_ankle"]]    = get("RightFoot")

    # Hand joints
    l_hand = get("LeftHand")
    r_hand = get("RightHand")
    for j in ["l_pinky", "l_index", "l_thumb"]:
        P[:, BP[j]] = l_hand
    for j in ["r_pinky", "r_index", "r_thumb"]:
        P[:, BP[j]] = r_hand

    # Face joints from Head + Neck
    head  = get("Head")
    neck  = get("Neck")
    l_sho = get("LeftShoulder")
    r_sho = get("RightShoulder")

    neck_to_head = head - neck
    head_len = np.maximum(np.linalg.norm(
        neck_to_head, axis=1, keepdims=True), 1e-6)
    head_dir = neck_to_head / head_len

    lat      = l_sho - r_sho
    lat_dir  = lat / np.maximum(
        np.linalg.norm(lat, axis=1, keepdims=True), 1e-6)

    sho_w      = np.linalg.norm(l_sho - r_sho, axis=1, keepdims=True)
    eye_spread = 0.08 * sho_w

    nose = head + 0.3 * head_len * head_dir
    P[:, BP["nose"]]        = nose
    l_eye = nose + eye_spread * lat_dir * 0.4
    r_eye = nose - eye_spread * lat_dir * 0.4
    P[:, BP["l_eye_inner"]] = l_eye * 0.8 + nose * 0.2
    P[:, BP["l_eye"]]       = l_eye
    P[:, BP["l_eye_outer"]] = l_eye * 0.8 + nose * 0.2
    P[:, BP["r_eye_inner"]] = r_eye * 0.8 + nose * 0.2
    P[:, BP["r_eye"]]       = r_eye
    P[:, BP["r_eye_outer"]] = r_eye * 0.8 + nose * 0.2
    ear_spread = eye_spread * 1.5
    P[:, BP["l_ear"]]   = head + ear_spread * lat_dir
    P[:, BP["r_ear"]]   = head - ear_spread * lat_dir
    mouth = head - 0.1 * head_len * head_dir
    P[:, BP["mouth_l"]] = mouth + eye_spread * 0.3 * lat_dir
    P[:, BP["mouth_r"]] = mouth - eye_spread * 0.3 * lat_dir

    # Foot joints from ankles + body forward
    l_ankle = get("LeftFoot")
    r_ankle = get("RightFoot")
    l_hip   = get("LeftUpLeg")
    r_hip   = get("RightUpLeg")

    right = r_hip - l_hip
    right_dir = right / np.maximum(
        np.linalg.norm(right, axis=1, keepdims=True), 1e-6)
    up    = neck - 0.5 * (l_hip + r_hip)
    up_dir = up / np.maximum(
        np.linalg.norm(up, axis=1, keepdims=True), 1e-6)
    fwd = np.cross(right_dir, up_dir)
    fwd_dir = fwd / np.maximum(
        np.linalg.norm(fwd, axis=1, keepdims=True), 1e-6)

    shin_len = np.maximum(np.linalg.norm(
        l_ankle - get("LeftLeg"), axis=1, keepdims=True), 1e-6)

    P[:, BP["l_heel"]] = l_ankle - 0.03 * shin_len * fwd_dir
    P[:, BP["r_heel"]] = r_ankle - 0.03 * shin_len * fwd_dir
    P[:, BP["l_foot"]] = l_ankle + 0.10 * shin_len * fwd_dir
    P[:, BP["r_foot"]] = r_ankle + 0.10 * shin_len * fwd_dir

    return P


# ============================================================
# STEP 2: Lower-body neutralization
# ============================================================
def _norm_vec(v, eps=1e-8):
    n = np.linalg.norm(v)
    return v / n if n >= eps else np.zeros_like(v)


def _norm_arr(v, axis=-1, keepdims=False, eps=1e-8):
    n = np.linalg.norm(v, axis=axis, keepdims=keepdims)
    return np.maximum(n, eps)


def neutralize_lower_body(seq33):
    """Freeze legs to neutral standing pose from first frame."""
    out = seq33.copy()
    f0  = out[0]

    lhip = f0[BP["l_hip"]]
    rhip = f0[BP["r_hip"]]
    lsho = f0[BP["l_shoulder"]]
    rsho = f0[BP["r_shoulder"]]

    pelvis = 0.5 * (lhip + rhip)
    sho_c  = 0.5 * (lsho + rsho)

    right = _norm_vec(rhip - lhip)
    up    = _norm_vec(sho_c - pelvis)
    fwd   = _norm_vec(np.cross(right, up))
    right = _norm_vec(np.cross(up, fwd))
    down  = -up

    l_thigh = np.linalg.norm(f0[BP["l_knee"]]  - f0[BP["l_hip"]])
    r_thigh = np.linalg.norm(f0[BP["r_knee"]]  - f0[BP["r_hip"]])
    l_shin  = np.linalg.norm(f0[BP["l_ankle"]] - f0[BP["l_knee"]])
    r_shin  = np.linalg.norm(f0[BP["r_ankle"]] - f0[BP["r_knee"]])
    thigh   = 0.5 * (l_thigh + r_thigh)
    shin    = 0.5 * (l_shin  + r_shin)

    toe_dir = ((f0[BP["l_foot"]] - f0[BP["l_ankle"]]) +
               (f0[BP["r_foot"]] - f0[BP["r_ankle"]]))
    if np.dot(toe_dir, fwd) < 0:
        fwd = -fwd

    lknee  = lhip  + thigh * down + 0.02 * thigh * fwd
    rknee  = rhip  + thigh * down + 0.02 * thigh * fwd
    lankle = lknee + shin  * down
    rankle = rknee + shin  * down

    neutral = {
        BP["l_knee"]:  lknee,
        BP["r_knee"]:  rknee,
        BP["l_ankle"]: lankle,
        BP["r_ankle"]: rankle,
        BP["l_heel"]:  lankle - 0.03 * shin * fwd,
        BP["r_heel"]:  rankle - 0.03 * shin * fwd,
        BP["l_foot"]:  lankle + 0.10 * shin * fwd,
        BP["r_foot"]:  rankle + 0.10 * shin * fwd,
    }

    for t in range(out.shape[0]):
        for j, xyz in neutral.items():
            out[t, j] = xyz

    return out


# ============================================================
# STEP 3: Better MA-52 alignment
# ============================================================
def similarity_transform(from_pts, to_pts, allow_reflection=False):
    """
    Compute similarity transform (s, R, t) such that:
        to ≈ s * R @ from + t
    """
    from_pts = np.asarray(from_pts, float)
    to_pts   = np.asarray(to_pts, float)
    assert from_pts.shape == to_pts.shape and from_pts.shape[1] == 3
    N = from_pts.shape[0]
    assert N >= 3, "Need at least 3 non-collinear correspondences."

    mu_from = from_pts.mean(axis=0)
    mu_to   = to_pts.mean(axis=0)

    X = from_pts - mu_from
    Y = to_pts   - mu_to

    Sigma = (Y.T @ X) / N
    U, D, Vt = np.linalg.svd(Sigma)
    R = U @ Vt

    if not allow_reflection and np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    var_X = (X ** 2).sum() / N
    s = (D if allow_reflection else np.abs(D)).sum() / max(var_X, 1e-8)
    t = mu_to - s * (R @ mu_from)

    pred = (s * (R @ from_pts.T).T) + t
    err = np.sqrt(np.mean(np.sum((to_pts - pred) ** 2, axis=1)))
    return s, R, t, err


def _frame_pelvis(frame33):
    return 0.5 * (frame33[BP["l_hip"]] + frame33[BP["r_hip"]])


def _frame_shoulder_mid(frame33):
    return 0.5 * (frame33[BP["l_shoulder"]] + frame33[BP["r_shoulder"]])


def _torso_length(frame33):
    return np.linalg.norm(_frame_shoulder_mid(frame33) - _frame_pelvis(frame33))


def _compute_body_axes(frame33):
    """
    Build a stable body frame from the first frame:
      X = subject right
      Y = subject down
      Z = subject forward
    """
    pelvis = _frame_pelvis(frame33)
    sho_mid = _frame_shoulder_mid(frame33)

    hip_right = frame33[BP["r_hip"]] - frame33[BP["l_hip"]]
    sho_right = frame33[BP["r_shoulder"]] - frame33[BP["l_shoulder"]]
    right = _norm_vec(0.5 * (hip_right + sho_right))

    up = _norm_vec(sho_mid - pelvis)
    forward = _norm_vec(np.cross(right, up))

    # If the cross product is unstable, fall back to a default forward
    if np.linalg.norm(forward) < 1e-6:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Fix the 180° ambiguity using face/toe direction when available
    face_vec = frame33[BP["nose"]] - sho_mid
    toe_vec = 0.5 * (frame33[BP["l_foot"]] + frame33[BP["r_foot"]]) \
            - 0.5 * (frame33[BP["l_heel"]] + frame33[BP["r_heel"]])
    facing_hint = face_vec + 0.25 * toe_vec
    if np.dot(facing_hint, forward) < 0:
        forward = -forward

    # Re-orthonormalize
    right = _norm_vec(np.cross(up, forward))
    forward = _norm_vec(np.cross(right, up))
    down = -up

    return right.astype(np.float32), down.astype(np.float32), forward.astype(np.float32)


def _canonical_body_align(seq33):
    """
    Fallback when no MA-52 template is available:
    1) center each frame at pelvis
    2) rotate the whole sequence using the first-frame body axes
    3) scale by torso length to MA-52 mean torso
    """
    out = seq33.copy().astype(np.float32)

    pelvis_t = 0.5 * (out[:, BP["l_hip"], :] + out[:, BP["r_hip"], :])
    centered = out - pelvis_t[:, None, :]

    right, down, forward = _compute_body_axes(out[0])
    basis = np.stack([right, down, forward], axis=1)  # world -> canonical via dot products

    aligned = np.einsum("tjc,ck->tjk", centered, basis)

    torso = _torso_length(aligned[0])
    if torso > 1e-6:
        aligned *= (target_torso := MA52_TORSO_LENGTH) / torso

    return aligned


def _apply_similarity(seq33, s, R, t):
    return (s * np.einsum("ij,tbj->tbi", R, seq33)) + t[None, None, :]


def _valid_control_mask(src_pts, tgt_pts):
    src_ok = np.isfinite(src_pts).all(axis=1)
    tgt_ok = np.isfinite(tgt_pts).all(axis=1)

    # reject all-zero control joints when mapping created empty placeholders
    src_nonzero = np.linalg.norm(src_pts, axis=1) > 1e-8
    tgt_nonzero = np.linalg.norm(tgt_pts, axis=1) > 1e-8

    return src_ok & tgt_ok & src_nonzero & tgt_nonzero


def _load_ma52_reference(template_path):
    if template_path is None:
        return None

    arr = np.load(str(template_path))
    if arr.ndim == 3:
        ref = arr[0]
    elif arr.ndim == 2:
        ref = arr
    else:
        raise ValueError(f"Unsupported MA-52 template shape: {arr.shape}")

    if ref.shape != (33, 3):
        raise ValueError(
            f"MA-52 template must be (33,3) or (T,33,3), got {arr.shape}"
        )

    return ref.astype(np.float32)


def align_to_ma52_space(seq33, target_torso=MA52_TORSO_LENGTH, ma52_ref33=None):
    """
    Better alignment than the old version.

    Case A: if ma52_ref33 is provided
        - fit similarity transform from source first frame to MA-52 reference
        - apply that transform to the whole sequence

    Case B: if no template is provided
        - center each frame by pelvis
        - rotate sequence to a canonical body frame
        - scale torso to target_torso

    Returns:
        aligned_seq, info_dict
    """
    out = seq33.copy().astype(np.float32)

    if ma52_ref33 is None:
        aligned = _canonical_body_align(out)
        return aligned, {"mode": "canonical_body_align"}

    src0 = out[0]
    tgt0 = ma52_ref33.astype(np.float32)

    src_ctrl = src0[CONTROL_JOINTS]
    tgt_ctrl = tgt0[CONTROL_JOINTS]

    mask = _valid_control_mask(src_ctrl, tgt_ctrl)
    if mask.sum() < 3:
        raise ValueError(
            f"Need >=3 valid control joints for template alignment, got {mask.sum()}"
        )

    s, R, t, err = similarity_transform(
        src_ctrl[mask], tgt_ctrl[mask], allow_reflection=False
    )
    aligned = _apply_similarity(out, s, R, t)

    return aligned, {
        "mode": "template_similarity",
        "scale": float(s),
        "rms_err": float(err),
        "valid_ctrl": int(mask.sum()),
    }


# ============================================================
# ANIMATION
# ============================================================
def animate_blazepose(poses_33, fps, title, step, out_path):
    """Animate (T, 33, 3) — uses MA-52 coordinate space (Y down)."""
    frames = poses_33[::step]
    T      = frames.shape[0]

    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=10, azim=-90)

    def axis_lim(vals):
        lo, hi = vals.min(), vals.max()
        mid = (lo + hi) / 2
        rng = max(hi - lo, 0.1) * 1.3 / 2
        return mid - rng, mid + rng

    ax.set_xlim(*axis_lim(frames[:, :, 0].flatten()))
    ax.set_ylim(*axis_lim(frames[:, :, 2].flatten()))
    ax.set_zlim(*axis_lim(-frames[:, :, 1].flatten()))
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y(up for display)")

    scat, = ax.plot([], [], [], "o", ms=4, color="tomato")
    bone_lines = [ax.plot([], [], [], "-", lw=1.5, color="tomato")[0]
                  for _ in BLAZEPOSE_EDGES]
    title_txt = ax.set_title("")

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
        for b, (i, j) in zip(bone_lines, BLAZEPOSE_EDGES):
            b.set_data([pts[i, 0], pts[j, 0]],
                       [pts[i, 2], pts[j, 2]])
            b.set_3d_properties([-pts[i, 1], -pts[j, 1]])
        title_txt.set_text(f"{title} | f{f * step}")
        return [scat] + bone_lines + [title_txt]

    ani = animation.FuncAnimation(
        fig, update, frames=T, init_func=init,
        interval=max(20, int(1000 / (fps / step))), blit=False
    )
    ani.save(str(out_path), writer="pillow", fps=max(1, int(fps // step)))
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("Step 3: Empkins -> BlazePose 33 + stronger MA-52 alignment")
    print(f"Subject  : {SUBJECT}")
    print(f"Target torso length: {MA52_TORSO_LENGTH}")
    print(f"MA-52 template     : {MA52_TEMPLATE_PATH}")
    print("=" * 60)
    print()

    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)

    ma52_ref33 = _load_ma52_reference(MA52_TEMPLATE_PATH)

    npz_files = sorted(SEGMENTS_DIR.rglob("seg_*.npz"))
    print(f"Found {len(npz_files)} segments")
    print()

    total_saved = 0
    anim_count = {}

    for npz_path in npz_files:
        data        = np.load(str(npz_path), allow_pickle=True)
        poses_21    = data["poses"]
        joint_names = list(data["joint_names"])
        fps         = float(data["fps"])
        condition   = str(data.get("condition", npz_path.parts[-3]))
        phase       = str(data.get("phase",     npz_path.parts[-2]))

        rel_path = npz_path.relative_to(SEGMENTS_DIR)
        out_dir  = OUTPUT_DIR / rel_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        poses_33 = empkins_to_blazepose(poses_21, joint_names)
        poses_33 = neutralize_lower_body(poses_33)
        poses_33, info = align_to_ma52_space(
            poses_33,
            target_torso=MA52_TORSO_LENGTH,
            ma52_ref33=ma52_ref33,
        )

        out_name = npz_path.stem + "_bp33.npy"
        out_path = out_dir / out_name
        np.save(str(out_path), poses_33.astype(np.float32))
        total_saved += 1

        if total_saved <= 5:
            print(f"[{total_saved:04d}] {npz_path.name} -> {info}")

        phase_key = f"{condition}/{phase}"
        if anim_count.get(phase_key, 0) < MAX_ANIM_SEGS:
            gif_path = out_dir / (npz_path.stem + "_bp33.gif")
            title    = f"{SUBJECT}_{condition}_{phase}_{npz_path.stem[-10:]}"
            animate_blazepose(poses_33, fps, title, ANIMATE_STEP, gif_path)
            anim_count[phase_key] = anim_count.get(phase_key, 0) + 1
            print(f"  Animated: {condition}/{phase} — {gif_path.name}")

    print()
    print("=" * 60)
    print(f"Done! {total_saved} segments normalized")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    samples = sorted(OUTPUT_DIR.rglob("*.npy"))
    if samples:
        s = np.load(str(samples[0]))
        print()
        print(f"Verification — {samples[0].name}")
        print(f"  Shape  : {s.shape}")
        print(f"  X range: [{s[:,:,0].min():.3f}, {s[:,:,0].max():.3f}]")
        print(f"  Y range: [{s[:,:,1].min():.3f}, {s[:,:,1].max():.3f}]")
        print(f"  Z range: [{s[:,:,2].min():.3f}, {s[:,:,2].max():.3f}]")

        torso = _torso_length(s[0])
        print(f"  Torso length: {torso:.4f}  (target: {MA52_TORSO_LENGTH})")
        print(f"  Any NaN    : {np.isnan(s).any()}")


if __name__ == "__main__":
    main()
