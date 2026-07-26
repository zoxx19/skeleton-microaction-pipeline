"""
step3_normalize.py
==================
Normalize Empkins segments (21 joints, 3D) to BlazePose 33-joint format
matching the MA-52 standing model training data coordinate space.

Pipeline per segment:
  1. Load .npz (T, 21, 3)
  2. Map 21 Empkins joints -> 33 BlazePose joints
  3. Apply lower-body neutralization
  4. Align to MA-52 coordinate space:
     a. Center by hip midpoint
     b. Flip Y axis (Empkins Y-up -> MA-52 Y-down)
     c. Scale so torso length matches MA-52 mean (~0.42)
  5. Save as .npy (T, 33, 3)

Run:
    python empkins_processing/step3_normalize.py
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
OUTPUT_DIR   = Path("empkins_processing/normalized") / SUBJECT

# MA-52 target torso length (hip midpoint to shoulder midpoint)
MA52_TORSO_LENGTH = 0.42

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
# STEP 3: Align to MA-52 coordinate space
# ============================================================
def align_to_ma52_space(seq33, target_torso=MA52_TORSO_LENGTH):
    """
    Align Empkins 33-joint sequence to MA-52 coordinate space.

    Steps:
      a. Center by hip midpoint (per frame)
      b. Flip Y axis (Empkins Y-up -> MA-52 Y-down)
      c. Scale so torso length matches MA-52 mean

    seq33: (T, 33, 3)
    Returns: aligned (T, 33, 3)
    """
    out = seq33.copy().astype(np.float32)

    # a. Center by hip midpoint (per frame)
    hip_mid = 0.5 * (out[:, BP["l_hip"], :] +
                     out[:, BP["r_hip"], :])       # (T, 3)
    out = out - hip_mid[:, None, :]

    # b. Flip Y axis (Y-up -> Y-down)
    out[:, :, 1] *= -1.0

    # c. Scale by torso length from first frame
    sho_mid    = 0.5 * (out[0, BP["l_shoulder"]] +
                        out[0, BP["r_shoulder"]])
    hip_mid_f0 = 0.5 * (out[0, BP["l_hip"]] +
                        out[0, BP["r_hip"]])
    torso_len  = np.linalg.norm(sho_mid - hip_mid_f0)

    if torso_len > 1e-6:
        scale = target_torso / torso_len
        out   = out * scale

    return out


# ============================================================
# ANIMATION
# ============================================================
def animate_blazepose(poses_33, fps, title, step, out_path):
    """Animate (T, 33, 3) — uses MA-52 coordinate space (Y down)."""
    frames = poses_33[::step]
    T      = frames.shape[0]

    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=10, azim=-90)   # front view in Y-down space

    def axis_lim(vals):
        lo, hi = vals.min(), vals.max()
        mid = (lo + hi) / 2
        rng = max(hi - lo, 0.1) * 1.3 / 2
        return mid - rng, mid + rng

    ax.set_xlim(*axis_lim(frames[:, :, 0].flatten()))
    ax.set_ylim(*axis_lim(frames[:, :, 2].flatten()))
    ax.set_zlim(*axis_lim(-frames[:, :, 1].flatten()))  # flip for display
    ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("Y(up for display)")

    scat,      = ax.plot([], [], [], "o", ms=4, color="tomato")
    bone_lines  = [ax.plot([], [], [], "-", lw=1.5, color="tomato")[0]
                   for _ in BLAZEPOSE_EDGES]
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
        # display: X stays, Z as depth, -Y as height (flip Y for display)
        scat.set_data(pts[:, 0], pts[:, 2])
        scat.set_3d_properties(-pts[:, 1])
        for b, (i, j) in zip(bone_lines, BLAZEPOSE_EDGES):
            b.set_data([pts[i, 0], pts[j, 0]],
                       [pts[i, 2], pts[j, 2]])
            b.set_3d_properties([-pts[i, 1], -pts[j, 1]])
        title_txt.set_text(f"{title} | f{f*step}")
        return [scat] + bone_lines + [title_txt]

    ani = animation.FuncAnimation(
        fig, update, frames=T, init_func=init,
        interval=max(20, int(1000/(fps/step))), blit=False
    )
    ani.save(str(out_path), writer="pillow",
             fps=max(1, int(fps//step)))
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"{'='*60}")
    print(f"Step 3: Empkins -> BlazePose 33 + MA-52 space alignment")
    print(f"Subject  : {SUBJECT}")
    print(f"Target torso length: {MA52_TORSO_LENGTH}")
    print(f"{'='*60}")
    print()

    # Clean old output
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)

    npz_files   = sorted(SEGMENTS_DIR.rglob("seg_*.npz"))
    print(f"Found {len(npz_files)} segments")
    print()

    total_saved = 0
    anim_count  = {}

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

        # Step 1: Map 21 -> 33 joints
        poses_33 = empkins_to_blazepose(poses_21, joint_names)

        # Step 2: Lower-body neutralization
        poses_33 = neutralize_lower_body(poses_33)

        # Step 3: Align to MA-52 coordinate space
        poses_33 = align_to_ma52_space(poses_33)

        # Save as .npy (T, 33, 3)
        out_name = npz_path.stem + "_bp33.npy"
        out_path = out_dir / out_name
        np.save(str(out_path), poses_33.astype(np.float32))
        total_saved += 1

        # Animate top N per phase
        phase_key = f"{condition}/{phase}"
        if anim_count.get(phase_key, 0) < MAX_ANIM_SEGS:
            gif_path = out_dir / (npz_path.stem + "_bp33.gif")
            title    = f"{SUBJECT}_{condition}_{phase}_{npz_path.stem[-10:]}"
            animate_blazepose(poses_33, fps, title, ANIMATE_STEP, gif_path)
            anim_count[phase_key] = anim_count.get(phase_key, 0) + 1
            print(f"  Animated: {condition}/{phase} — {gif_path.name}")

    print()
    print(f"{'='*60}")
    print(f"Done! {total_saved} segments normalized")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*60}")

    # Verify sample
    samples = sorted(OUTPUT_DIR.rglob("*.npy"))
    if samples:
        s = np.load(str(samples[0]))
        print()
        print(f"Verification — {samples[0].name}")
        print(f"  Shape  : {s.shape}")
        print(f"  X range: [{s[:,:,0].min():.3f}, {s[:,:,0].max():.3f}]")
        print(f"  Y range: [{s[:,:,1].min():.3f}, {s[:,:,1].max():.3f}]")
        print(f"  Z range: [{s[:,:,2].min():.3f}, {s[:,:,2].max():.3f}]")
        print()

        # Compare torso length to MA-52
        sho_mid = 0.5 * (s[0, BP["l_shoulder"]] + s[0, BP["r_shoulder"]])
        hip_mid = 0.5 * (s[0, BP["l_hip"]]      + s[0, BP["r_hip"]])
        torso   = np.linalg.norm(sho_mid - hip_mid)
        print(f"  Torso length: {torso:.4f}  (MA-52 target: {MA52_TORSO_LENGTH})")
        print(f"  Any NaN    : {np.isnan(s).any()}")


if __name__ == "__main__":
    main()