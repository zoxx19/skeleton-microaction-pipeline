"""
batch_normalize.py
==================
Normalize ALL segments from ALL subjects to BlazePose 33 + MA-52 space.
Uses the exact same normalization logic as step3_normalize_updated.py.

Run:
    python empkins_processing/batch_normalize.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")


# ============================================================
# CONFIG
# ============================================================
SUBJECTS = [
    "VP_01","VP_02","VP_03","VP_04","VP_05","VP_06","VP_07",
    "VP_08","VP_09","VP_10","VP_11","VP_12","VP_13","VP_14",
    "VP_15","VP_16","VP_17","VP_18","VP_19","VP_20","VP_21",
]

SEGMENTS_BASE    = Path("empkins_processing/segments")
OUTPUT_BASE      = Path("empkins_processing/normalized")
MA52_TEMPLATE_PATH = Path("empkins_processing/ma52_mean_pose.npy")
MA52_TORSO_LENGTH  = 0.42

# Control joints — no nose (synthesized, unreliable)
BP = {
    "nose": 0, "l_eye_inner": 1, "l_eye": 2, "l_eye_outer": 3,
    "r_eye_inner": 4, "r_eye": 5, "r_eye_outer": 6,
    "l_ear": 7, "r_ear": 8, "mouth_l": 9, "mouth_r": 10,
    "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16, "l_pinky": 17, "r_pinky": 18,
    "l_index": 19, "r_index": 20, "l_thumb": 21, "r_thumb": 22,
    "l_hip": 23, "r_hip": 24, "l_knee": 25, "r_knee": 26,
    "l_ankle": 27, "r_ankle": 28, "l_heel": 29, "r_heel": 30,
    "l_foot": 31, "r_foot": 32,
}

CONTROL_JOINTS = [
    BP["l_shoulder"], BP["r_shoulder"],
    BP["l_elbow"],    BP["r_elbow"],
    BP["l_wrist"],    BP["r_wrist"],
    BP["l_hip"],      BP["r_hip"],
    BP["l_knee"],     BP["r_knee"],
    BP["l_ankle"],    BP["r_ankle"],
]


# ============================================================
# STEP 1: Map Empkins 21 -> BlazePose 33
# ============================================================
def empkins_to_blazepose(poses_21, joint_names_21):
    T   = poses_21.shape[0]
    P   = np.zeros((T, 33, 3), dtype=np.float32)
    n2i = {n: i for i, n in enumerate(joint_names_21)}

    def get(name):
        return poses_21[:, n2i[name], :] if name in n2i \
               else np.zeros((T, 3), dtype=np.float32)

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

    l_hand, r_hand = get("LeftHand"), get("RightHand")
    for j in ["l_pinky","l_index","l_thumb"]: P[:, BP[j]] = l_hand
    for j in ["r_pinky","r_index","r_thumb"]: P[:, BP[j]] = r_hand

    head, neck   = get("Head"), get("Neck")
    l_sho, r_sho = get("LeftShoulder"), get("RightShoulder")

    neck_to_head = head - neck
    head_len = np.maximum(np.linalg.norm(neck_to_head, axis=1, keepdims=True), 1e-6)
    head_dir = neck_to_head / head_len
    lat_dir  = (l_sho - r_sho) / np.maximum(
        np.linalg.norm(l_sho - r_sho, axis=1, keepdims=True), 1e-6)
    eye_spread = 0.08 * np.linalg.norm(l_sho - r_sho, axis=1, keepdims=True)

    nose = head + 0.3 * head_len * head_dir
    P[:, BP["nose"]]        = nose
    l_eye = nose + eye_spread * lat_dir * 0.4
    r_eye = nose - eye_spread * lat_dir * 0.4
    P[:, BP["l_eye_inner"]] = l_eye*0.8 + nose*0.2
    P[:, BP["l_eye"]]       = l_eye
    P[:, BP["l_eye_outer"]] = l_eye*0.8 + nose*0.2
    P[:, BP["r_eye_inner"]] = r_eye*0.8 + nose*0.2
    P[:, BP["r_eye"]]       = r_eye
    P[:, BP["r_eye_outer"]] = r_eye*0.8 + nose*0.2
    ear_spread = eye_spread * 1.5
    P[:, BP["l_ear"]]   = head + ear_spread * lat_dir
    P[:, BP["r_ear"]]   = head - ear_spread * lat_dir
    mouth = head - 0.1 * head_len * head_dir
    P[:, BP["mouth_l"]] = mouth + eye_spread*0.3*lat_dir
    P[:, BP["mouth_r"]] = mouth - eye_spread*0.3*lat_dir

    l_ankle, r_ankle = get("LeftFoot"), get("RightFoot")
    l_hip,   r_hip   = get("LeftUpLeg"), get("RightUpLeg")
    right_dir = (r_hip-l_hip) / np.maximum(
        np.linalg.norm(r_hip-l_hip, axis=1, keepdims=True), 1e-6)
    up_dir = (neck-0.5*(l_hip+r_hip)) / np.maximum(
        np.linalg.norm(neck-0.5*(l_hip+r_hip), axis=1, keepdims=True), 1e-6)
    fwd = np.cross(right_dir, up_dir)
    fwd_dir = fwd / np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-6)
    shin_len = np.maximum(np.linalg.norm(
        l_ankle - get("LeftLeg"), axis=1, keepdims=True), 1e-6)
    P[:, BP["l_heel"]] = l_ankle - 0.03*shin_len*fwd_dir
    P[:, BP["r_heel"]] = r_ankle - 0.03*shin_len*fwd_dir
    P[:, BP["l_foot"]] = l_ankle + 0.10*shin_len*fwd_dir
    P[:, BP["r_foot"]] = r_ankle + 0.10*shin_len*fwd_dir

    return P


# ============================================================
# STEP 2: Lower-body neutralization
# ============================================================
def _nv(v, eps=1e-8):
    n = np.linalg.norm(v)
    return v/n if n >= eps else np.zeros_like(v)


def neutralize_lower_body(seq33):
    out, f0 = seq33.copy(), seq33[0]
    lhip, rhip = f0[BP["l_hip"]], f0[BP["r_hip"]]
    lsho, rsho = f0[BP["l_shoulder"]], f0[BP["r_shoulder"]]
    pelvis = 0.5*(lhip+rhip)
    sho_c  = 0.5*(lsho+rsho)
    right  = _nv(rhip-lhip)
    up     = _nv(sho_c-pelvis)
    fwd    = _nv(np.cross(right, up))
    right  = _nv(np.cross(up, fwd))
    down   = -up

    l_thigh = np.linalg.norm(f0[BP["l_knee"]] - f0[BP["l_hip"]])
    r_thigh = np.linalg.norm(f0[BP["r_knee"]] - f0[BP["r_hip"]])
    l_shin  = np.linalg.norm(f0[BP["l_ankle"]]- f0[BP["l_knee"]])
    r_shin  = np.linalg.norm(f0[BP["r_ankle"]]- f0[BP["r_knee"]])
    thigh   = 0.5*(l_thigh+r_thigh)
    shin    = 0.5*(l_shin+r_shin)

    toe_dir = ((f0[BP["l_foot"]]-f0[BP["l_ankle"]]) +
               (f0[BP["r_foot"]]-f0[BP["r_ankle"]]))
    if np.dot(toe_dir, fwd) < 0:
        fwd = -fwd

    lknee  = lhip  + thigh*down + 0.02*thigh*fwd
    rknee  = rhip  + thigh*down + 0.02*thigh*fwd
    lankle = lknee + shin*down
    rankle = rknee + shin*down

    neutral = {
        BP["l_knee"]:  lknee,  BP["r_knee"]:  rknee,
        BP["l_ankle"]: lankle, BP["r_ankle"]: rankle,
        BP["l_heel"]:  lankle-0.03*shin*fwd, BP["r_heel"]: rankle-0.03*shin*fwd,
        BP["l_foot"]:  lankle+0.10*shin*fwd, BP["r_foot"]: rankle+0.10*shin*fwd,
    }
    for t in range(out.shape[0]):
        for j, xyz in neutral.items():
            out[t, j] = xyz
    return out


# ============================================================
# STEP 3: Similarity transform + alignment (exact same as your script)
# ============================================================
def similarity_transform(from_pts, to_pts, allow_reflection=False):
    from_pts = np.asarray(from_pts, float)
    to_pts   = np.asarray(to_pts, float)
    N        = from_pts.shape[0]
    mu_from, mu_to = from_pts.mean(0), to_pts.mean(0)
    X, Y = from_pts - mu_from, to_pts - mu_to
    U, D, Vt = np.linalg.svd((Y.T @ X) / N)
    R = U @ Vt
    if not allow_reflection and np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    var_X = (X**2).sum() / N
    s = (D if allow_reflection else np.abs(D)).sum() / max(var_X, 1e-8)
    t = mu_to - s*(R @ mu_from)
    pred = (s*(R @ from_pts.T).T) + t
    err  = float(np.sqrt(np.mean(np.sum((to_pts-pred)**2, axis=1))))
    return s, R, t, err


def _valid_control_mask(src_pts, tgt_pts):
    src_ok = np.isfinite(src_pts).all(axis=1)
    tgt_ok = np.isfinite(tgt_pts).all(axis=1)
    src_nonzero = np.linalg.norm(src_pts, axis=1) > 1e-8
    tgt_nonzero = np.linalg.norm(tgt_pts, axis=1) > 1e-8
    return src_ok & tgt_ok & src_nonzero & tgt_nonzero


def _apply_similarity(seq33, s, R, t):
    return (s * np.einsum("ij,tbj->tbi", R, seq33)) + t[None,None,:]


def _frame_pelvis(f):
    return 0.5*(f[BP["l_hip"]]+f[BP["r_hip"]])

def _frame_sho_mid(f):
    return 0.5*(f[BP["l_shoulder"]]+f[BP["r_shoulder"]])

def _torso_length(f):
    return np.linalg.norm(_frame_sho_mid(f) - _frame_pelvis(f))

def _canonical_body_align(seq33):
    """Fallback when no template: center + rotate to body frame + scale."""
    out = seq33.copy().astype(np.float32)
    pelvis_t = 0.5*(out[:,BP["l_hip"],:]+out[:,BP["r_hip"],:])
    centered = out - pelvis_t[:,None,:]
    f0 = out[0]
    hip_r = f0[BP["r_hip"]]-f0[BP["l_hip"]]
    sho_r = f0[BP["r_shoulder"]]-f0[BP["l_shoulder"]]
    right = _nv(0.5*(hip_r+sho_r))
    up    = _nv(_frame_sho_mid(f0)-_frame_pelvis(f0))
    fwd   = _nv(np.cross(right, up))
    right = _nv(np.cross(up, fwd))
    basis = np.stack([right, -up, fwd], axis=1)
    aligned = np.einsum("tjc,ck->tjk", centered, basis)
    torso = _torso_length(aligned[0])
    if torso > 1e-6:
        aligned *= MA52_TORSO_LENGTH / torso
    return aligned


def align_to_ma52_space(seq33, ma52_ref33=None):
    out = seq33.copy().astype(np.float32)
    if ma52_ref33 is None:
        return _canonical_body_align(out), {"mode": "canonical"}

    src_ctrl = out[0, CONTROL_JOINTS]
    tgt_ctrl = ma52_ref33[CONTROL_JOINTS]
    mask = _valid_control_mask(src_ctrl, tgt_ctrl)

    if mask.sum() < 3:
        # fallback
        return _canonical_body_align(out), {"mode": "canonical_fallback"}

    s, R, t, err = similarity_transform(
        src_ctrl[mask], tgt_ctrl[mask], allow_reflection=False)
    aligned = _apply_similarity(out, s, R, t)

    return aligned, {"mode": "similarity", "scale": float(s), "rms_err": float(err)}


# ============================================================
# MAIN
# ============================================================
def main():
    print("="*55)
    print("BATCH NORMALIZATION — ALL SUBJECTS")
    print(f"Template: {MA52_TEMPLATE_PATH}")
    print("="*55)
    print()

    # Load MA-52 reference
    arr = np.load(str(MA52_TEMPLATE_PATH))
    ma52_ref = (arr[0] if arr.ndim == 3 else arr).astype(np.float32)
    print(f"Loaded MA-52 reference pose: shape={ma52_ref.shape}")
    print()

    grand_total = 0
    errors = []

    for subject in SUBJECTS:
        seg_dir = SEGMENTS_BASE / subject
        if not seg_dir.exists():
            print(f"[SKIP] {subject} — no segments dir")
            continue

        npz_files = sorted(seg_dir.rglob("seg_*.npz"))
        if not npz_files:
            print(f"[SKIP] {subject} — 0 segments")
            continue

        count = 0
        for npz_path in npz_files:
            try:
                data        = np.load(str(npz_path), allow_pickle=True)
                poses_21    = data["poses"]
                joint_names = list(data["joint_names"])
                condition   = str(data.get("condition", npz_path.parts[-3]))
                phase       = str(data.get("phase",     npz_path.parts[-2]))

                out_dir = OUTPUT_BASE / subject / condition / phase
                out_dir.mkdir(parents=True, exist_ok=True)

                # Step 1: joint mapping
                poses_33 = empkins_to_blazepose(poses_21, joint_names)

                # Step 2: lower body neutralization
                poses_33 = neutralize_lower_body(poses_33)

                # Step 3: similarity transform to MA-52 space
                poses_33, info = align_to_ma52_space(poses_33, ma52_ref)

                out_path = out_dir / (npz_path.stem + "_bp33.npy")
                np.save(str(out_path), poses_33.astype(np.float32))
                count += 1

            except Exception as e:
                errors.append(f"{subject}/{npz_path.name}: {e}")

        grand_total += count
        print(f"  {subject}: {count} segments normalized -> {OUTPUT_BASE / subject}")

    print()
    print(f"{'='*55}")
    print(f"Total normalized : {grand_total}")
    print(f"Errors           : {len(errors)}")
    if errors:
        for e in errors[:5]:
            print(f"  {e}")
    print(f"Output: {OUTPUT_BASE}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()