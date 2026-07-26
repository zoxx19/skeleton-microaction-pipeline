"""
batch_inference.py
==================
Run MMN inference on ALL normalized segments from ALL subjects.

Run:
    python empkins_processing/batch_inference.py
"""

import sys
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================
SUBJECTS = [
    "VP_01","VP_02","VP_03","VP_04","VP_05","VP_06","VP_07",
    "VP_08","VP_09","VP_10","VP_11","VP_12","VP_13","VP_14",
    "VP_15","VP_16","VP_17","VP_18","VP_19","VP_20","VP_21",
]

NORMALIZED_BASE = Path("empkins_processing/normalized")
OUTPUT_DIR      = Path("empkins_processing/inference")
WEIGHTS         = Path("work_dir/train/STANDING_SINGLE_UPPER/runs-70-19180.pt")

TIME_STEPS   = 96
COORD_DIM    = 3
CENTER_JOINT = 23
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K        = 5

LABEL_NAMES = {
    0:"shaking body", 1:"sitting straightly", 2:"shrugging",
    3:"turning around", 4:"rising up", 5:"bowing head", 6:"head up",
    7:"tilting head", 8:"turning head", 9:"nodding", 10:"shaking head",
    11:"scratching arms", 12:"playing objects", 13:"putting hands together",
    14:"rubbing hands", 15:"pointing oneself", 16:"clenching fist",
    17:"stretching arms", 18:"retracting arms", 19:"waving",
    20:"spreading hands", 21:"hands touching fingers",
    22:"other finger movements", 23:"illustrative gestures",
    24:"scratching or touching neck", 25:"scratching or touching chest",
    26:"scratching or touching back", 27:"scratching or touching shoulder",
    28:"arms akimbo", 29:"crossing arms", 30:"playing or tidying hair",
    31:"scratching or touching hindbrain", 32:"scratching or touching forehead",
    33:"scratching or touching face", 34:"rubbing eyes", 35:"touching nose",
    36:"touching ears", 37:"covering face", 38:"covering mouth",
    39:"pushing glasses",
}


# ============================================================
# PREPROCESSING
# ============================================================
def preprocess(x):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)[..., :COORD_DIM]
    x = x - x[0, CENTER_JOINT].copy()
    flat  = x.reshape(-1, x.shape[-1])
    scale = np.maximum(flat.max(0) - flat.min(0), 1e-6)
    flat  = (flat - flat.min(0)) / scale * 2.0 - 1.0
    x     = flat.reshape(x.shape)
    T     = x.shape[0]
    idx   = np.linspace(0, T-1, TIME_STEPS).astype(int)
    x     = x[idx]
    index_t = 2.0 * idx.astype(np.float32) / max(1, idx.max()+1) - 1.0
    return x.transpose(2,0,1).astype(np.float32)[:,:,:,None], index_t


# ============================================================
# LOAD MODEL
# ============================================================
def load_model():
    sys.path.insert(0, str(Path(".").resolve()))
    from model.MMN import MMN_
    model = MMN_(in_channels=3, num_classes=40, num_people=1, num_points=33,
                 kernel_size=3, num_heads=4, mlp_ratio=2.0,
                 drop=0.0, head_drop=0.1, drop_path=0.2, index_t=True)
    weights = torch.load(str(WEIGHTS), map_location=DEVICE, weights_only=True)
    model.load_state_dict(weights, strict=False)
    model.to(DEVICE); model.eval()
    return model


# ============================================================
# MAIN
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("="*55)
    print("BATCH INFERENCE — ALL SUBJECTS")
    print(f"Device: {DEVICE}")
    print("="*55)

    model = load_model()
    print(f"Loaded model: {WEIGHTS}")
    print()

    all_results = []
    grand_total = 0

    for subject in SUBJECTS:
        npy_files = sorted((NORMALIZED_BASE / subject).rglob("*.npy"))
        if not npy_files:
            print(f"[SKIP] {subject} — no normalized files")
            continue

        subj_results = []
        for npy_path in npy_files:
            parts     = npy_path.relative_to(NORMALIZED_BASE / subject).parts
            condition = parts[0] if len(parts) > 1 else "unknown"
            phase     = parts[1] if len(parts) > 2 else "unknown"

            x_raw = np.load(str(npy_path))
            x, idx_t = preprocess(x_raw)
            x_t   = torch.from_numpy(x).unsqueeze(0).float().to(DEVICE)
            idx_t = torch.from_numpy(idx_t).unsqueeze(0).float().to(DEVICE)

            with torch.no_grad():
                probs = F.softmax(model(x_t, idx_t), dim=1).squeeze(0).cpu().numpy()

            top5_idx  = np.argsort(-probs)[:TOP_K]
            top5_prob = probs[top5_idx]

            result = {
                "subject":   subject,
                "condition": condition,
                "phase":     phase,
                "file":      npy_path.name,
                "top1_idx":  int(top5_idx[0]),
                "top1_name": LABEL_NAMES.get(int(top5_idx[0]), "?"),
                "top1_prob": float(top5_prob[0]),
                "top5_idx":  top5_idx.tolist(),
                "top5_names":[LABEL_NAMES.get(i,"?") for i in top5_idx],
                "top5_probs":top5_prob.tolist(),
                "all_probs": probs.tolist(),
            }
            subj_results.append(result)
            all_results.append(result)

        # Save per-subject results
        subj_dir = OUTPUT_DIR / subject
        subj_dir.mkdir(parents=True, exist_ok=True)
        with open(subj_dir / "inference_results.pkl", "wb") as f:
            pickle.dump(subj_results, f)

        grand_total += len(subj_results)
        dist = Counter(r["top1_name"] for r in subj_results)
        top3 = dist.most_common(3)
        print(f"  {subject}: {len(subj_results)} segs | "
              + " | ".join(f"{n}:{c}" for n, c in top3))

    # Save all results
    with open(OUTPUT_DIR / "all_results.pkl", "wb") as f:
        pickle.dump(all_results, f)

    # Global label distribution
    print()
    print("="*55)
    print(f"GLOBAL LABEL DISTRIBUTION ({grand_total} segments, {len(SUBJECTS)} subjects)")
    print("="*55)
    dist = Counter(r["top1_name"] for r in all_results)
    for label, count in dist.most_common():
        pct = 100 * count / grand_total
        bar = "█" * int(pct / 2)
        print(f"  {label:35s}: {count:4d} ({pct:5.1f}%) {bar}")

    # Save summary CSV
    with open(OUTPUT_DIR / "label_distribution.csv", "w") as f:
        f.write("label,count,pct\n")
        for label, count in dist.most_common():
            f.write(f"{label},{count},{100*count/grand_total:.2f}\n")

    print()
    print(f"Saved all_results.pkl and label_distribution.csv to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()