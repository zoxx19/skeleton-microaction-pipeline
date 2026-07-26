"""
step4_inference.py
==================
Run the trained MMN standing model on normalized Empkins segments
and collect predicted micro-action labels.

Run:
    python empkins_processing/step4_inference.py
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ============================================================
# CONFIG
# ============================================================
SUBJECT       = "VP_04"
NORMALIZED_DIR = Path("empkins_processing/normalized_v2") / SUBJECT
OUTPUT_DIR     = Path("empkins_processing/inference") / SUBJECT

# Best trained model weights
WEIGHTS = Path("work_dir/train/STANDING_SINGLE_UPPER/runs-70-19180.pt")

# Label names file
LABEL_FILE = Path("data/standing_ma/annotations/label_name.txt")

# Feeder settings (must match training config)
TIME_STEPS  = 96
COORD_DIM   = 3
CENTER_JOINT = 23   # left hip

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Top-K predictions to save
TOP_K = 5


# ============================================================
# LABEL NAMES
# ============================================================
def load_label_names(label_file):
    """Load label index -> name mapping."""
    labels = {}
    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                labels[int(parts[0])] = parts[1]
    return labels


# ============================================================
# FEEDER PREPROCESSING
# (mirrors feeder_single.py exactly)
# ============================================================
def preprocess(x, time_steps=96, coord_dim=3, center_joint=23):
    """
    Apply feeder preprocessing to a single .npy sample.
    x: (T, V, C) numpy array
    Returns: (C, T, V, 1) float32 tensor, index_t (T,) float32 tensor
    """
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Channel selection
    if x.shape[-1] >= 4:
        x = x[..., :coord_dim]
    elif x.shape[-1] == coord_dim:
        pass
    else:
        raise ValueError(f"Unexpected channels: {x.shape[-1]}")

    # Center by joint at first frame
    center = x[0, center_joint].copy()
    x = x - center

    # Normalize to [-1, 1]
    flat  = x.reshape(-1, x.shape[-1])
    minv  = flat.min(axis=0)
    maxv  = flat.max(axis=0)
    scale = np.maximum(maxv - minv, 1e-6)
    flat  = (flat - minv) / scale
    flat  = flat * 2.0 - 1.0
    x     = flat.reshape(x.shape)

    # Uniform temporal sampling to time_steps
    T = x.shape[0]
    if T <= 1:
        idx = np.zeros(time_steps, dtype=int)
    else:
        idx = np.linspace(0, T - 1, time_steps).astype(int)
    x = x[idx]

    # Time index
    denom   = float(max(1, idx.max() + 1))
    index_t = 2.0 * idx.astype(np.float32) / denom - 1.0

    # (C, T, V, 1)
    x = x.transpose(2, 0, 1).astype(np.float32)
    x = x[:, :, :, None]

    return x, index_t


# ============================================================
# LOAD MODEL
# ============================================================
def load_model(weights_path, device):
    """Load MMN model with trained weights."""
    # Import model
    sys.path.insert(0, str(Path(".").resolve()))
    from model.MMN import MMN_

    model = MMN_(
        in_channels=3,
        num_classes=40,
        num_people=1,
        num_points=33,
        kernel_size=3,
        num_heads=4,
        mlp_ratio=2.0,
        drop=0.0,
        head_drop=0.1,
        drop_path=0.2,
        index_t=True,
    )

    weights = torch.load(str(weights_path),
                         map_location=device,
                         weights_only=True)
    model.load_state_dict(weights, strict=False)
    model.to(device)
    model.eval()

    print(f"Loaded model from: {weights_path}")
    print(f"Device: {device}")
    return model


# ============================================================
# INFERENCE ON ONE SEGMENT
# ============================================================
def predict_segment(model, npy_path, device):
    """
    Run inference on one .npy segment.
    Returns: scores (40,), top5_indices, top5_probs
    """
    x_raw = np.load(str(npy_path))
    x, index_t = preprocess(x_raw, TIME_STEPS, COORD_DIM, CENTER_JOINT)

    # Add batch dim
    x_t       = torch.from_numpy(x).unsqueeze(0).float().to(device)      # (1, C, T, V, 1)
    idx_t     = torch.from_numpy(index_t).unsqueeze(0).float().to(device) # (1, T)

    with torch.no_grad():
        output = model(x_t, idx_t)   # (1, 40)
        probs  = F.softmax(output, dim=1).squeeze(0).cpu().numpy()  # (40,)

    top5_idx  = np.argsort(-probs)[:TOP_K]
    top5_prob = probs[top5_idx]

    return probs, top5_idx, top5_prob


# ============================================================
# MAIN
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Step 4: MMN Inference on Empkins segments")
    print(f"Subject  : {SUBJECT}")
    print(f"Weights  : {WEIGHTS}")
    print(f"Device   : {DEVICE}")
    print(f"{'='*60}")
    print()

    # Load label names
    if LABEL_FILE.exists():
        label_names = {
            0:  "shaking body",
            1:  "sitting straightly",
            2:  "shrugging",
            3:  "turning around",
            4:  "rising up",
            5:  "bowing head",
            6:  "head up",
            7:  "tilting head",
            8:  "turning head",
            9:  "nodding",
            10: "shaking head",
            11: "scratching arms",
            12: "playing objects",
            13: "putting hands together",
            14: "rubbing hands",
            15: "pointing oneself",
            16: "clenching fist",
            17: "stretching arms",
            18: "retracting arms",
            19: "waving",
            20: "spreading hands",
            21: "hands touching fingers",
            22: "other finger movements",
            23: "illustrative gestures",
            24: "scratching or touching neck",
            25: "scratching or touching chest",
            26: "scratching or touching back",
            27: "scratching or touching shoulder",
            28: "arms akimbo",
            29: "crossing arms",
            30: "playing or tidying hair",
            31: "scratching or touching hindbrain",
            32: "scratching or touching forehead",
            33: "scratching or touching face",
            34: "rubbing eyes",
            35: "touching nose",
            36: "touching ears",
            37: "covering face",
            38: "covering mouth",
            39: "pushing glasses",
        }
        print(f"Using correct 40-class label mapping")
    else:
        label_names = {i: f"class_{i}" for i in range(40)}
        print("Label file not found, using generic names")
    print()

    # Load model
    model = load_model(WEIGHTS, DEVICE)
    print()

    # Find all normalized segments
    npy_files = sorted(NORMALIZED_DIR.rglob("*.npy"))
    print(f"Found {len(npy_files)} normalized segments")
    print()

    # Run inference
    results = []

    for npy_path in npy_files:
        # Parse condition/phase from path
        parts     = npy_path.relative_to(NORMALIZED_DIR).parts
        condition = parts[0] if len(parts) > 1 else "unknown"
        phase     = parts[1] if len(parts) > 2 else "unknown"

        probs, top5_idx, top5_prob = predict_segment(model, npy_path, DEVICE)

        top1_idx  = int(top5_idx[0])
        top1_prob = float(top5_prob[0])
        top1_name = label_names.get(top1_idx, f"class_{top1_idx}")

        result = {
            "file":      str(npy_path.name),
            "condition": condition,
            "phase":     phase,
            "top1_idx":  top1_idx,
            "top1_name": top1_name,
            "top1_prob": top1_prob,
            "top5_idx":  top5_idx.tolist(),
            "top5_names": [label_names.get(i, f"class_{i}") for i in top5_idx],
            "top5_probs": top5_prob.tolist(),
            "all_probs": probs.tolist(),
        }
        results.append(result)

        print(f"[{condition}/{phase}] {npy_path.stem[-25:]}")
        print(f"  Top1: {top1_name} ({top1_prob*100:.1f}%)")
        for i in range(1, min(3, TOP_K)):
            print(f"  Top{i+1}: {label_names.get(top5_idx[i], '?')} "
                  f"({top5_prob[i]*100:.1f}%)")
        print()

    # Save results as pickle
    out_pkl = OUTPUT_DIR / "inference_results.pkl"
    with open(str(out_pkl), "wb") as f:
        pickle.dump(results, f)

    # Save human-readable summary
    out_txt = OUTPUT_DIR / "inference_summary.txt"
    with open(str(out_txt), "w") as f:
        f.write(f"Subject: {SUBJECT}\n")
        f.write(f"Model: {WEIGHTS}\n")
        f.write(f"Total segments: {len(results)}\n\n")

        f.write(f"{'Condition':>8}  {'Phase':>6}  {'Top1 Label':>35}  "
                f"{'Conf':>6}  {'Top2 Label':>35}\n")
        f.write("-" * 100 + "\n")

        for r in results:
            top2_name = r['top5_names'][1] if len(r['top5_names']) > 1 else ""
            top2_prob = r['top5_probs'][1] if len(r['top5_probs']) > 1 else 0.0
            f.write(f"{r['condition']:>8}  {r['phase']:>6}  "
                    f"{r['top1_name']:>35}  {r['top1_prob']*100:>5.1f}%  "
                    f"{top2_name:>35}  {top2_prob*100:>5.1f}%\n")

    print(f"{'='*60}")
    print(f"Saved results: {out_pkl}")
    print(f"Saved summary: {out_txt}")
    print(f"{'='*60}")

    # Print label distribution
    print()
    print("=== Label distribution (top1) ===")
    from collections import Counter
    label_counts = Counter(r["top1_name"] for r in results)
    for label, count in label_counts.most_common(15):
        bar = "█" * count
        print(f"  {label:35s}: {count:3d} {bar}")


if __name__ == "__main__":
    main()