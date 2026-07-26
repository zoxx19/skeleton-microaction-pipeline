import os
import re
from collections import defaultdict, Counter

# Change these paths if needed
folders = {
    "train": "/home/hpc/iwso/REDACTED_ACCOUNT/repos/Micro-Action/Micro-action skeleton/MMN/data/ma52_train_neutralized",
    "val": "/home/hpc/iwso/REDACTED_ACCOUNT/repos/Micro-Action/Micro-action skeleton/MMN/data/ma52_val_neutralized",
    # "test": "ma52_test_neutralized",   # uncomment if you have it
}

annotation_txt = "/home/hpc/iwso/REDACTED_ACCOUNT/repos/Micro-Action/mar_scripts/manet/mmaction2/data/ma52/train_list_videos.txt"

def parse_pose_name(filename):
    name = os.path.basename(filename)

    m_full = re.match(r"poses_(train|val|test)(\d{4})\.npy$", name)
    if m_full:
        return {
            "original": name,
            "split": m_full.group(1),
            "pose_id": m_full.group(2),
            "start": None,
            "end": None,
            "type": "full",
        }

    m_seg = re.match(r"poses_(train|val|test)(\d{4})_(\d+)_(\d+)\.npy$", name)
    if m_seg:
        return {
            "original": name,
            "split": m_seg.group(1),
            "pose_id": m_seg.group(2),
            "start": int(m_seg.group(3)),
            "end": int(m_seg.group(4)),
            "type": "segment",
        }

    return {
        "original": name,
        "split": None,
        "pose_id": None,
        "start": None,
        "end": None,
        "type": "unrecognized",
    }

def parse_annotation_line(line):
    line = line.strip()
    if not line:
        return None

    parts = line.split()
    name_token = parts[0]
    label = parts[1] if len(parts) > 1 else None

    # raw format like: 0003_01_0001.mp4 9
    m = re.match(r"(\d{4})_(\d{2})_(\d{4})\.mp4$", name_token)
    if m:
        return {
            "original": line,
            "name_token": name_token,
            "subject_id": m.group(1),
            "camera_id": m.group(2),
            "clip_id": m.group(3),
            "label": label,
            "type": "raw_video",
        }

    return {
        "original": line,
        "name_token": name_token,
        "subject_id": None,
        "camera_id": None,
        "clip_id": None,
        "label": label,
        "type": "unrecognized",
    }

# -----------------------------
# Read pose files from all splits
# -----------------------------
all_pose_entries = []
per_split_entries = {}

for split_name, folder in folders.items():
    if not os.path.isdir(folder):
        print(f"[WARNING] Folder not found: {folder}")
        continue

    entries = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".npy"):
            entries.append(parse_pose_name(fname))

    per_split_entries[split_name] = entries
    all_pose_entries.extend(entries)

# -----------------------------
# Read annotations
# -----------------------------
annotation_entries = []
with open(annotation_txt, "r", encoding="utf-8") as f:
    for line in f:
        parsed = parse_annotation_line(line)
        if parsed:
            annotation_entries.append(parsed)

ann_ok = [e for e in annotation_entries if e["type"] == "raw_video"]

# -----------------------------
# Print split summaries
# -----------------------------
print("=" * 70)
print("POSE SPLIT SUMMARY")
print("=" * 70)

for split_name, entries in per_split_entries.items():
    full_count = sum(e["type"] == "full" for e in entries)
    seg_count = sum(e["type"] == "segment" for e in entries)
    bad_count = sum(e["type"] == "unrecognized" for e in entries)

    pose_ids = sorted({e["pose_id"] for e in entries if e["pose_id"] is not None})

    print(f"\n[{split_name.upper()}]")
    print(f"Total files: {len(entries)}")
    print(f"Full files: {full_count}")
    print(f"Segment files: {seg_count}")
    print(f"Unrecognized: {bad_count}")
    print(f"Unique pose_ids: {len(pose_ids)}")
    print(f"First 20 pose_ids: {pose_ids[:20]}")
    print(f"Last 20 pose_ids: {pose_ids[-20:]}")

# -----------------------------
# Compare pose IDs across splits
# -----------------------------
print("\n" + "=" * 70)
print("POSE ID OVERLAP ACROSS SPLITS")
print("=" * 70)

split_pose_ids = {
    split_name: {e["pose_id"] for e in entries if e["pose_id"] is not None}
    for split_name, entries in per_split_entries.items()
}

split_names = list(split_pose_ids.keys())
for i in range(len(split_names)):
    for j in range(i + 1, len(split_names)):
        a, b = split_names[i], split_names[j]
        overlap = split_pose_ids[a] & split_pose_ids[b]
        print(f"{a} vs {b}: overlap={len(overlap)}")
        print(f"First 30 overlaps: {sorted(list(overlap))[:30]}")

# -----------------------------
# Annotation summary
# -----------------------------
print("\n" + "=" * 70)
print("ANNOTATION SUMMARY")
print("=" * 70)
print(f"Total annotation lines: {len(annotation_entries)}")
print(f"Recognized raw-video lines: {len(ann_ok)}")

subject_ids = sorted({e['subject_id'] for e in ann_ok if e['subject_id'] is not None})
camera_ids = sorted({e['camera_id'] for e in ann_ok if e['camera_id'] is not None})
clip_ids = sorted({e['clip_id'] for e in ann_ok if e['clip_id'] is not None})

print(f"Unique subject_ids: {len(subject_ids)}")
print(f"First 20 subject_ids: {subject_ids[:20]}")
print(f"Unique camera_ids: {camera_ids}")
print(f"Unique clip_ids: {len(clip_ids)}")
print(f"First 20 clip_ids: {clip_ids[:20]}")

# -----------------------------
# Compare each split with annotation 4-digit subject IDs
# -----------------------------
print("\n" + "=" * 70)
print("SPLIT vs ANNOTATION 4-DIGIT OVERLAP")
print("=" * 70)

ann_subject_set = set(subject_ids)

for split_name, pose_id_set in split_pose_ids.items():
    overlap = pose_id_set & ann_subject_set
    only_pose = pose_id_set - ann_subject_set
    only_ann = ann_subject_set - pose_id_set

    print(f"\n[{split_name.upper()}]")
    print(f"Overlap count: {len(overlap)}")
    print(f"First 30 overlaps: {sorted(list(overlap))[:30]}")
    print(f"Pose-only IDs first 30: {sorted(list(only_pose))[:30]}")
    print(f"Annotation-only IDs first 30: {sorted(list(only_ann))[:30]}")

# -----------------------------
# Check if train/val restart indexing from 0000
# -----------------------------
print("\n" + "=" * 70)
print("INDEX RESTART CHECK")
print("=" * 70)

for split_name, entries in per_split_entries.items():
    full_ids = sorted({e["pose_id"] for e in entries if e["type"] == "full" and e["pose_id"] is not None})
    print(f"{split_name}: first 10 full pose_ids = {full_ids[:10]}")
    print(f"{split_name}: last 10 full pose_ids = {full_ids[-10:]}")

# -----------------------------
# Show examples of segments per split
# -----------------------------
print("\n" + "=" * 70)
print("SEGMENT EXAMPLES")
print("=" * 70)

for split_name, entries in per_split_entries.items():
    segs_by_id = defaultdict(list)
    for e in entries:
        if e["type"] == "segment":
            segs_by_id[e["pose_id"]].append((e["start"], e["end"], e["original"]))

    some_ids = sorted(list(segs_by_id.keys()))[:10]
    print(f"\n[{split_name.upper()}]")
    for pid in some_ids:
        segs = sorted(segs_by_id[pid], key=lambda x: (x[0], x[1]))
        print(f"pose_id={pid}, num_segments={len(segs)}, first_segments={segs[:5]}")