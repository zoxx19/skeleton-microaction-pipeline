#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import yaml


# =========================================================
# 1) Label mapping
# =========================================================
def build_label_mapping(
    num_classes: int,
    merge_groups: List[List[int]],
    omit_labels: List[int],
) -> Tuple[Dict[int, Optional[int]], int]:
    """
    Build mapping old(0..num_classes-1) -> new class index (or None if omitted).

    - merge_groups: list of lists; each group becomes one new class.
    - omit_labels: labels to be dropped.
    - remaining labels are assigned consecutive new IDs.

    Returns:
        index_map: dict old -> new (or None)
        new_num_classes: final number of classes
    """
    omit_set = set(omit_labels)

    # validate omit_labels
    for idx in omit_set:
        if not (0 <= idx < num_classes):
            raise IndexError(f"Invalid label in omit_labels: {idx}")

    cleaned_groups: List[List[int]] = []
    used = set()

    # validate merge groups
    for group in merge_groups:
        g = sorted(set(group))
        if any(idx in used for idx in g):
            raise ValueError(f"Overlapping label in merge_groups: {g}")
        if any(idx in omit_set for idx in g):
            raise ValueError(f"Label in merge_groups also appears in omit_labels: {g}")
        cleaned_groups.append(g)
        used.update(g)

    index_map: Dict[int, Optional[int]] = {}
    new_idx = 0

    # step 1: merged groups
    for group in cleaned_groups:
        for old_idx in group:
            index_map[old_idx] = new_idx
        new_idx += 1

    # step 2: remaining labels
    for old_idx in range(num_classes):
        if old_idx in omit_set:
            index_map[old_idx] = None
        elif old_idx not in index_map:
            index_map[old_idx] = new_idx
            new_idx += 1

    return index_map, new_idx


# =========================================================
# 2) Detect label mode
# =========================================================
def detect_label_mode(labels_any, original_classes: int):
    """
    Detect whether labels are:
      - 1D integer array -> single-label
      - 2D multi-hot     -> multi-label

    Returns:
      mode: "single" or "multi"
      labels: numpy array (int for single, int for multi-hot)
    """
    labels = np.array(labels_any)

    if labels.ndim == 1:
        return "single", labels.astype(int)

    if labels.ndim == 2:
        if labels.shape[1] != original_classes:
            raise ValueError(
                f"Label dimension mismatch: labels.shape[1]={labels.shape[1]} "
                f"but original_classes={original_classes}"
            )
        # multi-hot, keep as int
        return "multi", labels.astype(int)

    raise ValueError("Unknown label format: must be 1-D (single) or 2-D (multi-hot)")


# =========================================================
# 3) Convert one split
# =========================================================
def process_dataset(
    input_data_file: Path,
    input_label_file: Path,
    output_data_file: Path,
    output_label_file: Path,
    index_map: Dict[int, Optional[int]],
    new_num_classes: int,
    original_num_classes: int,
    omit_labels: List[int],
) -> str:
    """
    Convert a dataset (data_list.pkl + label_list.pkl) for either single or multi labels.

    - Single-label: drop omitted-label samples, then map old label -> new label.
    - Multi-label : zero-out omitted columns, drop all-zero rows, then merge columns.
    """
    if not input_data_file.exists() or not input_label_file.exists():
        raise FileNotFoundError(f"Missing input pkl: {input_data_file} / {input_label_file}")

    with input_data_file.open("rb") as f:
        data = pickle.load(f)
    with input_label_file.open("rb") as f:
        labels_raw = pickle.load(f)

    mode, labels = detect_label_mode(labels_raw, original_num_classes)

    # ---------- single-label ----------
    if mode == "single":
        labels = labels.astype(int)

        keep_mask = np.array([lbl not in set(omit_labels) for lbl in labels], dtype=bool)
        new_data = np.array(data)[keep_mask]
        new_labels = labels[keep_mask]

        final_labels = np.array([index_map[int(lbl)] for lbl in new_labels], dtype=int)

        output_data_file.parent.mkdir(parents=True, exist_ok=True)
        with output_data_file.open("wb") as f:
            pickle.dump(new_data, f)
        with output_label_file.open("wb") as f:
            pickle.dump(final_labels, f)

        return mode

    raise RuntimeError("Unreachable")


def write_mapping_txt(out_path: Path, index_map: Dict[int, Optional[int]], new_num_classes: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("old_id -> new_id\n")
        for k in sorted(index_map.keys()):
            f.write(f"{k:2d} -> {index_map[k]}\n")
        f.write(f"\nnew_num_classes = {new_num_classes}\n")


# =========================================================
# 4) Main (config-based)
# =========================================================
def main():
    ap = argparse.ArgumentParser(description="Convert label space (merge + omit) for PKL datasets.")
    ap.add_argument("--paths", default="config/paths.yaml", help="Path to paths.yaml")
    ap.add_argument("--datasets", default="config/datasets.yaml", help="Path to datasets.yaml")
    ap.add_argument("--dataset", default="LABEL_CONVERT_52_TO_47_SINGLE", help="Dataset key in datasets.yaml")
    ap.add_argument("--splits", default="train,val", help="Comma-separated splits to convert")
    args = ap.parse_args()

    # load configs
    paths_cfg = yaml.safe_load(Path(args.paths).read_text(encoding="utf-8"))
    ds_cfg_all = yaml.safe_load(Path(args.datasets).read_text(encoding="utf-8"))
    ds = ds_cfg_all["datasets"][args.dataset]

    data_root = Path(paths_cfg["data_root"])

    input_dir = data_root / ds["source_dir"]
    output_dir = data_root / ds["out_dir"]

    original_num_classes = int(ds["original_num_classes"])
    merge_groups = ds.get("merge_groups", [])
    omit_labels = ds.get("omit_labels", [])
    write_map = bool(ds.get("write_mapping_txt", True))

    index_map, new_num_classes = build_label_mapping(
        original_num_classes, merge_groups, omit_labels
    )

    print("=== Class Mapping (old -> new) ===")
    for k in sorted(index_map.keys()):
        print(f"{k:2d} -> {index_map[k]}")
    print(f"New total classes = {new_num_classes}\n")

    if write_map:
        write_mapping_txt(output_dir / "label_mapping.txt", index_map, new_num_classes)
        print(f"[OK] Saved mapping: {output_dir / 'label_mapping.txt'}")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        in_data = input_dir / f"{split}_data_list.pkl"
        in_label = input_dir / f"{split}_label_list.pkl"
        out_data = output_dir / f"{split}_data_list.pkl"
        out_label = output_dir / f"{split}_label_list.pkl"

        mode = process_dataset(
            input_data_file=in_data,
            input_label_file=in_label,
            output_data_file=out_data,
            output_label_file=out_label,
            index_map=index_map,
            new_num_classes=new_num_classes,
            original_num_classes=original_num_classes,
            omit_labels=omit_labels,
        )

        print(f"[{split}] Converted ({mode}-label format) -> {output_dir}")

    print("\n[DONE] Label conversion completed.\n")


if __name__ == "__main__":
    main()
