#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_anno_file(path: Path) -> List[Tuple[str, int]]:
    """
    Annotation format: "<filename> <label>"
    Example: "0003_01_0001.mp4 9"
    """
    items: List[Tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid format at {path}:{ln}: {line}")
            file_name = parts[0]
            label = int(parts[1])
            items.append((file_name, label))
    return items


def resolve_npy_path(npy_dir: Path, file_name: str, patterns: List[str]) -> Optional[Path]:
    """
    Try to match npy files using configured patterns.
    {file} keeps extension (e.g., 0003_01_0001.mp4)
    {stem} drops extension (e.g., 0003_01_0001)
    """
    stem = Path(file_name).stem
    for pat in patterns:
        candidate = npy_dir / pat.format(file=file_name, stem=stem)
        if candidate.exists():
            return candidate
    return None


def build_single_split(
    split: str,
    anno_path: Path,
    npy_dir: Path,
    out_dir: Path,
    patterns: List[str],
    write_missing_report: bool,
) -> None:
    print("\n" + "=" * 72)
    print(f"[{split.upper()}] anno: {anno_path}")
    print(f"[{split.upper()}] npy : {npy_dir}")
    print(f"[{split.upper()}] out : {out_dir}")
    print("=" * 72)

    if not anno_path.exists():
        print(f"[WARN] Missing annotation file. Skipping split='{split}': {anno_path}")
        return

    items = read_anno_file(anno_path)

    data_list: List[str] = []
    label_list: List[int] = []
    missing: List[str] = []

    for file_name, label in items:
        p = resolve_npy_path(npy_dir, file_name, patterns)
        if p is None:
            missing.append(file_name)
            continue
        data_list.append(str(p))
        label_list.append(label)

    out_dir.mkdir(parents=True, exist_ok=True)

    data_arr = np.array(data_list)
    label_arr = np.array(label_list, dtype=np.int64)

    data_pkl = out_dir / f"{split}_data_list.pkl"
    label_pkl = out_dir / f"{split}_label_list.pkl"

    with data_pkl.open("wb") as f:
        pickle.dump(data_arr, f)
    with label_pkl.open("wb") as f:
        pickle.dump(label_arr, f)

    print(f"[INFO] split={split} #samples={len(data_arr)} labels_shape={label_arr.shape}")
    print(f"[OK] saved: {data_pkl}")
    print(f"[OK] saved: {label_pkl}")

    if write_missing_report:
        report = out_dir / f"{split}_missing_npy.txt"
        with report.open("w", encoding="utf-8") as f:
            for fn in missing:
                f.write(fn + "\n")
        if missing:
            print(f"[WARN] missing npy: {len(missing)} (see {report})")
        else:
            print(f"[OK] no missing npy files")


def main():
    ap = argparse.ArgumentParser(description="Build MMA-style PKL lists from dataset configs.")
    ap.add_argument("--paths", default="config/paths.yaml", help="Path to paths.yaml")
    ap.add_argument("--datasets", default="config/datasets.yaml", help="Path to datasets.yaml")
    ap.add_argument("--dataset", required=True, help="Dataset key in datasets.yaml (e.g., MA52_SINGLE)")
    ap.add_argument("--splits", default="train,val,test", help="Comma-separated splits to build")
    args = ap.parse_args()

    paths_cfg = load_yaml(Path(args.paths))
    ds_cfg_all = load_yaml(Path(args.datasets))

    base_dir = Path(paths_cfg["base_dir"])
    npy_root = Path(paths_cfg["npy_root"])
    data_root = Path(paths_cfg["data_root"])

    datasets: Dict[str, dict] = ds_cfg_all["datasets"]
    if args.dataset not in datasets:
        raise KeyError(f"Unknown dataset '{args.dataset}'. Available: {list(datasets.keys())}")

    ds = datasets[args.dataset]
    label_type = ds.get("label_type", "single")
    if label_type != "single":
        raise NotImplementedError("This script currently implements single-label build only.")

    annotations = ds["annotations"]
    npy_subdirs = ds["npy_subdirs"]
    out_dir = data_root / ds["out_dir"]
    patterns = ds["npy_patterns"]
    write_missing_report = bool(ds.get("write_missing_report", True))

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        if split not in annotations or split not in npy_subdirs:
            print(f"[WARN] split '{split}' not configured. Skipping.")
            continue

        anno_path = data_root / annotations[split]
        npy_dir = npy_root / npy_subdirs[split]

        build_single_split(
            split=split,
            anno_path=anno_path,
            npy_dir=npy_dir,
            out_dir=out_dir,
            patterns=patterns,
            write_missing_report=write_missing_report,
        )

    print("\n[DONE] Completed.\n")


if __name__ == "__main__":
    main()
