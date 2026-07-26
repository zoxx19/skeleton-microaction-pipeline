import os
import random
import pickle
from typing import Dict, Tuple, Sequence, Optional

import numpy as np
from torch.utils.data import Dataset


##############################################################################
# BlazePose 33-joint bone definition (for bone / motion representations)
##############################################################################
BLAZEPOSE_BONES = [
    (11, 13), (13, 15),            # left arm
    (12, 14), (14, 16),            # right arm
    (15, 17), (17, 19), (19, 21),  # left hand
    (16, 18), (18, 20), (20, 22),  # right hand
    (23, 25), (25, 27),            # left leg
    (24, 26), (26, 28),            # right leg
    (27, 29), (29, 31),            # left foot
    (28, 30), (30, 32),            # right foot
    (11, 12),                      # shoulders
    (23, 24),                      # hips
    (11, 23), (12, 24),            # torso
    (0, 1), (1, 2), (2, 3),        # left eye chain
    (0, 4), (4, 5), (5, 6),        # right eye chain
    (0, 9), (0, 10),               # nose → mouth
    (7, 1), (8, 4)                 # ears → eyes
]


##############################################################################
# Helpers
##############################################################################
def _nan_safe(x: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0."""
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _center_by_joint(x: np.ndarray, joint_idx: int = 23) -> np.ndarray:
    """
    Center the skeleton sequence by a reference joint at the first frame.
    Args:
        x: (T, V, C)
    Returns:
        centered x: (T, V, C)
    """
    center = x[0, joint_idx].copy()
    return x - center


def _normalize_to_unit_range(x: np.ndarray) -> np.ndarray:
    """
    Per-axis normalization to [-1, 1].
    Args:
        x: (T, V, C)
    """
    flat = x.reshape(-1, x.shape[-1])
    minv = flat.min(axis=0)
    maxv = flat.max(axis=0)
    scale = np.maximum(maxv - minv, 1e-6)
    flat = (flat - minv) / scale
    flat = flat * 2.0 - 1.0
    return flat.reshape(x.shape)


def _uniform_temporal_sample(x: np.ndarray, T_out: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniformly sample sequence to fixed length.
    Returns:
        x_sampled: (T_out, V, C)
        idx: (T_out,) indices from original timeline
    """
    T = x.shape[0]
    if T <= 1:
        idx = np.zeros(T_out, dtype=int)
    else:
        idx = np.linspace(0, T - 1, T_out).astype(int)
    return x[idx], idx


def _apply_rotation_z(x: np.ndarray, max_deg: float = 15.0) -> np.ndarray:
    """
    Apply random rotation around Z axis (3D only).
    Args:
        x: (T, V, 3)
    """
    angle = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    rot = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle),  np.cos(angle), 0.0],
        [0.0,            0.0,           1.0],
    ], dtype=np.float32)
    return x @ rot.T


def _to_bone(x: np.ndarray, bones=BLAZEPOSE_BONES) -> np.ndarray:
    """Convert joint coords -> bone vectors."""
    bone = np.zeros_like(x)
    for a, b in bones:
        bone[:, a] = x[:, a] - x[:, b]
    return bone


def _to_motion(x: np.ndarray) -> np.ndarray:
    """Convert coords -> motion (frame difference)."""
    motion = np.zeros_like(x)
    motion[:-1] = x[1:] - x[:-1]
    return motion


##############################################################################
# Unified Feeder: BlazePose(33) / Single-label / 2D or 3D
##############################################################################
class Feeder(Dataset):
    """
    MMN-compatible feeder for BlazePose 33-joint skeleton inputs (single-label).

    Expects in data_root:
      - {split}_data_list.pkl  : list/np.array of .npy file paths
      - {split}_label_list.pkl : list/np.array of integer labels

    Produces:
      data   : (C, T, V, 1) float32
      index_t: (T,) normalized time indices in [-1, 1]
      label  : int
      idx    : int (real sample index)
    """

    def __init__(
        self,
        split: str = "train",
        data_root: str = "./data/MMA33",
        data_type: str = "j",              # "j", "b", "m", or combos like "jm"
        coord_dim: int = 3,                # 2 or 3
        time_steps: int = 64,
        repeat: int = 1,
        aug_prob: float = 0.2,
        rot_max_deg: float = 15.0,         # used only for coord_dim==3
        normalize: bool = True,
        center_joint: int = 23,
        debug: bool = False,
        split_map: Optional[Dict[str, Tuple[str, str]]] = None,
        train_splits: Sequence[str] = ("train", "finetune_train"),
    ):
        super().__init__()

        if coord_dim not in (2, 3):
            raise ValueError("coord_dim must be 2 or 3")

        self.split = split
        self.data_root = data_root
        self.data_type = data_type
        self.coord_dim = coord_dim
        self.time_steps = time_steps
        self.repeat = repeat
        self.aug_prob = aug_prob
        self.rot_max_deg = rot_max_deg
        self.normalize = normalize
        self.center_joint = center_joint
        self.train_splits = set(train_splits)

        # default split mapping (supports finetune splits)
        default_split_map = {
            "train": ("train_label_list.pkl", "train_data_list.pkl"),
            "val":   ("val_label_list.pkl",   "val_data_list.pkl"),
            "test":  ("test_label_list.pkl",  "test_data_list.pkl"),
            "finetune": ("finetune_label_list.pkl", "finetune_data_list.pkl"),
            "finetune_train": ("finetune_train_label_list.pkl", "finetune_train_data_list.pkl"),
            "finetune_val":   ("finetune_val_label_list.pkl",   "finetune_val_data_list.pkl"),
        }
        self.split_map = split_map or default_split_map

        if split not in self.split_map:
            raise ValueError(f"Unknown split: {split}. Expected one of {list(self.split_map.keys())}.")

        label_name, data_name = self.split_map[split]
        self.label_file = os.path.join(data_root, label_name)
        self.data_file = os.path.join(data_root, data_name)

        if not os.path.exists(self.label_file):
            raise FileNotFoundError(f"Label pkl not found: {self.label_file}")
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data pkl not found: {self.data_file}")

        with open(self.label_file, "rb") as f:
            labels = pickle.load(f)
        with open(self.data_file, "rb") as f:
            data_list = pickle.load(f)

        self.labels = np.array(labels, dtype=int)
        self.data_list = list(data_list)

        if len(self.data_list) != len(self.labels):
            raise ValueError(
                f"data_list ({len(self.data_list)}) and labels ({len(self.labels)}) length mismatch"
            )

        if debug:
            self.data_list = self.data_list[:64]
            self.labels = self.labels[:64]

        self.N = len(self.data_list)
        print(f"[Feeder] split={self.split}, N={self.N}, coord_dim={self.coord_dim}, time_steps={self.time_steps}")
        print(f"[Feeder] data_root={self.data_root}")

    def __len__(self):
        return self.N * self.repeat

    def __getitem__(self, index: int):
        real_idx = index % self.N
        npy_path = self.data_list[real_idx]
        label = int(self.labels[real_idx])

        # 1) load (T, 33, C_in)
        x = np.load(npy_path)

        # 2) channel selection:
        #    - if (x,y,z,vis) -> select first 2 or 3 dims
        #    - else expect already coord_dim
        x = _nan_safe(x)
        if x.shape[-1] >= 4:
            x = x[..., :self.coord_dim]
        elif x.shape[-1] == self.coord_dim:
            pass
        else:
            raise ValueError(f"Unexpected input channels: {x.shape[-1]} (expected {self.coord_dim} or 4+)")

        # 3) centering + normalize
        x = _center_by_joint(x, self.center_joint)
        if self.normalize:
            x = _normalize_to_unit_range(x)

        # 4) temporal sampling
        x, idx = _uniform_temporal_sample(x, self.time_steps)  # (T_out, 33, C)
        # For time index, we want normalization based on original timeline indices:
        # idx contains original indices, so we normalize by max(idx)+1, not by T_out.
        denom = float(max(1, (idx.max() + 1)))
        index_t = 2.0 * idx.astype(np.float32) / denom - 1.0

        # 5) augmentation (train splits only)
        if self.split in self.train_splits:
            # rotation only in 3D
            if self.coord_dim == 3 and random.random() < self.aug_prob:
                x = _apply_rotation_z(x, max_deg=self.rot_max_deg)

            # temporal reverse
            if random.random() < self.aug_prob:
                x = x[::-1].copy()

            # axis masking (2D/3D)
            if random.random() < self.aug_prob:
                axis = np.random.randint(0, self.coord_dim)
                x[:, :, axis] = 0.0

        # 6) representation: bone / motion
        if "b" in self.data_type:
            x = _to_bone(x, bones=BLAZEPOSE_BONES)

        if "m" in self.data_type:
            x = _to_motion(x)

        # 7) final: (C, T, V, 1)
        x = x.transpose(2, 0, 1).astype(np.float32)  # (C, T, V)
        x = x[:, :, :, None]                         # (C, T, V, 1)

        return x, index_t.astype(np.float32), label, real_idx

    def top_k(self, score, top_k: int, f: bool = False):
        """
        Compute top-k accuracy for single-label classification.
        Args:
            score: (N, num_classes)
            top_k: int
        Returns:
            top-k accuracy in [0,1]
        """
        score = np.array(score)
        rank = score.argsort(axis=1)
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.labels)]
        return float(sum(hit_top_k)) / len(hit_top_k)
