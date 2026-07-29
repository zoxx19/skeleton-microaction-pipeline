# skeleton-microaction-pipeline

A skeleton-based pipeline for detecting **micro-actions** — small, often subconscious
body movements (a head tilt, a hand-to-face touch, a shoulder shift) — first on the
**MA-52** benchmark dataset, then applied to real motion-capture recordings from an
**EmpkinS / TSST** (Trier Social Stress Test) stress study.

Developed as part of student research project at the MaD Lab, Friedrich-Alexander-Universität
Erlangen-Nürnberg (FAU).

> **This is a clean, code-only extraction from a larger working research tree.**
> No datasets, trained model weights, or generated results are included — see
> [Data policy](#data-policy). What's here is the pipeline itself: the code and
> configuration needed to reproduce it, given the right data.

---

## Table of contents

- [Background](#background)
- [Pipeline overview](#pipeline-overview)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Running each stage](#running-each-stage)
- [Data policy](#data-policy)
- [Known limitations / open items](#known-limitations--open-items)
- [Acknowledgments](#acknowledgments)

---

## Background

**Micro-actions** are brief, low-intensity body movements — as opposed to full gestures
or actions like "walking" or "waving." Think: touching your face, rubbing your neck,
folding your arms, shifting your weight. They're of interest because they often correlate
with internal states like stress, discomfort, or engagement, even when a person isn't
consciously aware of making them.

This project builds on two things:

1. **[MA-52](https://github.com/VUT-HFUT/Micro-Action)** — a public benchmark dataset of
   52 labeled micro-action classes, annotated over video clips. It's used here as
   *skeleton* data (body-pose coordinates extracted from the videos via
   [MediaPipe BlazePose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)),
   not as raw video.
2. **[MMN](https://github.com/momiji-bit/MMN)** (Motion-guided Modulation Network,
   ACM MM 2025) — a skeleton-based action-recognition model, used here as the
   classifier that's trained on MA-52 and then applied to new data.

The project's specific angle: many MA-52 micro-action classes involve leg/foot movement
(e.g. "leg shaking"), which isn't relevant when the goal is analyzing **upper-body**
behavior in a seated, stationary context (like someone undergoing a stress test). So
before training, every skeleton in the dataset is modified to **neutralize the lower
body** — replacing whatever the legs were doing with a fixed, neutral standing pose built
from that person's own proportions — so the model can't "cheat" by keying off leg motion
and instead has to learn to recognize the upper-body micro-actions the project actually
cares about.

The trained model is then applied to real mocap data from an **EmpkinS / TSST** pilot
study — participants performing the Trier Social Stress Test (a standardized psychological
stress-induction protocol) while wearing a full-body motion-capture suit — to see what
micro-actions the model detects during a stressful social/cognitive task.

---

## Pipeline overview

A three-stage pipeline: MediaPipe pose → neutralization → MMN model → EmpkinS inference.

```
┌─────────────────┐     ┌────────────────┐     ┌──────────────────────┐
│ 1. neutralization│ ──► │ 2. mmn_training│ ──► │ 3. empkins_pipeline  │
│  (MA-52 skeletons│     │ (train MMN on  │     │ (apply trained model │
│   → legs zeroed) │     │  neutralized   │     │  to real TSST mocap) │
└─────────────────┘     │  MA-52 poses)  │     └──────────────────────┘
                         └────────────────┘
```

### 1. Neutralization — `neutralization/`

**Input:** raw MediaPipe-33 body-pose coordinates (`x, y, z, visibility` per joint, per
frame), already extracted from the MA-52 videos.
**Output:** the same poses, but with every lower-body joint (hips down: hips, knees,
ankles, heels, toes — MediaPipe indices 23–32) replaced by a neutral standing pose.

How the neutral pose is built, per sample:
1. Compute the person's own pelvis position, and their "up," "right," and "forward" body
   directions from their shoulders and hips in the first frame.
2. Measure that person's own thigh and shin length from their original pose.
3. Place a standing hip → knee → ankle chain along the "down" direction, using their own
   limb lengths (so everyone's neutral pose matches their own body proportions, not a
   generic template).
4. Add feet (heel/toe) with a small forward offset, auto-correcting the orientation if
   the initial guess points the feet backwards.
5. Apply that one computed template to every frame of the sequence — only the lower-body
   joints are touched; everything from the hips up is left exactly as recorded.

Files:
- `neutralize_ma52.py` — the neutralization script. Reads pose `.npy` files from
  `npy_out_ma/{train,val,test}/`, writes neutralized versions to
  `npy_out_ma_neutralized/{split}/`, and saves one before/after visualization per split.
- `neutralization_vis/` — 3 example before/after comparison images (train/val/test).
- `testscripts/` — small inspection utilities used while building this stage:
  - `preview_official_ma52_sample.py` / `debug_official_ma52_sample.py` — view a single
    packed MA-52 sample as an image/GIF, with or without joint-index overlays.
  - `summarize_official_ma52_joints.py` — per-joint visibility/position statistics,
    used to confirm which joint indices are "lower body."
  - `inspect_labels.py` — label distribution over the neutralized dataset.

### 2. MMN training — `mmn_training/`

Trains the MMN skeleton-classification model on the neutralized MA-52 poses.

Two important simplifications from the original 52-class MA-52 task:
- **Single-label instead of multi-label** — each clip is treated as one dominant
  micro-action, not a set of co-occurring ones.
- **40 classes instead of 52** — the 12 classes that are inherently about lower-body
  movement (leg shaking, foot tapping, etc.) are dropped, since those joints are now
  neutralized and carry no signal.

Files:
- `model/MMN.py` — the MMN network architecture itself.
- `feeders/` — PyTorch `Dataset` classes that load and preprocess the skeleton data for
  training:
  - `feeder_single.py` — the feeder actually used for this project (single-label,
    40-class, BlazePose-33 joints, 96-frame temporal window, with augmentation:
    rotation, scaling, translation, time-jitter).
  - `feeder_ma52.py` — the original upstream MA-52 feeder (multi-label, 52-class),
    kept for reference / compatibility.
  - `tools.py` — shared augmentation and normalization helper functions.
- `scripts/`
  - `main_single.py` — the training/evaluation entry point actually run for this
    project.
  - `build_ma52_single.py` — packs the neutralized `.npy` files plus annotation lists
    into the `.pkl` dataset format the feeder expects.
  - `label_conversion.py` — performs the 52→40 label remapping (drops the lower-body
    classes, writes out a `label_mapping.txt`).
- `config/`
  - `paths.yaml` — where the dataset, models, and outputs live.
  - `datasets.yaml` — declarative spec for how each dataset variant (52-class vs.
    40-class "upper-only") gets built.
  - `paths_mediapipe.yaml` — configuration for the (separate, upstream) pose-extraction
    step that produced the raw `.npy` files in the first place.
  - `train/standing_single.yaml` — the actual training config used (model
    hyperparameters, optimizer, learning-rate schedule, number of epochs, paths).
  - `train/base_single_blazepose.yaml` — a base template config this was derived from.
- `train_gpu.sh` — a SLURM batch script for launching training on a GPU cluster.
  Cluster-specific (partition name, conda environment, absolute paths) — treat it as an
  example to adapt, not something portable as-is.

### 3. EmpkinS / TSST inference — `empkins_pipeline/`

Applies the trained MMN model to real EmpkinS/TSST motion-capture recordings, to detect
what micro-actions participants exhibited during the stress test.

The chain (each script corresponds to one stage — filenames don't always say "step1" etc.
but the docstrings do):

1. **`inspect_and_animate.py`** — loads one subject's recording, extracts the relevant
   task phase (using session timing metadata), and animates the skeleton as a sanity
   check that the data loaded correctly.
2. **`segment.py`** — splits a continuous recording into discrete movement "clips." Uses
   a per-subject, cross-phase 85th-percentile activity threshold (an earlier, simpler
   per-joint threshold approach was tried and rejected — see
   [Known limitations](#known-limitations--open-items) for why fixed/percentile
   thresholds needed iteration).
3. **`normalize.py`** — maps the mocap suit's 21 tracked joints onto MediaPipe's 33-joint
   BlazePose layout, applies the same lower-body neutralization as stage 1, and aligns
   the result into the same coordinate space as MA-52 (hip-centered, axis-flipped, scaled
   to match MA-52's torso-length convention).
4. **`normalize_updated.py`** — a second, improved version of step 3 that additionally
   corrects for body *orientation* (not just position/scale) using a similarity
   transform, run across the full 21-subject cohort.
5. **`inference.py`** — runs the trained MMN model on the normalized segments and records
   the predicted micro-action label(s) per segment.
6. **`verify_alignment.py`** — a QA tool that overlays a normalized EmpkinS skeleton
   against an MA-52 reference skeleton, to visually confirm the alignment in step 3/4
   actually lines the two coordinate systems up.
7. **`visualize_inference.py`** — generates timeline plots, skeleton-grid images, and
   GIFs of the inference results, for qualitative review.
8. **`label_analysis.py`** — aggregates predictions across subjects, flags likely
   artifacts (e.g. one label being predicted suspiciously often), and proposes reduced
   label groupings for downstream analysis.
9. **`batch_final/`** — batch versions of segmentation, normalization (v2), and
   inference that run the pipeline across all 21 subjects at once, rather than one
   subject at a time.

---

## Repository structure

```
skeleton-microaction-pipeline/
├── neutralization/
│   ├── neutralize_ma52.py
│   ├── neutralization_vis/          # 3 example before/after images
│   └── testscripts/                 # inspection/debug utilities
├── mmn_training/
│   ├── model/MMN.py
│   ├── feeders/                     # feeder_single, feeder_ma52, tools
│   ├── scripts/                     # main_single, build_ma52_single, label_conversion
│   ├── config/                      # paths, dataset spec, training configs
│   ├── torchlight/ torchpack/       # vendored helper libraries (see Setup)
│   ├── train_gpu.sh
│   └── requirements.txt
├── empkins_pipeline/
│   ├── segment.py, normalize.py, normalize_updated.py, inference.py, ...
│   └── batch_final/                 # all-21-subjects batch versions
└── README.md
```

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
git clone https://github.com/zoxx19/skeleton-microaction-pipeline.git
cd skeleton-microaction-pipeline/mmn_training

uv venv
uv pip install -r requirements.txt
```

### About `torchlight` / `torchpack`

The MMN model depends on two small helper libraries, `torchlight` and `torchpack`.
**Do not `pip install` them** — the packages published on PyPI under those names are
unrelated projects and will cause import errors (e.g. `DictAction` not found). This repo
vendors the correct versions directly under `mmn_training/torchlight/` and
`mmn_training/torchpack/` (originally from the [MMN repository](https://github.com/momiji-bit/MMN)),
so no separate install step is needed for them — they just need to be importable, which
works automatically once `mmn_training/` is on your `PYTHONPATH`:

```bash
cd mmn_training
PYTHONPATH=$PWD python -c "import model.MMN; import feeders.feeder_single"
# should print nothing and exit 0 if everything is set up correctly
```

### Note on `torch`

`uv pip install -r requirements.txt` will pull the default CUDA build of PyTorch, which
drags in the full NVIDIA/Triton stack (several GB on disk). If you only need CPU
inference or don't have a CUDA GPU available, install a CPU-only build instead:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Running each stage

All commands below assume you're inside the relevant stage's folder with the `uv`
environment active (`source .venv/bin/activate`, or prefix commands with `uv run`), and
that you've supplied your own copy of the required data (see
[Data policy](#data-policy) — none is included in this repo).

```bash
# 1. Neutralize MA-52 skeletons
cd neutralization
python neutralize_ma52.py

# 2. Build the training dataset, then train
cd ../mmn_training
python scripts/label_conversion.py --config config/datasets.yaml
python scripts/build_ma52_single.py --config config/datasets.yaml
PYTHONPATH=$PWD python scripts/main_single.py --config config/train/standing_single.yaml

# 3. Run the EmpkinS inference pipeline (single subject example)
cd ../empkins_pipeline
python segment.py
python normalize.py
python inference.py

# ...or the full-cohort batch versions:
python batch_final/batch_segment.py
python batch_final/batch_normalize.py
python batch_final/batch_inference.py
```

Exact CLI arguments (data roots, subject IDs, etc.) depend on how you've laid out your
own copy of the data — check each script's argument parser / top-of-file constants before
running. SLURM scripts (`train_gpu.sh`, `*.sbatch`) are examples from a specific HPC
cluster and will need adapting to your own environment (partition names, conda/uv setup,
absolute paths).

---

## Data policy

**No data is included in this repository** — no datasets, no model checkpoints, no
generated predictions, logs, or videos. Everything in that category is excluded via
`.gitignore` (`*.npy`, `*.npz`, `*.pt`, `*.pth`, `*.pkl`, `work_dir/`, `data/`,
`segments/`, `normalized*/`, `logs/`, and similar). What's published here is the code and
configuration only.

To actually run this pipeline, you need to independently obtain:

- **MA-52** — the dataset this project is built on. See the
  [official MA-52 repository](https://github.com/VUT-HFUT/Micro-Action) for access and
  licensing terms. This project uses MediaPipe-extracted pose coordinates from the MA-52
  videos, not the raw video itself, as its primary input for stages 1–2.
- **EmpkinS/TSST mocap data** — used in stage 3. This is participant motion-capture data
  from a stress-study protocol and is not public; it requires appropriate ethics
  clearance/data-sharing agreement to obtain, and is not something this repository can
  distribute. If you have institutional access to the source dataset, the expected format
  is documented in `empkins_pipeline/segment.py`.
- **Trained model weights** — not included; train your own using the steps above, or
  obtain them separately if you already have access to a prior training run.

---

## Known limitations / open items

- **This is a code extraction from an active research tree**, not a versioned release —
  there's no CLI entry point wrapping the whole pipeline, and some scripts still assume
  paths/environment details specific to the cluster they were developed on. Treat SLURM
  scripts and default config paths as examples to adapt, not as portable out of the box.
- The 40-class "upper-only" label set and the neutralization approach are project-specific
  design choices, not part of the original MA-52 benchmark — see `label_mapping.txt`
  (generated by `label_conversion.py`) for the exact class list and mapping used here.

---

## Acknowledgments

- **[MA-52](https://github.com/VUT-HFUT/Micro-Action)** dataset — VUT-HFUT.
- **[MMN](https://github.com/momiji-bit/MMN)** (Motion-guided Modulation Network,
  ACM Multimedia 2025) — the base model architecture used in `mmn_training/`.
- **[MediaPipe](https://developers.google.com/mediapipe)** — pose extraction (BlazePose).
- Developed at the MaD Lab, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
