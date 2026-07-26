# micro-action-pipeline

Code, configs, and small docs for a skeleton-based micro-action pipeline built on
the **MA-52** micro-action dataset and applied to **EmpkinS / TSST** motion-capture
recordings. This repository is a **clean, code-only extraction** from a larger working
tree — it is safe to publish on GitHub.

> **⚠️ No data is included in this repository.** See
> [Data policy](#data-policy) below. Every array/checkpoint/log/video artifact is
> `.gitignore`d. What you get here is the *pipeline* (scripts + configs), not the
> inputs or outputs.

---

## Pipeline overview

The project is four stages. Stages 1–3 are the skeleton pipeline
(MediaPipe pose → neutralization → MMN model → EmpkinS inference); stage 4 is an
independent **video**-recognition baseline kept alongside for comparison.

```
(1) neutralization  ─►  (2) mmn_training  ─►  (3) empkins_pipeline
                                                (inference on real mocap)

(4) manet_baseline   ── independent MA-52 video-recognition track
```

### 1. Neutralization — `neutralization/`
Takes raw MediaPipe-33 body-pose `.npy` (extracted from MA-52 videos) and **neutralizes
the lower body** (zeros / normalizes lower-body joints) so the model focuses on
upper-body micro-actions.

- `neutralize_ma52.py` — the neutralization step: reads `npy_out_ma/{train,val,test}`,
  writes neutralized poses to `npy_out_ma_neutralized/{split}`, saves a before/after
  visualization sample.
- `neutralization_vis/` — 3 small before/after comparison PNGs (train/val/test sample).
- `testscripts/` — inspection utilities: preview/debug a packed MA-52 sample, summarize
  per-joint visibility (to identify lower-body joint indices), and count label
  distribution over the neutralized `.npy` folders.

### 2. MMN training — `mmn_training/`
Trains the **MMN** (Motion-guided Modulation Network, ACM MM 2025) skeleton classifier on
the neutralized poses, in a single-label "standing" 40-class variant (lower-body labels
removed from the 52-class MA-52 space).

- `model/MMN.py` — the network definition.
- `feeders/` — skeleton data feeders (`feeder_single` = the standing single-label feeder;
  `feeder_ma52` = upstream MA-52 feeder; `tools.py` = augmentation/normalization helpers).
- `scripts/` — `main_single.py` (the trainer actually launched), `build_ma52_single.py`
  (pack neutralized `.npy` + annotation lists into dataset PKLs), `label_conversion.py`
  (52→40 upper-only label remapping).
- `config/` — `paths.yaml` (project paths — **reconciled to this account**, see below),
  `datasets.yaml` (dataset-build spec), `paths_mediapipe.yaml` (pose-extraction config);
  `config/train/` holds `standing_single.yaml` (the live 40-class training config) and
  `base_single_blazepose.yaml` (base template).
- `train_gpu.sh` — SLURM launcher (partition `rtx3080`, conda env `mmn_clean`).

### 3. EmpkinS inference — `empkins_pipeline/`
Applies the trained MMN model to **real EmpkinS / TSST mocap** recordings. Pipeline is a
segment → normalize → infer chain (each script's docstring calls these `step1…step4`):

- `inspect_and_animate.py` — load one subject, extract the TSST phase, animate (sanity).
- `segment.py` — segment movement clips per phase via a global p85 activity threshold.
- `normalize.py` — map 21 EmpkinS joints → 33 BlazePose, neutralize lower body, align to
  MA-52 space (hip-center, flip-Y, scale torso→0.42).
- `normalize_updated.py` — **v2 normalization**, additionally aligns body orientation.
  ⚠️ **Only run for subject `VP_04` so far**, not the full batch (see notes).
- `inference.py` — run the trained MMN standing model on one subject's normalized segments.
- `verify_alignment.py`, `visualize_inference.py`, `label_analysis.py` — QA overlays,
  timeline/GIF visualizations, and cross-subject label analysis.
- `batch_final/` — the all-21-subjects batch versions of segment / normalize(v2) / infer.

### 4. MANet baseline — `manet_baseline/`
An **independent** MA-52 *video*-recognition baseline (MANet on `mmaction2`), kept for
comparison — **not** part of the skeleton→MMN→EmpkinS chain. Only the run-specific files
are included here (the `mmaction2` framework itself and its data/checkpoints are not):

- `manet.py` — MANet R50 config for MA-52 video recognition.
- `train_manet_cpu.sbatch`, `eval_manet_cpu.sbatch` — SLURM train/eval jobs.
- `eval.py` — convert model output → Codabench submission + score.

---

## Data policy

**(a) No data is included in this repo.** It contains code, configs, and a few small
docs/PNGs only. All of the following are excluded (and `.gitignore`d): pose/skeleton
`.npy`/`.npz`, model checkpoints (`.pt`/`.pth`/`.pkl`), predictions, logs, videos, and any
`work_dir/ data/ segments/ normalized*/ npy_out*/ inference/ logs/` directories. To run
anything here you must supply the corresponding data out-of-band.

**(b) The raw EmpkinS mocap data is not on this cluster.** The stage-3 scripts reference
raw input at `data/empkins_pilot/data_per_subject/` (`.bvh.gz`,
`_global_pose.csv.gz`, …). That raw data was **confirmed absent from every storage tier
reachable from this account** — `/home/hpc` ($HOME), `/home/woody` ($WORK, incl. the full
~760 GB of Track-1/Track-2 artifacts), and `/home/vault` (empty). Only *derived* products
(segmented `.npz`, normalized `.npy`, per-subject inference `.pkl`) ever existed on the
cluster. **To re-run stage 3 from scratch you must source the raw mocap from the original
EmpkinS system or the original laptop copy** — it cannot be recovered from this cluster.

**(c) v2 normalization is partial.** `empkins_pipeline/normalize_updated.py` (the v2
orientation-aligning normalization) and its batch equivalent have **only been run for
subject `VP_04`**, not for the full 21-subject cohort. The v1 `normalize.py` output was
the full run.

---

## Path configuration note

`mmn_training/config/paths.yaml` was reconciled during extraction: several entries in the
original pointed at a **collaborator's account and layout**
(`/home/hpc/iwso/REDACTED_COLLABORATOR/MMN_MMA33/…`). Those were rewritten to this account's live
`REDACTED_ACCOUNT` layout under `.../Micro-Action/Micro-action skeleton/MMN`, consistent with the
`base_dir` / `npy_root` / `data_root` entries. Absolute cluster paths (conda roots, SLURM
partitions, `~/repos/...` locations) remain HPC-specific — adjust for your environment.

---

## Status / caveats

- Absolute paths and SLURM directives throughout are specific to the NHR@FAU HPC account
  this was extracted from; treat them as examples.
- This is an extraction of an active research tree, not a packaged release — there is no
  `requirements.txt` pin or entry-point CLI here.
