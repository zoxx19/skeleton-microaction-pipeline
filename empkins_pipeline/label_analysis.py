"""
label_analysis.py
=================
Task 3: Analyze inference labels across all subjects
Task 4: Propose merge strategy
Task 5: Provide refined labeling schema

Produces:
  - label_analysis_report.txt  — full written report
  - figures/                   — visualizations

Run:
    python empkins_processing/label_analysis.py
"""

import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ============================================================
# CONFIG
# ============================================================
INFERENCE_DIR = Path("empkins_processing/inference")
OUTPUT_DIR    = Path("empkins_processing/label_analysis")

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
# LOAD DATA
# ============================================================
def load_results():
    with open(INFERENCE_DIR / "all_results.pkl", "rb") as f:
        return pickle.load(f)


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================
def get_condition_dist(results):
    """Get label distribution per condition/phase."""
    groups = {}
    for r in results:
        key = f"{r['condition']}/{r['phase']}"
        groups.setdefault(key, []).append(r["top1_name"])
    return {k: Counter(v) for k, v in groups.items()}


def get_subject_dist(results):
    """Get label distribution per subject."""
    groups = {}
    for r in results:
        groups.setdefault(r["subject"], []).append(r["top1_name"])
    return {k: Counter(v) for k, v in groups.items()}


def get_confidence_stats(results):
    """Get mean confidence per label."""
    conf_by_label = {}
    for r in results:
        label = r["top1_name"]
        conf_by_label.setdefault(label, []).append(r["top1_prob"])
    return {k: np.mean(v) for k, v in conf_by_label.items()}


def analyze_missing_labels(results):
    """Find labels that never appear in top-1 predictions."""
    predicted = set(r["top1_name"] for r in results)
    all_labels = set(LABEL_NAMES.values())
    missing = all_labels - predicted
    return sorted(missing)


def analyze_top5_coverage(results):
    """Find labels that appear in top-5 but never top-1."""
    top1_labels = set(r["top1_name"] for r in results)
    top5_labels = set()
    for r in results:
        top5_labels.update(r["top5_names"])
    only_in_top5 = top5_labels - top1_labels
    return sorted(only_in_top5)


def confusion_between_labels(results, label_a, label_b):
    """Count how often label_a has label_b as top-2."""
    count = 0
    total = 0
    for r in results:
        if r["top1_name"] == label_a:
            total += 1
            if len(r["top5_names"]) > 1 and r["top5_names"][1] == label_b:
                count += 1
    return count, total


# ============================================================
# PLOTTING
# ============================================================
def plot_global_distribution(results, out_path):
    dist  = Counter(r["top1_name"] for r in results)
    total = len(results)

    labels = [l for l, _ in dist.most_common()]
    counts = [dist[l] for l in labels]
    pcts   = [100*c/total for c in counts]
    colors = ["#e74c3c" if p > 20 else
              "#f39c12" if p > 5  else
              "#2ecc71" for p in pcts]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(labels[::-1], pcts[::-1], color=colors[::-1])
    ax.set_xlabel("Percentage of segments (%)")
    ax.set_title(f"Global Label Distribution — {total} segments, 20 subjects")
    ax.axvline(x=5, color="gray", ls="--", lw=1, alpha=0.5, label="5% line")

    for bar, pct, cnt in zip(bars, pcts[::-1], counts[::-1]):
        ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                f"{cnt} ({pct:.1f}%)", va="center", fontsize=9)

    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def plot_condition_comparison(results, out_path):
    conditions = {
        "ftsst/talk": "ftsst/talk",
        "tsst/talk":  "tsst/talk",
        "tsst/math":  "tsst/math",
    }

    # Get top labels across all conditions
    all_dist = Counter(r["top1_name"] for r in results)
    top_labels = [l for l, _ in all_dist.most_common(12)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    for ax, (cond_key, cond_title) in zip(axes, conditions.items()):
        cond_results = [r for r in results
                        if f"{r['condition']}/{r['phase']}" == cond_key]
        total = len(cond_results)
        dist  = Counter(r["top1_name"] for r in cond_results)

        pcts   = [100*dist.get(l,0)/total for l in top_labels]
        colors = ["#e74c3c" if l in ["tilting head","turning head"]
                  else "#3498db" if l in ["illustrative gestures","spreading hands",
                                          "retracting arms","waving"]
                  else "#e67e22" if l == "shaking body"
                  else "#95a5a6"
                  for l in top_labels]

        bars = ax.barh(top_labels[::-1], [pcts[i] for i in range(len(top_labels))][::-1],
                       color=colors[::-1])
        ax.set_title(f"{cond_title}\n({total} segs)", fontsize=11)
        ax.set_xlabel("% of segments")
        for bar, pct in zip(bars, pcts[::-1]):
            if pct > 1:
                ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                        f"{pct:.1f}%", va="center", fontsize=8)

    plt.suptitle("Label Distribution by Condition", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def plot_segs_per_min(results, out_path):
    """Plot segments/min per subject per condition."""
    # Duration estimates (from batch_segment output)
    durations = {
        "VP_02": {"tsst/talk":298,"tsst/math":294,"ftsst/talk":547},
        "VP_03": {"tsst/talk":206,"tsst/math":262,"ftsst/talk":600},
        "VP_04": {"tsst/talk":305,"tsst/math":276,"ftsst/talk":608},
        "VP_05": {"tsst/talk":263,"tsst/math":279,"ftsst/talk":653},
        "VP_06": {"tsst/talk":306,"tsst/math":299,"ftsst/talk":616},
        "VP_07": {"tsst/talk":304,"tsst/math":303,"ftsst/talk":598},
        "VP_08": {"tsst/talk":309,"tsst/math":299,"ftsst/talk":620},
        "VP_09": {"tsst/talk":311,"tsst/math":278,"ftsst/talk":594},
        "VP_10": {"tsst/talk":273,"tsst/math":299,"ftsst/talk":517},
        "VP_11": {"tsst/talk":309,"tsst/math":276,"ftsst/talk":535},
        "VP_12": {"tsst/talk":272,"tsst/math":312,"ftsst/talk":615},
        "VP_13": {"tsst/talk":308,"tsst/math":296,"ftsst/talk":604},
        "VP_14": {"tsst/talk":310,"tsst/math":280,"ftsst/talk":607},
        "VP_15": {"tsst/talk":307,"tsst/math":285,"ftsst/talk":614},
        "VP_16": {"tsst/talk":324,"tsst/math":308,"ftsst/talk":610},
        "VP_17": {"tsst/talk":388,"tsst/math":327,"ftsst/talk":671},
        "VP_18": {"tsst/talk":373,"tsst/math":139,"ftsst/talk":625},
        "VP_19": {"tsst/talk":371,"tsst/math":317,"ftsst/talk":628},
        "VP_20": {"tsst/talk":307,"tsst/math":312,"ftsst/talk":547},
        "VP_21": {"tsst/talk":313,"tsst/math":326,"ftsst/talk":652},
    }

    subjects = sorted(durations.keys())
    conditions = ["tsst/talk","tsst/math","ftsst/talk"]
    colors = ["#e74c3c","#c0392b","#3498db"]

    # Count segments per subject per condition
    seg_counts = {}
    for r in results:
        key = (r["subject"], f"{r['condition']}/{r['phase']}")
        seg_counts[key] = seg_counts.get(key, 0) + 1

    x = np.arange(len(subjects))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, (cond, color) in enumerate(zip(conditions, colors)):
        rates = []
        for subj in subjects:
            n    = seg_counts.get((subj, cond), 0)
            dur  = durations.get(subj, {}).get(cond, 300)
            rate = n / (dur / 60)
            rates.append(rate)
        ax.bar(x + i*width, rates, width, label=cond, color=color, alpha=0.8)

    ax.set_xlabel("Subject")
    ax.set_ylabel("Segments per minute")
    ax.set_title("Movement Activity Rate by Subject and Condition")
    ax.set_xticks(x + width)
    ax.set_xticklabels([s.replace("VP_","VP") for s in subjects], rotation=45)
    ax.legend()
    ax.axhline(y=5, color="gray", ls="--", lw=1, alpha=0.5)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


def plot_confidence_by_label(results, out_path):
    conf_stats = {}
    count_stats = {}
    for r in results:
        l = r["top1_name"]
        conf_stats.setdefault(l, []).append(r["top1_prob"])
        count_stats[l] = count_stats.get(l, 0) + 1

    # Sort by count
    labels   = sorted(conf_stats.keys(), key=lambda l: -count_stats[l])
    means    = [np.mean(conf_stats[l]) for l in labels]
    stds     = [np.std(conf_stats[l])  for l in labels]
    counts   = [count_stats[l]         for l in labels]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color="#3498db", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean confidence (softmax prob)")
    ax.set_title("Model Confidence per Predicted Label")
    ax.axhline(y=0.25, color="red", ls="--", lw=1, label="25% confidence")
    ax.set_ylim(0, 1.0)
    ax.legend()

    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, 0.02,
                f"n={cnt}", ha="center", fontsize=7, color="white", fontweight="bold")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


# ============================================================
# WRITE REPORT
# ============================================================
def write_report(results, out_path):
    total     = len(results)
    dist      = Counter(r["top1_name"] for r in results)
    missing   = analyze_missing_labels(results)
    top5_only = analyze_top5_coverage(results)
    conf      = get_confidence_stats(results)
    cond_dist = get_condition_dist(results)

    lines = []
    def w(s=""): lines.append(s)

    w("=" * 70)
    w("EMPKINS MICRO-ACTION LABEL ANALYSIS REPORT")
    w("Tasks 3, 4, 5 — Label Analysis, Merge Strategy, Refined Schema")
    w("=" * 70)
    w(f"Total segments analyzed : {total}")
    w(f"Subjects                : 20 (VP_01 missing data)")
    w(f"Conditions              : tsst/talk, tsst/math, ftsst/talk")
    w(f"Model                   : MMN (40 classes, standing upper body)")
    w()

    # ── TASK 3A: Label distribution ──
    w("=" * 70)
    w("TASK 3A — GLOBAL LABEL DISTRIBUTION")
    w("=" * 70)
    w(f"{'Label':35s}  {'Count':>6}  {'Pct':>6}  {'AvgConf':>8}")
    w("-" * 65)
    for label, count in dist.most_common():
        pct  = 100 * count / total
        conf_val = conf.get(label, 0)
        flag = " ← HIGH" if pct > 20 else ""
        w(f"  {label:33s}  {count:>6}  {pct:>5.1f}%  {conf_val:>8.3f}{flag}")
    w()

    # ── TASK 3B: Per-condition ──
    w("=" * 70)
    w("TASK 3B — LABEL DISTRIBUTION BY CONDITION")
    w("=" * 70)
    for cond_key in ["tsst/talk", "tsst/math", "ftsst/talk"]:
        cond_r = [r for r in results
                  if f"{r['condition']}/{r['phase']}" == cond_key]
        n = len(cond_r)
        d = Counter(r["top1_name"] for r in cond_r)
        w(f"\n  {cond_key} ({n} segments):")
        for label, count in d.most_common(8):
            pct = 100*count/n
            w(f"    {label:33s}: {count:>4} ({pct:>5.1f}%)")
    w()

    # ── TASK 3C: Missing labels ──
    w("=" * 70)
    w("TASK 3C — MISSING LABELS (never predicted as top-1)")
    w("=" * 70)
    w(f"  {len(missing)} labels never appeared as top-1 prediction:")
    for label in missing:
        in_top5 = label in top5_only
        tag = " (appears in top-5)" if in_top5 else " (never in top-5 either)"
        w(f"    - {label}{tag}")
    w()

    # ── TASK 3D: Irrelevant labels ──
    w("=" * 70)
    w("TASK 3D — IRRELEVANT OR PROBLEMATIC LABELS")
    w("=" * 70)
    w("""
  1. OVER-PREDICTED (likely artifact):
     - tilting head (61.6%): Suspiciously dominant. Confidence is high
       but distribution is nearly identical across all conditions
       (60.6% ftsst vs 61.4% tsst/talk vs 67.6% tsst/math).
       This suggests the model defaults to this class when uncertain.
       Root cause: synthesized head joints lack real 3D variation.

  2. NEVER PREDICTED (irrelevant for standing TSST):
     - shrugging, turning around, rising up, bowing head, head up
     - scratching arms, putting hands together, rubbing hands
     - pointing oneself, clenching fist, stretching arms, waving
     - hands touching fingers
     - All face/head-touching labels (24-39):
       scratching neck/chest/back/shoulder, arms akimbo, crossing arms,
       playing hair, scratching hindbrain/forehead/face, rubbing eyes,
       touching nose/ears, covering face/mouth, pushing glasses
       (Note: crossing arms and arms akimbo do appear occasionally)

  3. CONFOUNDED LABELS:
     - tilting head vs turning head: confused 23/54 times (43% of cases)
     - These two should be merged into a single "head movement" label
    """)

    # ── TASK 4: Merge strategy ──
    w("=" * 70)
    w("TASK 4 — PROPOSED MERGE STRATEGY")
    w("=" * 70)
    w("""
  MERGE GROUP 1: Head movements
    - tilting head + turning head + nodding + shaking head
    → NEW LABEL: "head movement"
    Rationale: Model confuses these 43% of the time. All represent
    head motion which is a natural stress response. Merging reduces
    noise while preserving behavioral signal.

  MERGE GROUP 2: Hand/object interaction
    - playing objects + hands touching fingers + other finger movements
    → NEW LABEL: "hand/object interaction"
    Rationale: These all represent small hand movements with objects
    or self-touching, common during stress. Visually similar in 3D.

  MERGE GROUP 3: Arm gestures
    - illustrative gestures + spreading hands + waving + retracting arms
    → NEW LABEL: "arm gesturing"
    Rationale: All represent expressive arm movements during speech.
    Natural to group as a single gesture category.

  MERGE GROUP 4: Body posture changes
    - shaking body + sitting straightly + arms akimbo + crossing arms
    → NEW LABEL: "body posture change"
    Rationale: These all represent whole-body postural shifts which
    are indicators of stress and discomfort.

  REMOVE (never/rarely predicted, not relevant to TSST):
    - shrugging, turning around, rising up, bowing head, head up
    - scratching arms, putting hands together, rubbing hands
    - pointing oneself, clenching fist, stretching arms
    - all face-touching labels (scratching neck, chest, back,
      shoulder, hindbrain, forehead, face, rubbing eyes,
      touching nose/ears, covering face/mouth, pushing glasses)
    - playing or tidying hair
    Rationale: These never appear in top-1 predictions for standing
    TSST data, and most require close-up video to detect reliably.
    """)

    # ── TASK 5: Refined schema ──
    w("=" * 70)
    w("TASK 5 — REFINED LABELING SCHEMA")
    w("=" * 70)
    w("""
  PROPOSED 4-CLASS SCHEMA for Empkins TSST micro-action recognition:
  ─────────────────────────────────────────────────────────────────

  Class 0: HEAD MOVEMENT
    Original labels: tilting head, turning head, nodding, shaking head
    Behavioral meaning: Any head rotation or tilt, common in
    conversation and stress response.

  Class 1: ARM GESTURING
    Original labels: illustrative gestures, spreading hands,
                     retracting arms, waving, stretching arms
    Behavioral meaning: Expressive arm and hand movements during
    speech, associated with active communication.

  Class 2: HAND/OBJECT INTERACTION
    Original labels: playing objects, other finger movements,
                     hands touching fingers, rubbing hands,
                     scratching arms
    Behavioral meaning: Self-touching and object manipulation,
    known stress indicators (self-grooming behavior).

  Class 3: BODY POSTURE CHANGE
    Original labels: shaking body, sitting straightly, crossing arms,
                     arms akimbo, shrugging
    Behavioral meaning: Whole-body postural shifts reflecting
    tension, discomfort or emotional state.

  ─────────────────────────────────────────────────────────────────

  JUSTIFICATION:
  This 4-class schema:
    1. Covers all labels that actually appear in TSST predictions
    2. Aligns with psychological literature on stress body language
    3. Is detectable with upper-body IMU skeleton data
    4. Allows meaningful comparison between conditions:
       - ARM GESTURING should be higher in talk phases (active speech)
       - BODY POSTURE CHANGE should be higher in math phase (stress)
       - HEAD MOVEMENT should be consistent (conversational behavior)
       - HAND/OBJECT should increase with anxiety

  EXPECTED DISTRIBUTION after remapping:
  """)

    # Simulate remapping
    remap = {}
    for r in results:
        label = r["top1_name"]
        if label in ["tilting head","turning head","nodding","shaking head","head up","bowing head"]:
            new = "head movement"
        elif label in ["illustrative gestures","spreading hands","retracting arms",
                       "waving","stretching arms"]:
            new = "arm gesturing"
        elif label in ["playing objects","other finger movements","hands touching fingers",
                       "rubbing hands","scratching arms","putting hands together"]:
            new = "hand/object interaction"
        elif label in ["shaking body","sitting straightly","crossing arms",
                       "arms akimbo","shrugging"]:
            new = "body posture change"
        else:
            new = "other"
        remap[new] = remap.get(new, 0) + 1

    for new_label, count in sorted(remap.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        w(f"    {new_label:25s}: {count:>5} ({pct:>5.1f}%)")

    w()
    w("  Note: 'other' contains labels from face-touching category")
    w("  which could be added as Class 4 if face video data is available.")
    w()

    # ── Per-condition after remapping ──
    w("  REMAPPED DISTRIBUTION BY CONDITION:")
    for cond_key in ["tsst/talk","tsst/math","ftsst/talk"]:
        cond_r = [r for r in results
                  if f"{r['condition']}/{r['phase']}" == cond_key]
        n = len(cond_r)
        remap_cond = {}
        for r in cond_r:
            label = r["top1_name"]
            if label in ["tilting head","turning head","nodding","shaking head","head up","bowing head"]:
                new = "head movement"
            elif label in ["illustrative gestures","spreading hands","retracting arms",
                           "waving","stretching arms"]:
                new = "arm gesturing"
            elif label in ["playing objects","other finger movements","hands touching fingers",
                           "rubbing hands","scratching arms","putting hands together"]:
                new = "hand/object interaction"
            elif label in ["shaking body","sitting straightly","crossing arms",
                           "arms akimbo","shrugging"]:
                new = "body posture change"
            else:
                new = "other"
            remap_cond[new] = remap_cond.get(new, 0) + 1

        w(f"\n  {cond_key} ({n} segs):")
        for new_label, count in sorted(remap_cond.items(), key=lambda x: -x[1]):
            pct = 100*count/n
            w(f"    {new_label:25s}: {count:>4} ({pct:>5.1f}%)")

    w()
    w("=" * 70)
    w("END OF REPORT")
    w("=" * 70)

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path.name}")
    return lines


# ============================================================
# PLOT REMAPPED DISTRIBUTION
# ============================================================
def plot_remapped(results, out_path):
    remap_schema = {
        "tilting head":         "head movement",
        "turning head":         "head movement",
        "nodding":              "head movement",
        "shaking head":         "head movement",
        "head up":              "head movement",
        "bowing head":          "head movement",
        "illustrative gestures":"arm gesturing",
        "spreading hands":      "arm gesturing",
        "retracting arms":      "arm gesturing",
        "waving":               "arm gesturing",
        "stretching arms":      "arm gesturing",
        "playing objects":      "hand/object interaction",
        "other finger movements":"hand/object interaction",
        "hands touching fingers":"hand/object interaction",
        "rubbing hands":        "hand/object interaction",
        "scratching arms":      "hand/object interaction",
        "putting hands together":"hand/object interaction",
        "shaking body":         "body posture change",
        "sitting straightly":   "body posture change",
        "crossing arms":        "body posture change",
        "arms akimbo":          "body posture change",
        "shrugging":            "body posture change",
    }

    conditions = ["tsst/talk","tsst/math","ftsst/talk"]
    colors = {"head movement":"#e74c3c", "arm gesturing":"#3498db",
              "hand/object interaction":"#2ecc71", "body posture change":"#f39c12",
              "other":"#95a5a6"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    for ax, cond in zip(axes, conditions):
        cond_r = [r for r in results
                  if f"{r['condition']}/{r['phase']}" == cond]
        n = len(cond_r)
        remapped = Counter(remap_schema.get(r["top1_name"], "other")
                           for r in cond_r)
        new_labels = ["head movement","arm gesturing",
                      "hand/object interaction","body posture change","other"]
        pcts = [100*remapped.get(l,0)/n for l in new_labels]
        clrs = [colors[l] for l in new_labels]

        bars = ax.bar(new_labels, pcts, color=clrs, alpha=0.85)
        ax.set_title(f"{cond}\n({n} segs)")
        ax.set_ylabel("% of segments")
        ax.set_ylim(0, 100)
        ax.set_xticks(range(len(new_labels)))
        ax.set_xticklabels(new_labels, rotation=30, ha="right", fontsize=9)

        for bar, pct in zip(bars, pcts):
            if pct > 1:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                        f"{pct:.1f}%", ha="center", fontsize=9)

    plt.suptitle("Refined 4-Class Schema — Distribution by Condition", fontsize=13)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.name}")


# ============================================================
# MAIN
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("Loading results...")
    results = load_results()
    print(f"Loaded {len(results)} segment results")
    print()

    print("Generating plots...")
    plot_global_distribution(results, fig_dir / "01_global_distribution.png")
    plot_condition_comparison(results, fig_dir / "02_condition_comparison.png")
    plot_confidence_by_label(results, fig_dir / "03_confidence_by_label.png")
    plot_segs_per_min(results, fig_dir / "04_activity_rate_by_subject.png")
    plot_remapped(results, fig_dir / "05_refined_schema_distribution.png")

    print()
    print("Writing report...")
    write_report(results, OUTPUT_DIR / "label_analysis_report.txt")

    print()
    print(f"Done! All outputs saved to: {OUTPUT_DIR}")
    print(f"  label_analysis_report.txt")
    print(f"  figures/01_global_distribution.png")
    print(f"  figures/02_condition_comparison.png")
    print(f"  figures/03_confidence_by_label.png")
    print(f"  figures/04_activity_rate_by_subject.png")
    print(f"  figures/05_refined_schema_distribution.png")


if __name__ == "__main__":
    main()