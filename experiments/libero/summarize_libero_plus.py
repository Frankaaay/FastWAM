#!/usr/bin/env python
"""Aggregate LIBERO-plus per-case results into factor success rates."""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict


CAT2COL = {
    "Camera Viewpoints": "Camera",
    "Robot Initial States": "Robot",
    "Language Instructions": "Lang.",
    "Light Conditions": "Light",
    "Background Textures": "BG",
    "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
}
COL_ORDER = ["Camera", "Robot", "Lang.", "Light", "BG", "Noise", "Layout"]


def _pct(success_total):
    success, total = success_total
    return 100.0 * success / total if total else float("nan")


def _avg(values):
    return sum(values) / len(values) if values else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.output_dir, "*", "gpu*_task*_results.json"))
    if not files:
        print(f"[warn] no results.json under {args.output_dir}")
        return

    by_cat = defaultdict(lambda: [0, 0])
    by_suite = defaultdict(lambda: [0, 0])
    by_diff = defaultdict(lambda: [0, 0])
    by_cat_time = defaultdict(list)
    overall = [0, 0]
    overall_time = []
    n_files = 0

    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        total = int(data.get("total_episodes", 0))
        if total == 0:
            continue
        success = int(data.get("successes", 0))
        n_files += 1
        col = CAT2COL.get(data.get("category"), data.get("category") or "?")
        by_cat[col][0] += success
        by_cat[col][1] += total
        by_suite[data.get("task_suite", "?")][0] += success
        by_suite[data.get("task_suite", "?")][1] += total
        difficulty = data.get("difficulty_level")
        if difficulty is not None:
            by_diff[difficulty][0] += success
            by_diff[difficulty][1] += total
        overall[0] += success
        overall[1] += total
        duration = data.get("duration")
        if duration is not None:
            by_cat_time[col].append(float(duration))
            overall_time.append(float(duration))

    n_failed = len(glob.glob(os.path.join(args.output_dir, "*", "gpu*_task*.FAILED")))

    print(f"\n# LIBERO-plus aggregate ({n_files} cases, {n_failed} FAILED markers)")
    print(f"  output_dir = {args.output_dir}\n")
    print("== By perturbation factor ==")
    print("  " + "".join(f"{col:>8}" for col in COL_ORDER) + f"{'Total':>9}")
    row = "  "
    for col in COL_ORDER:
        row += f"{_pct(by_cat[col]):>8.1f}" if col in by_cat else f"{'-':>8}"
    row += f"{_pct(overall):>9.1f}"
    print(row)
    print(
        "  (succ/total: "
        + ", ".join(f"{col} {by_cat[col][0]}/{by_cat[col][1]}" for col in COL_ORDER if col in by_cat)
        + f", Total {overall[0]}/{overall[1]})"
    )

    print("\n== By suite ==")
    for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        if suite in by_suite:
            print(f"  {suite:16} {_pct(by_suite[suite]):5.1f}  ({by_suite[suite][0]}/{by_suite[suite][1]})")

    if by_diff:
        print("\n== By difficulty L1-L5 ==")
        for difficulty in sorted(by_diff):
            print(f"  L{difficulty}: {_pct(by_diff[difficulty]):5.1f}  ({by_diff[difficulty][0]}/{by_diff[difficulty][1]})")

    csv_path = os.path.join(args.output_dir, "summary.csv")
    cols = [col for col in COL_ORDER if col in by_cat]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([os.path.basename(os.path.normpath(args.output_dir))])
        writer.writerow([""] + cols + ["Total"])
        writer.writerow(["Success Rate (%)"] + [f"{_pct(by_cat[col]):.2f}" for col in cols] + [f"{_pct(overall):.2f}"])
        writer.writerow(["Successes"] + [str(by_cat[col][0]) for col in cols] + [str(overall[0])])
        writer.writerow(["Total Cases"] + [str(by_cat[col][1]) for col in cols] + [str(overall[1])])
        writer.writerow(["Average Time (s)"] + [f"{_avg(by_cat_time[col]):.2f}" for col in cols] + [f"{_avg(overall_time):.2f}"])
        writer.writerow(
            ["Max Time (s)"]
            + [f"{max(by_cat_time[col]):.2f}" if by_cat_time[col] else "nan" for col in cols]
            + [f"{max(overall_time):.2f}" if overall_time else "nan"]
        )
        writer.writerow([])
        writer.writerow(["by suite", "Success Rate (%)", "succ", "total"])
        for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
            if suite in by_suite:
                writer.writerow([suite, f"{_pct(by_suite[suite]):.2f}", by_suite[suite][0], by_suite[suite][1]])
        writer.writerow([])
        writer.writerow(["by difficulty", "Success Rate (%)", "succ", "total"])
        for difficulty in sorted(by_diff):
            writer.writerow([f"L{difficulty}", f"{_pct(by_diff[difficulty]):.2f}", by_diff[difficulty][0], by_diff[difficulty][1]])

    print(f"\n  summary.csv written to: {csv_path}")


if __name__ == "__main__":
    main()
