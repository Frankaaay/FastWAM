#!/usr/bin/env python
# ----------------------------------------------------------------------------
# summarize_libero_plus.py — 把 LIBERO-plus 每个 case 的 results.json 聚合成
# 按扰动 factor 的成功率表,对齐 robustness paper Table 4 的列。
#
#   python experiments/libero/summarize_libero_plus.py --output_dir <OUT>
#
# 每个 results.json 里已带 "category"(扰动 factor)和 "difficulty_level",
# 所以直接读这俩聚合即可,不依赖外部映射。
# ----------------------------------------------------------------------------
import argparse
import glob
import json
import os
from collections import defaultdict

# task_classification.json 的 category 名 -> paper 列名
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.output_dir, "*", "gpu*_task*_results.json"))
    if not files:
        print(f"[warn] no results.json under {args.output_dir}")
        return

    by_cat = defaultdict(lambda: [0, 0])      # col -> [succ, total]
    by_suite = defaultdict(lambda: [0, 0])
    by_diff = defaultdict(lambda: [0, 0])
    overall = [0, 0]
    n_files = 0

    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        s = int(d.get("successes", 0))
        t = int(d.get("total_episodes", 0))
        if t == 0:
            continue
        n_files += 1
        col = CAT2COL.get(d.get("category"), d.get("category") or "?")
        by_cat[col][0] += s; by_cat[col][1] += t
        by_suite[d.get("task_suite", "?")][0] += s; by_suite[d.get("task_suite", "?")][1] += t
        dl = d.get("difficulty_level")
        if dl is not None:
            by_diff[dl][0] += s; by_diff[dl][1] += t
        overall[0] += s; overall[1] += t

    def pct(sd):
        return 100.0 * sd[0] / sd[1] if sd[1] else float("nan")

    # FAILED 标记统计(诊断用)
    n_failed = len(glob.glob(os.path.join(args.output_dir, "*", "gpu*_task*.FAILED")))

    print(f"\n# LIBERO-plus 聚合  ({n_files} cases 出分, {n_failed} 个 .FAILED)")
    print(f"  output_dir = {args.output_dir}\n")

    print("== 按扰动 factor(对齐 paper Table 4)==")
    hdr = "  " + "".join(f"{c:>8}" for c in COL_ORDER) + f"{'Total':>9}"
    print(hdr)
    row = "  "
    for c in COL_ORDER:
        row += f"{pct(by_cat[c]):>8.1f}" if c in by_cat else f"{'-':>8}"
    row += f"{pct(overall):>9.1f}"
    print(row)
    print("  (succ/total: " + ", ".join(
        f"{c} {by_cat[c][0]}/{by_cat[c][1]}" for c in COL_ORDER if c in by_cat)
        + f", Total {overall[0]}/{overall[1]})")

    print("\n== 按 suite ==")
    for s in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
        if s in by_suite:
            print(f"  {s:16} {pct(by_suite[s]):5.1f}  ({by_suite[s][0]}/{by_suite[s][1]})")

    if by_diff:
        print("\n== 按难度 L1-L5 ==")
        for dl in sorted(by_diff):
            print(f"  L{dl}: {pct(by_diff[dl]):5.1f}  ({by_diff[dl][0]}/{by_diff[dl][1]})")

    print("\n  注:Original 列 = 标准 LIBERO(无扰动)成绩,见 evaluate_results/libero 下的全量 eval。")


if __name__ == "__main__":
    main()
