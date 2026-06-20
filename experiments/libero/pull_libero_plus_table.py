#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动把 evaluate_results/libero_plus 下所有 summary.csv 拉成一张对比表。

每个 eval run 目录里有一个 summary.csv(由 summarize_libero_plus.py 生成),
本脚本扫描全部、抽出每个 run 的各 factor 成功率,并额外算一个「去 Noise」
的 Total —— 因为含 Noise(10030 case)和不含 Noise(8429 case)的 run 的
原始 Total 不可直接比,去 Noise 后才是苹果对苹果。

用法:
  python experiments/libero/pull_libero_plus_table.py
  python experiments/libero/pull_libero_plus_table.py --root evaluate_results/libero_plus
  python experiments/libero/pull_libero_plus_table.py --csv  # 额外输出 csv

可选:在 LABEL 里给目录名映射一个好看的实验名。
"""
import argparse, csv, glob, os

# 目录名 -> 展示名(没命中的用目录名)。按需补充。
LABEL = {
    "MEMOFF_20260618_143153": "mem-off (baseline)",
    "FULL_20260617_180757":   "stage1 full (mem-on)",
    "stage2_step3000_FULL":   "stage2 step3000",
    "stage2_step4000_FULL":   "stage2 step4000",
    "stage2_step5000_FULL":   "stage2 step5000",
    "stage2_step6000_FULL":   "stage2 step6000",
}
# 去 Noise 的 6 个 factor(Noise 因 case 集不同、不是所有 run 都跑,故不计入公平 Total)
FACT = ["Camera", "Robot", "Lang.", "Light", "BG", "Layout"]


def parse_summary(path):
    R = list(csv.reader(open(path)))
    hdr = R[1]                      # ,Camera,...,Total
    idx = {c: hdr.index(c) for c in hdr}
    sr, succ, tot = R[2], R[3], R[4]  # Success Rate(%) / Successes / Total Cases

    def f(row, c):
        try:
            return float(row[idx[c]])
        except (KeyError, ValueError, IndexError):
            return None

    s = sum(f(succ, c) or 0 for c in FACT)
    t = sum(f(tot, c) or 0 for c in FACT)
    rec = {c: f(sr, c) for c in FACT}
    rec["TotalNN"] = 100 * s / t if t else None
    rec["TotalRaw"] = f(sr, "Total")
    rec["N"] = int(f(tot, "Total") or 0)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="evaluate_results/libero_plus")
    ap.add_argument("--csv", action="store_true", help="同时把表写到 root/_comparison.csv")
    args = ap.parse_args()

    recs = {}
    for p in sorted(glob.glob(os.path.join(args.root, "**", "summary.csv"), recursive=True)):
        d = os.path.basename(os.path.dirname(p))
        recs[LABEL.get(d, d)] = parse_summary(p)

    if not recs:
        print(f"没找到 summary.csv,root={args.root}")
        return

    cols = ["experiment"] + FACT + ["Total(noNoise)", "Total(raw)", "N"]

    def cell(v):
        return ("%.1f" % v) if isinstance(v, float) else "-"

    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    rows = []
    for name, r in sorted(recs.items(), key=lambda kv: (kv[1]["TotalNN"] or 0), reverse=True):
        row = [name] + [cell(r[c]) for c in FACT] + [cell(r["TotalNN"]), cell(r["TotalRaw"]), str(r["N"])]
        print("| " + " | ".join(row) + " |")
        rows.append(row)

    if args.csv:
        out = os.path.join(args.root, "_comparison.csv")
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            w.writerows(rows)
        print(f"\n[csv] {out}")


if __name__ == "__main__":
    main()
