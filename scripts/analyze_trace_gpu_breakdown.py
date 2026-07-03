#!/usr/bin/env python3
"""一次性探针：把 trace 中 GPU kernel 时间归因到 gpu_user_annotation 窗口并按类型分类。

用法: python3 attn_gpu_breakdown_probe.py TRACE.json
"""
import bisect
import json
import re
import sys
from collections import defaultdict

ATTN_FWD = re.compile(r"fmha_cutlassF|flash_fwd|cutlassF")
ATTN_BWD = re.compile(r"fmha_cutlassB|flash_bwd|cutlassB")
GEMM = re.compile(r"gemm|nvjet|cublas|matmul", re.IGNORECASE)

TARGET_PREFIXES = ("model/", "train/", "mot/", "ProfilerStep")


def classify(name: str) -> str:
    if ATTN_FWD.search(name):
        return "attn_fwd"
    if ATTN_BWD.search(name):
        return "attn_bwd"
    if GEMM.search(name):
        return "gemm"
    return "other"


def main() -> None:
    path = sys.argv[1]
    with open(path) as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data

    # GPU 轨道上的注解窗口与 kernel 事件，按 (pid, tid) 分组
    gpu_annos = defaultdict(list)  # (pid,tid) -> [(ts, end, name)]
    kernels = defaultdict(list)  # (pid,tid) -> [(ts, dur, name)]
    cpu_anno_counts = defaultdict(int)
    for ev in events:
        cat = ev.get("cat")
        if cat == "gpu_user_annotation":
            name = ev.get("name", "")
            if name.startswith(TARGET_PREFIXES):
                key = (ev.get("pid"), ev.get("tid"))
                gpu_annos[key].append((ev["ts"], ev["ts"] + ev.get("dur", 0), name))
        elif cat == "kernel":
            key = (ev.get("pid"), ev.get("tid"))
            kernels[key].append((ev["ts"], ev.get("dur", 0), ev.get("name", "")))
        elif cat == "user_annotation":
            name = ev.get("name", "")
            if name.startswith(TARGET_PREFIXES):
                cpu_anno_counts[name] += 1

    # 归因：kernel 的 ts 落在同轨道注解 [ts, end) 内则计入（嵌套注解全部计入）
    per_anno = defaultdict(lambda: defaultdict(float))
    per_anno_kernels = defaultdict(int)
    anno_instances = defaultdict(int)
    for key, annos in gpu_annos.items():
        annos.sort()
        for _, _, name in annos:
            anno_instances[name] += 1
        starts = [a[0] for a in annos]
        for ts, dur, kname in kernels.get(key, []):
            idx = bisect.bisect_right(starts, ts)
            for j in range(idx - 1, -1, -1):
                a_ts, a_end, a_name = annos[j]
                if ts < a_end:
                    per_anno[a_name][classify(kname)] += dur
                    per_anno_kernels[a_name] += 1
                # 继续向前找嵌套的外层注解（外层 start 更小、end 更大）

    # 全局 kernel 类别汇总
    global_cls = defaultdict(float)
    global_cnt = defaultdict(int)
    top_kernels = defaultdict(float)
    for key, ks in kernels.items():
        for ts, dur, kname in ks:
            c = classify(kname)
            global_cls[c] += dur
            global_cnt[c] += 1
            top_kernels[kname.split("(")[0][:80]] += dur

    print("== per-annotation GPU kernel time (ms) ==")
    print("annotation\tgpu_inst\tcpu_inst\ttotal\tattn_fwd\tattn_bwd\tgemm\tother\tkernels")
    for name in sorted(per_anno, key=lambda n: -sum(per_anno[n].values())):
        cls = per_anno[name]
        total = sum(cls.values())
        print(
            f"{name}\t{anno_instances[name]}\t{cpu_anno_counts.get(name, 0)}\t"
            f"{total/1e3:.2f}\t{cls['attn_fwd']/1e3:.2f}\t{cls['attn_bwd']/1e3:.2f}\t"
            f"{cls['gemm']/1e3:.2f}\t{cls['other']/1e3:.2f}\t{per_anno_kernels[name]}"
        )

    print("\n== global kernel class totals (ms) ==")
    for c in sorted(global_cls, key=lambda x: -global_cls[x]):
        print(f"{c}\t{global_cls[c]/1e3:.2f}\t{global_cnt[c]}")

    print("\n== top 15 kernels by total time (ms) ==")
    for kname in sorted(top_kernels, key=lambda k: -top_kernels[k])[:15]:
        print(f"{top_kernels[kname]/1e3:.2f}\t{kname}")


if __name__ == "__main__":
    main()
