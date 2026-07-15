"""v4 history 消融各模式的 infer_action 端到端延迟基准（单卡，无仿真环境）。

测量五种模式的单次 replan 延迟（含 T5 文本编码 + VAE 编码 + prefill + denoise 循环）：
  none        完整 v4（A）
  video_only  丢弃 action history（B，结构性：token 不进 DiT）
  action_only 丢弃 video history（C，结构性：history latent 不进 DiT）
  no_history  两路都丢弃（D）
  off         完全不传 history，原版 first-frame KV 路径（base 参照）

另含数值等价自检（--check）：对 B/C/D，用同一 seed 对比「mask 屏蔽（is_pad/置零，
计算量与 A 相同）」与「结构性丢弃（drop_history_*）」的输出动作差异，验证两种实现
对分数等价。

用法（h200，fastwam env）：
  python experiments/libero/bench_v4_history_ablation.py \
    ckpt=/path/to/step_014470.pt task=libero_uncond_2cam224_1e-4 \
    EVALUATION.dataset_stats_path=checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
    EVALUATION.output_dir=/tmp/bench_v4 gpu_id=0 \
    '+EVALUATION.bench_iters=20' '+EVALUATION.bench_check_equiv=true'
"""

import json
import logging
import sys
import time
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)
if not OmegaConf.has_resolver("max"):
    OmegaConf.register_new_resolver("max", lambda x: max(x))
if not OmegaConf.has_resolver("split"):
    OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

BENCH_MODES = ("none", "video_only", "action_only", "no_history", "off")
HISTORY_RAW_FRAMES = 5  # 在线 buffer 窗口：range(-16, 1, 4) → 5 帧
HISTORY_ACTION_LEN = 20
BENCH_SEED = 1234


def _build_base_kwargs(model, cfg, device) -> dict:
    """构造与在线 eval 同形状的合成输入（内容随机，只测延迟）。"""
    video_size = cfg.data.train.get("video_size", [224, 224])
    height, width = int(video_size[0]), int(video_size[1])
    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    num_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    num_inference_steps = (
        int(cfg.get("eval_num_inference_steps", 20)) if num_steps_cfg is None else int(num_steps_cfg)
    )

    gen = torch.Generator(device="cpu").manual_seed(BENCH_SEED)
    dtype = model.torch_dtype
    input_image = (torch.rand((1, 3, height, width), generator=gen) * 2 - 1).to(device=device, dtype=dtype)
    history_video = (
        torch.rand((3, HISTORY_RAW_FRAMES, height, width), generator=gen) * 2 - 1
    ).to(device=device, dtype=dtype)
    # 末帧 = current observation，与在线 buffer 语义一致
    history_video[:, -1] = input_image[0]
    history_action = (torch.rand((HISTORY_ACTION_LEN, model.action_expert.action_dim), generator=gen) * 2 - 1).to(
        device=device, dtype=dtype
    )

    kwargs = {
        "prompt": DEFAULT_PROMPT.format(task="pick up the black bowl and place it on the plate"),
        "input_image": input_image,
        "action_horizon": action_horizon,
        "num_inference_steps": num_inference_steps,
        "seed": BENCH_SEED,
        "history_video": history_video,
        "history_action": history_action,
        "history_video_is_pad": torch.zeros((HISTORY_RAW_FRAMES,), dtype=torch.bool, device=device),
        "history_action_is_pad": torch.zeros((HISTORY_ACTION_LEN,), dtype=torch.bool, device=device),
    }
    if getattr(model, "proprio_dim", None) is not None:
        kwargs["proprio"] = torch.zeros((1, int(model.proprio_dim)), device=device, dtype=dtype)
    return kwargs


def _mode_kwargs(base: dict, mode: str, *, structural: bool = True) -> dict:
    """按消融 mode 生成 infer_action kwargs。

    structural=True  → drop_history_*（token 不进 DiT，省算力）
    structural=False → mask 屏蔽（is_pad/置零，计算量与完整 v4 相同），仅供等价自检
    """
    kw = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
    if mode == "off":
        for key in ("history_video", "history_action", "history_video_is_pad", "history_action_is_pad"):
            kw.pop(key, None)
        return kw
    if mode == "none":
        return kw
    if mode in ("video_only", "no_history"):  # 丢弃 action history
        if structural:
            kw.pop("history_action", None)
            kw.pop("history_action_is_pad", None)
            kw["drop_history_action"] = True
        else:
            kw["history_action"] = torch.zeros_like(kw["history_action"])
            kw["history_action_is_pad"] = torch.ones_like(kw["history_action_is_pad"])
    if mode in ("action_only", "no_history"):  # 丢弃 video history
        history_video = kw["history_video"].clone()  # [3,T,H,W]
        history_video[:, :-1] = history_video[:, -1:]
        kw["history_video"] = history_video
        if structural:
            kw.pop("history_video_is_pad", None)
            kw["drop_history_video"] = True
        else:
            video_is_pad = torch.ones_like(kw["history_video_is_pad"])
            video_is_pad[-1] = False
            kw["history_video_is_pad"] = video_is_pad
    return kw


def _run_once(model, kwargs) -> torch.Tensor:
    with torch.no_grad():
        return model.infer_action(**kwargs)["action"]


def _check_equivalence(model, base_kwargs) -> dict:
    """同一 seed 下对比 mask 屏蔽与结构性丢弃的输出动作，报告逐模式最大绝对差。"""
    report = {}
    for mode in ("video_only", "action_only", "no_history"):
        masked = _run_once(model, _mode_kwargs(base_kwargs, mode, structural=False))
        structural = _run_once(model, _mode_kwargs(base_kwargs, mode, structural=True))
        max_abs = float((masked - structural).abs().max())
        report[mode] = max_abs
        level = logging.WARNING if max_abs > 5e-2 else logging.INFO
        logging.log(level, "equiv check [%s]: max|Δaction| = %.3e %s",
                    mode, max_abs, "(EXCEEDS 5e-2, investigate!)" if max_abs > 5e-2 else "")
    return report


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig):
    logging.getLogger().setLevel(logging.INFO)
    device = str(cfg.EVALUATION.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    mixed = str(cfg.get("mixed_precision", "bf16")).strip().lower()
    model_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[mixed]

    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    model.load_checkpoint(str(cfg.ckpt))
    model = model.to(device).eval()
    if not getattr(model, "enable_mem_stage_v4", False):
        raise ValueError("本 bench 只针对 mem-stage-v4 模型（enable_mem_stage_v4=True）。")

    base_kwargs = _build_base_kwargs(model, cfg, device)
    modes = [m.strip() for m in str(cfg.EVALUATION.get("bench_modes", ",".join(BENCH_MODES))).split(",") if m.strip()]
    warmup = int(cfg.EVALUATION.get("bench_warmup", 3))
    iters = int(cfg.EVALUATION.get("bench_iters", 20))

    results = {
        "ckpt": str(cfg.ckpt),
        "device": device,
        "dtype": mixed,
        "num_inference_steps": base_kwargs["num_inference_steps"],
        "action_horizon": base_kwargs["action_horizon"],
        "warmup": warmup,
        "iters": iters,
        "modes": {},
    }

    if bool(cfg.EVALUATION.get("bench_check_equiv", True)):
        results["equiv_max_abs_diff"] = _check_equivalence(model, base_kwargs)

    for mode in modes:
        if mode not in BENCH_MODES:
            raise ValueError(f"unknown bench mode: {mode!r}, expected subset of {BENCH_MODES}")
        kwargs = _mode_kwargs(base_kwargs, mode, structural=True)
        for _ in range(warmup):
            _run_once(model, kwargs)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        samples_ms = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _run_once(model, kwargs)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            samples_ms.append((time.perf_counter() - t0) * 1000.0)
        t = torch.tensor(samples_ms)
        stats = {
            "mean_ms": float(t.mean()),
            "std_ms": float(t.std()),
            "p50_ms": float(t.median()),
            "min_ms": float(t.min()),
            "max_ms": float(t.max()),
            "hz": 1000.0 / float(t.mean()),
        }
        results["modes"][mode] = stats
        logging.info(
            "mode=%-12s mean=%.1f ms  p50=%.1f  min=%.1f  max=%.1f  (%.2f Hz)",
            mode, stats["mean_ms"], stats["p50_ms"], stats["min_ms"], stats["max_ms"], stats["hz"],
        )

    out_dir = Path(str(cfg.EVALUATION.get("output_dir", "evaluate_results/bench_v4_history_ablation")))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "bench_v4_history_ablation.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logging.info("bench results written to %s", out_file)


if __name__ == "__main__":
    main()
