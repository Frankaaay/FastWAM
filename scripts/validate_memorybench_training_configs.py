#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


TASKS = {
    "original": "memorybench_short_fastwam_original_4gpu_1e-5",
    "v4": "memorybench_short_v4_1e-5",
}
SHARED_EXPECTED = {
    "batch_size": 24,
    "num_workers": 8,
    "learning_rate": 1e-5,
    "num_epochs": 10,
    "max_steps": 10000,
    "gradient_accumulation_steps": 1,
    "weight_decay": 1e-2,
    "align_optimizer_param_order": True,
    "save_every": 0,
    "save_final_checkpoint": True,
    "eval_every": 100000,
}


def compose_task(task: str):
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        return compose(config_name="train", overrides=[f"task={task}"])


def main() -> None:
    configs = {name: compose_task(task) for name, task in TASKS.items()}
    for key, expected in SHARED_EXPECTED.items():
        values = {name: OmegaConf.select(cfg, key) for name, cfg in configs.items()}
        if any(value != expected for value in values.values()):
            raise ValueError(f"Config mismatch for {key}: expected {expected}, got {values}")

    expected_history = {"original": False, "v4": True}
    if str(configs["original"].resume) != str(configs["v4"].resume):
        raise ValueError("Original and v4 must use the same resume checkpoint")
    if str(configs["original"].wandb.group) != str(configs["v4"].wandb.group):
        raise ValueError("Original and v4 must use the same W&B comparison group")
    summary = {}
    for name, cfg in configs.items():
        if bool(cfg.model.enable_mem_stage_v4) != expected_history[name]:
            raise ValueError(f"{name} enable_mem_stage_v4 must be {expected_history[name]}")
        if bool(cfg.model.mot_checkpoint_mixed_attn):
            raise ValueError(f"{name} mot_checkpoint_mixed_attn must be false for the profiled setup")
        if int(cfg.data.train.processor.action_output_dim) != 8:
            raise ValueError(f"{name} action_output_dim must be 8")
        if int(cfg.model.action_dit_config.action_dim) != 8:
            raise ValueError(f"{name} ActionDiT action_dim must resolve to 8")
        if int(cfg.model.video_dit_config.action_dim) != 8:
            raise ValueError(f"{name} video action_dim must resolve to 8")
        if len(cfg.data.train.processor.delta_action_dim_mask.default) != 8:
            raise ValueError(f"{name} delta_action_dim_mask must have 8 entries")
        dataset_dir = Path(str(cfg.data.train.dataset_dirs[0])).expanduser()
        if dataset_dir.name != "memorybench_short_train_v2" or not dataset_dir.is_dir():
            raise FileNotFoundError(f"{name} v2 dataset not found: {dataset_dir}")
        cache_root = Path(str(cfg.data.train.vae_latent_cache_dir)).expanduser()
        if not cache_root.is_dir():
            raise FileNotFoundError(f"{name} VAE cache root not found: {cache_root}")
        if str(cfg.data.train.vae_latent_cache_model_id) != "Wan-AI/Wan2.2-TI2V-5B":
            raise ValueError(f"{name} VAE cache model id is incorrect")
        resume = Path(str(cfg.resume)).expanduser()
        if not resume.is_file():
            raise FileNotFoundError(f"{name} resume checkpoint not found: {resume}")
        if not bool(cfg.wandb.enabled) or str(cfg.wandb.mode) != "offline":
            raise ValueError(f"{name} W&B must be enabled in offline mode")
        summary[name] = {
            "task": TASKS[name],
            "history": expected_history[name],
            "dataset": str(dataset_dir),
            "cache_root": str(cache_root),
            "resume": str(resume),
            "wandb_group": str(cfg.wandb.group),
        }

    print(json.dumps({"status": "ok", "shared": SHARED_EXPECTED, "runs": summary}, indent=2))


if __name__ == "__main__":
    main()
