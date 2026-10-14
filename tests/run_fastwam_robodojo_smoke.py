"""Short Wan-initialized training using the RoboDojo team's vendored FastWAM."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    for name in ("xpolicy", "data", "stats", "text-cache", "backbone", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    assert args.steps > 0
    source_sha = subprocess.check_output(
        ["git", "-C", str(args.xpolicy), "rev-parse", "HEAD"], text=True
    ).strip()
    assert source_sha == "53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2", source_sha
    assert not subprocess.check_output(
        ["git", "-C", str(args.xpolicy), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    repo = args.xpolicy / "policy/FastWAM/FastWAM"
    sys.path[:0] = [str(repo / "src"), str(repo)]
    for path in (args.data / "meta/info.json", args.stats, args.backbone):
        assert path.is_file(), str(path)
    assert args.text_cache.is_dir()
    assert os.environ.get("DIFFSYNTH_MODEL_BASE_PATH")
    info = json.loads((args.data / "meta/info.json").read_text())
    assert info["features"]["action"]["shape"] == [14]

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    with initialize_config_dir(config_dir=str(repo / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["task=robotwin_uncond_3cam_384_1e-4"])
    # Matches XPolicyLab train.sh's native joint-data bindings. Paper batch/steps
    # are recorded separately; the short runtime budget is not a score recipe.
    overrides = {
        "seed": 1, "num_epochs": 16, "batch_size": 2,
        "gradient_accumulation_steps": 1, "num_workers": 2,
        "max_steps": args.steps, "save_every": args.steps, "log_every": 1,
        "eval_every": 0, "wandb.enabled": False, "output_dir": str(args.output),
        "model.action_dit_pretrained_path": str(args.backbone),
    }
    for split in ("train", "val"):
        overrides.update({
            f"data.{split}.dataset_dirs": [str(args.data)],
            f"data.{split}.text_embedding_cache_dir": str(args.text_cache),
            f"data.{split}.pretrained_norm_stats": str(args.stats),
        })
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value)
    assert cfg.resume is None and not cfg.model.skip_dit_load_from_pretrain
    assert cfg.model.model_id == "Wan-AI/Wan2.2-TI2V-5B"
    assert cfg.model.tokenizer_max_len == cfg.data.train.context_len == 128
    assert list(cfg.data.train.video_size) == [384, 320]
    assert cfg.data.train.processor.action_output_dim == 14
    assert cfg.data.train.processor.proprio_output_dim == 14
    assert cfg.data.train._target_.endswith("RobotVideoDataset")
    assert cfg.learning_rate == 1e-4 and cfg.weight_decay == 1e-2
    if int(os.environ.get("RANK", "0")) == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        contract = {
            "xpolicylab_sha": source_sha, "action_type": "joint",
            "embodiment": "arx_x5", "smoke_overrides": overrides,
            "resolved_config": OmegaConf.to_container(cfg, resolve=True),
            "paper": "https://arxiv.org/html/2607.04434v1#A11",
            "paper_batch_size": 256, "paper_training_steps": 20000,
            "script_default_per_device_batch": 8,
            "smoke_global_batch": 2 * int(os.environ.get("WORLD_SIZE", "1")),
            "scope": "code smoke; not full-data training or score reproduction",
        }
        (args.output / "resolved-smoke-contract.json").write_text(json.dumps(contract, indent=2))
        print("FASTWAM_ROBODOJO_NATIVE_CONTRACT_PASS", flush=True)
    from fastwam.runtime import run_training
    import torch

    # Test-only boundary checks locate corruption before a bad smoke checkpoint
    # can be accepted. Chunking avoids a model-sized temporary boolean tensor.
    original_step = torch.optim.AdamW.step

    def check_tensor(value, label):
        flat = value.detach().reshape(-1)
        for start in range(0, flat.numel(), 1048576):
            assert torch.isfinite(flat[start:start + 1048576]).all(), label

    def checked_step(optimizer, *step_args, **step_kwargs):
        # ZeRO1 creates a >2**31-element FP32 partition on this two-GPU smoke.
        # Use AdamW's single-tensor implementation for this runtime diagnostic;
        # learning rate, betas, epsilon and weight decay remain unchanged.
        for group in optimizer.param_groups:
            group["foreach"] = False
        for group_index, group in enumerate(optimizer.param_groups):
            for index, param in enumerate(group["params"]):
                label = f"rank={os.environ.get('RANK', '0')} group={group_index} param={index}"
                check_tensor(param, f"before AdamW parameter: {label}")
                if param.grad is not None:
                    check_tensor(param.grad, f"before AdamW gradient: {label}")
                print("ADAMW_BOUNDARY", label, "dtype", param.dtype,
                      "grad_dtype", None if param.grad is None else param.grad.dtype,
                      "eps", group.get("eps"), "lr", group.get("lr"),
                      "foreach", group.get("foreach"), flush=True)
        result = original_step(optimizer, *step_args, **step_kwargs)
        for group_index, group in enumerate(optimizer.param_groups):
            for index, param in enumerate(group["params"]):
                check_tensor(param, f"after AdamW parameter: group={group_index} param={index}")
        print("FASTWAM_NATIVE_ADAMW_FINITE_PASS", os.environ.get("RANK", "0"), flush=True)
        return result

    torch.optim.AdamW.step = checked_step
    try:
        run_training(cfg)
    finally:
        torch.optim.AdamW.step = original_step


if __name__ == "__main__":
    main()
