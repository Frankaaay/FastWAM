"""Run the pinned RoboDojo team's LingBot adapter, with explicit smoke bounds."""
import argparse
import copy
from functools import partial
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def main():
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    for name in ("xpolicy", "data", "base", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    assert args.steps > 0
    sha = subprocess.check_output(
        ["git", "-C", str(args.xpolicy), "rev-parse", "HEAD"], text=True
    ).strip()
    assert sha == "53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2"
    assert not subprocess.check_output(
        ["git", "-C", str(args.xpolicy), "diff", "HEAD", "--name-only"], text=True
    ).strip()
    source = args.xpolicy / "policy/LingBot_VA/lingbot_va/wan_va"
    sys.path.insert(0, str(source))
    from configs import VA_CONFIGS
    from dataset.lerobot_latent_dataset import LatentLeRobotDataset

    cfg = copy.deepcopy(VA_CONFIGS["robotwin30_train"])
    cfg.update(dataset_path=str(args.data), empty_emb_path=str(args.data / "empty_emb.pt"),
               wan22_pretrained_model_name_or_path=str(args.base),
               save_root=str(args.output), enable_wandb=False, load_worker=2,
               num_steps=args.steps, save_interval=args.steps)
    assert cfg.action_dim == 30 and cfg.action_per_frame == 12
    assert cfg.batch_size == 1 and cfg.gradient_accumulation_steps == 8
    dataset = LatentLeRobotDataset(str(args.data), config=cfg)
    expected = sum(len(ep["action_config"]) for ep in dataset.meta.episodes.values())
    assert len(dataset) == expected > 0, "Missing camera latent windows"
    action_counts = []
    import torch
    for index in range(len(dataset)):
        sample = dataset[index]
        for key, value in sample.items():
            if torch.is_tensor(value):
                assert torch.isfinite(value).all(), (index, key)
        action_counts.append(int(sample["actions"].shape[2]))
    if int(os.environ.get("RANK", "0")) == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "resolved-smoke-contract.json").write_text(json.dumps({
            "adapter_source_sha": sha, "model_source": str(source),
            "resolved_config": dict(cfg), "complete_windows": expected,
            "training_action_steps_per_latent_frame": sorted(set(action_counts)),
            "published_inference_action_per_frame": cfg.action_per_frame,
            "scope": "published adapter code smoke; temporal cadence equality not asserted",
            "dataset_init_workers": 2, "multiprocessing_start_method": "spawn",
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        }, indent=2, default=str))
        print("LINGBOT_ROBODOJO_NATIVE_CONTRACT_PASS", expected, action_counts, flush=True)
    del dataset
    VA_CONFIGS["robotwin30_train"] = cfg
    import train
    train.MultiLatentLeRobotDataset = partial(train.MultiLatentLeRobotDataset, num_init_worker=2)
    train.init_logger()
    train.run(SimpleNamespace(config_name="robotwin30_train", save_root=str(args.output)))


if __name__ == "__main__":
    main()
