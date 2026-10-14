"""Short training with the original LingBot RoboTwin recipe and published data."""
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
    for name in ("repo", "data", "base", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    assert args.steps > 0
    sha = subprocess.check_output(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
    ).strip()
    assert sha == "7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb", sha
    assert not subprocess.check_output(
        ["git", "-C", str(args.repo), "status", "--porcelain"], text=True
    ).strip()
    sys.path.insert(0, str(args.repo / "wan_va"))
    from configs import VA_CONFIGS

    official = copy.deepcopy(VA_CONFIGS["robotwin_train"])
    cfg = copy.deepcopy(official)
    overrides = dict(dataset_path=str(args.data),
                     empty_emb_path=str(args.data / "empty_emb.pt"),
                     wan22_pretrained_model_name_or_path=str(args.base),
                     save_root=str(args.output), enable_wandb=False,
                     load_worker=2, num_steps=args.steps, save_interval=args.steps)
    cfg.update(overrides)
    assert not cfg.get("resume_from")
    assert cfg.batch_size == 1 and cfg.gradient_accumulation_steps == 1
    assert cfg.action_dim == 30 and cfg.action_per_frame == 16
    assert cfg.env_type == "robotwin_tshape"
    for key in official:
        if key not in overrides:
            assert cfg[key] == official[key], key
    assert Path(cfg.empty_emb_path).is_file()
    assert (args.base / "transformer/config.json").is_file()
    subsets = {
        "lerobot_robotwin_eef_clean_50/adjust_bottle-demo_clean_collect_200-50": 50,
        "lerobot_robotwin_eef_aug_500/adjust_bottle-aloha-agilex_randomized_500-1000": 500,
    }
    coverage = {}
    for name, expected in subsets.items():
        root = args.data / name
        info = json.loads((root / "meta/info.json").read_text())
        episodes = [json.loads(line) for line in
                    (root / "meta/episodes.jsonl").read_text().splitlines()]
        assert len(episodes) == info["total_episodes"] == expected
        windows = 0
        for episode in episodes:
            index = episode["episode_index"]
            chunk = index // info["chunks_size"]
            assert (root / info["data_path"].format(
                episode_chunk=chunk, episode_index=index)).is_file()
            for window in episode["action_config"]:
                for camera in cfg.obs_cam_keys:
                    latent = (root / "latents" / f"chunk-{chunk:03d}" / camera /
                              f"episode_{index:06d}_{window['start_frame']}_{window['end_frame']}.pth")
                    assert latent.is_file(), str(latent)
                windows += 1
        assert windows > 0
        coverage[name] = dict(episodes=expected, complete_windows=windows)
    if int(os.environ.get("RANK", "0")) == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "resolved-smoke-contract.json").write_text(json.dumps({
            "official_source_sha": sha, "resolved_config": dict(cfg),
            "smoke_overrides": overrides, "dataset_coverage": coverage,
            "dataset_revision": "595d4acc7e026aa57afba218f4170c6d79e39103",
            "scope": "adjust_bottle clean and aug code smoke, not full benchmark training",
            "dataset_init_workers": 2, "multiprocessing_start_method": "spawn",
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        }, indent=2, default=str))
        print("LINGBOT_ROBOTWIN_NATIVE_CONTRACT_PASS", coverage, flush=True)
    VA_CONFIGS["robotwin_train"] = cfg
    import train
    train.MultiLatentLeRobotDataset = partial(train.MultiLatentLeRobotDataset, num_init_worker=2)
    train.init_logger()
    train.run(SimpleNamespace(config_name="robotwin_train", save_root=str(args.output)))


if __name__ == "__main__":
    main()
