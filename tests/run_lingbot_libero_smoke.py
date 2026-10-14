"""Short native LingBot LIBERO training; no FastWAM model/data imports."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def main():
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
    ).strip(), "LingBot source must be clean"
    sys.path.insert(0, str(args.repo / "wan_va"))
    from configs import VA_CONFIGS

    official = copy.deepcopy(VA_CONFIGS["libero_train"])
    cfg = copy.deepcopy(official)
    overrides = {
        "dataset_path": str(args.data),
        "empty_emb_path": str(args.data / "empty_emb.pt"),
        "wan22_pretrained_model_name_or_path": str(args.base),
        "save_root": str(args.output),
        "enable_wandb": False,
        "num_steps": args.steps,
        "save_interval": args.steps,
        "load_worker": 2,
    }
    cfg.update(overrides)
    assert not cfg.get("resume_from"), "Initialize from official LingBot base"
    for key in official:
        if key not in overrides:
            assert cfg[key] == official[key], key
    assert cfg.batch_size == 1 and cfg.gradient_accumulation_steps == 10
    assert cfg.action_dim == 30 and cfg.used_action_channel_ids == list(range(7))
    assert (args.base / "transformer" / "config.json").is_file()
    assert Path(cfg.empty_emb_path).is_file()

    # Check all published windows before the native reader can filter any out.
    info = json.loads((args.data / "meta" / "info.json").read_text())
    episodes = [json.loads(line) for line in
                (args.data / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert len(episodes) == info["total_episodes"]
    windows = 0
    for episode in episodes:
        index = episode["episode_index"]
        chunk = index // info["chunks_size"]
        parquet = args.data / info["data_path"].format(
            episode_chunk=chunk, episode_index=index
        )
        assert parquet.is_file(), str(parquet)
        for window in episode["action_config"]:
            for camera in cfg.obs_cam_keys:
                latent = (args.data / "latents" / f"chunk-{chunk:03d}" / camera /
                          f"episode_{index:06d}_{window['start_frame']}_{window['end_frame']}.pth")
                assert latent.is_file(), str(latent)
            windows += 1
    assert windows > 0
    if int(os.environ.get("RANK", "0")) == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        evidence = {
            "official_source_sha": sha,
            "initialization": str(args.base),
            "episodes": len(episodes), "complete_two_camera_windows": windows,
            "smoke_overrides": overrides,
            "resolved_config": dict(cfg),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "meaning": "code-path smoke, not score reproduction",
        }
        (args.output / "resolved-smoke-contract.json").write_text(
            json.dumps(evidence, indent=2, default=str)
        )
        print("LINGBOT_NATIVE_LIBERO_CONTRACT_PASS", flush=True)
    VA_CONFIGS["libero_train"] = cfg
    import train
    train.run(SimpleNamespace(config_name="libero_train", save_root=str(args.output)))


if __name__ == "__main__":
    main()
