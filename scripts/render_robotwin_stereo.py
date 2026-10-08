"""Replay saved clean demonstrations with three parallel stereo camera rigs.

Requires matching original seed.txt, _traj_data and HDF5 observations. Writes
new episode directories, never overwrites originals or deletes render caches.
Run from a server with the pinned RoboTwin simulation assets installed.
"""
import argparse
import importlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

PAIRS = (("head_camera", "head_camera_right", "head_baseline_m"),
         ("left_camera", "left_camera_right", "left_wrist_baseline_m"),
         ("right_camera", "right_camera_right", "right_wrist_baseline_m"))


def verify_episode(original, replay, baselines, pixel_mae_limit=1.0, joint_atol=1e-5):
    import cv2
    import h5py
    errors = {}
    with h5py.File(original, "r") as source, h5py.File(replay, "r") as target:
        old_q, new_q = source["joint_action/vector"][:], target["joint_action/vector"][:]
        if old_q.shape != new_q.shape or old_q.ndim != 2 or old_q.shape[1] != 14 or not len(old_q):
            raise ValueError("RoboTwin Aloha frame/action shape mismatch")
        if not np.isfinite(old_q).all() or not np.isfinite(new_q).all():
            raise ValueError("Nonfinite RoboTwin joint observations")
        np.testing.assert_allclose(new_q, old_q, atol=joint_atol, rtol=0)
        for left, right, key in PAIRS:
            old, new = source[f"observation/{left}/rgb"], target[f"observation/{left}/rgb"]
            if len(old) != len(new) or len(new) != len(target[f"observation/{right}/rgb"]) or len(new) != len(old_q):
                raise ValueError(f"Frame count mismatch for {left}")
            max_mae = 0.0
            for index in range(len(old)):
                a = cv2.imdecode(np.frombuffer(bytes(old[index]), np.uint8), cv2.IMREAD_COLOR)
                b = cv2.imdecode(np.frombuffer(bytes(new[index]), np.uint8), cv2.IMREAD_COLOR)
                r = cv2.imdecode(np.frombuffer(bytes(target[f"observation/{right}/rgb"][index]), np.uint8), cv2.IMREAD_COLOR)
                if a is None or b is None or r is None or a.shape != b.shape or r.shape != a.shape:
                    raise ValueError(f"Invalid stereo image at {left}/{index}")
                mae = float(np.abs(a.astype(np.float32) - b).mean())
                max_mae = max(max_mae, mae)
                if mae > pixel_mae_limit:
                    raise ValueError(f"Replay diverged at {left}/{index}: pixel MAE={mae}")
            kl = target[f"observation/{left}/intrinsic_cv"][:]
            kr = target[f"observation/{right}/intrinsic_cv"][:]
            el = target[f"observation/{left}/extrinsic_cv"][:]
            er = target[f"observation/{right}/extrinsic_cv"][:]
            if any(not np.isfinite(matrix).all() for matrix in (kl, kr, el, er)):
                raise ValueError(f"Nonfinite calibration: {left}")
            np.testing.assert_allclose(kl, kr, atol=1e-6, rtol=0)
            rotation_l, rotation_r = el[:, :3, :3], er[:, :3, :3]
            np.testing.assert_allclose(rotation_l, rotation_r, atol=1e-5, rtol=0)
            center_l = -np.einsum("nji,nj->ni", rotation_l, el[:, :3, 3])
            center_r = -np.einsum("nji,nj->ni", rotation_r, er[:, :3, 3])
            offset_cv = np.einsum("nij,nj->ni", rotation_l, center_r - center_l)
            expected = np.zeros_like(offset_cv)
            expected[:, 0] = baselines[key]
            np.testing.assert_allclose(offset_cv, expected, atol=1e-5, rtol=0)
            errors[left] = max_mae
    return dict(frames=len(old_q), max_left_pixel_mae=errors, joint_atol=joint_atol,
                pixel_mae_limit=pixel_mae_limit)


def replay(args):
    import yaml
    robotwin = Path(args.robotwin_root).resolve(strict=True)
    source_root = Path(args.source_root).resolve(strict=True)
    output_root = Path(args.output_root).resolve()
    if source_root == output_root or source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError("Source and output roots must be disjoint")
    config_path = Path(args.task_config_path).resolve(strict=True)
    config = yaml.safe_load(config_path.read_text())
    if config.get("embodiment") != ["aloha-agilex"]:
        raise ValueError("This experiment uses official Aloha clean demonstrations only")
    randomization = config["domain_randomization"]
    if any(randomization.get(key, False) for key in
           ("cluttered_table", "random_background", "random_light", "random_table_height", "random_head_camera_dis")):
        raise ValueError("Expected the original clean task config")
    baselines = dict(head_baseline_m=args.head_baseline_m,
                     left_wrist_baseline_m=args.left_wrist_baseline_m,
                     right_wrist_baseline_m=args.right_wrist_baseline_m)
    if any(not np.isfinite(value) or value <= 0 for value in baselines.values()):
        raise ValueError("Baselines must be finite and positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index/count")
    if not np.isfinite(args.pixel_mae_limit) or not np.isfinite(args.joint_atol) or args.pixel_mae_limit < 0 or args.joint_atol < 0:
        raise ValueError("Invalid alignment thresholds")
    tasks = args.tasks or sorted(path.name for path in source_root.iterdir()
                                 if path.is_dir() and (path / "demo_clean").is_dir())
    if not tasks:
        raise ValueError("No clean task directories found")
    # Resolve every selected input before rendering. Each job gets disjoint episodes.
    episodes = []
    for task in tasks:
        folder = source_root / task / "demo_clean"
        seeds = [int(value) for value in (folder / "seed.txt").read_text().split()]
        files = sorted((folder / "data").glob("episode*.hdf5"), key=lambda p: int(p.stem[7:]))
        if not files or [int(p.stem[7:]) for p in files] != list(range(len(files))) or len(seeds) < len(files):
            raise ValueError(f"Incomplete original episode index for {task}")
        for index, file in enumerate(files):
            trajectory = folder / "_traj_data" / f"episode{index}.pkl"
            if not trajectory.is_file():
                raise FileNotFoundError(trajectory)
            episodes.append((task, index, seeds[index], file, trajectory))
    sys.path.insert(0, str(robotwin))
    os.chdir(robotwin)  # RoboTwin resolves embodiment/assets from its working directory.
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH
    embodiments = yaml.safe_load((Path(CONFIGS_PATH) / "_embodiment_config.yml").read_text())
    robot_file = embodiments["aloha-agilex"]["file_path"]
    embodiment_config = yaml.safe_load((Path(robot_file) / "config.yml").read_text())
    for ordinal, (task_name, index, seed, source, trajectory) in enumerate(episodes):
        if ordinal % args.num_shards != args.shard_index:
            continue
        destination = output_root / task_name / "demo_clean" / f"episode{index}"
        destination.mkdir(parents=True, exist_ok=False)
        options = dict(config, task_name=task_name, task_config="demo_clean", save_path=str(destination),
                       left_robot_file=robot_file, right_robot_file=robot_file,
                       left_embodiment_config=embodiment_config, right_embodiment_config=embodiment_config,
                       dual_arm_embodied=True, embodiment_name="aloha-agilex", need_plan=False,
                       render_freq=0, save_data=True, stereo=dict(enabled=True, **baselines))
        with trajectory.open("rb") as stream:
            recorded = pickle.load(stream)
        options.update(left_joint_path=recorded["left_joint_path"], right_joint_path=recorded["right_joint_path"])
        env = getattr(importlib.import_module(f"envs.{task_name}"), task_name)()
        try:
            env.setup_demo(now_ep_num=index, seed=seed, **options)
            env.set_path_lst(options)
            scene_info = env.play_once()
            if not env.check_success():
                raise RuntimeError(f"Replay failed: {task_name}/{index}, seed={seed}")
            env.merge_pkl_to_hdf5_video()
            generated = destination / "data" / f"episode{index}.hdf5"
            result = verify_episode(source, generated, baselines, args.pixel_mae_limit, args.joint_atol)
            result.update(task=task_name, episode=index, seed=seed, source=str(source),
                          source_trajectory=str(trajectory), baselines=baselines, scene_info=scene_info,
                          task_config_path=str(config_path), task_config=config)
            (destination / "verified.json").write_text(json.dumps(result, indent=2, default=str))
            print(f"Verified {task_name}/{index}: {result['frames']} frames", flush=True)
        finally:
            env.close_env(clear_cache=True)
        # Preserve caches for review. Do not call remove_data_cache or collect_data.sh.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", default=str(Path(__file__).resolve().parents[1] / "third_party/RoboTwin"))
    parser.add_argument("--source-root", required=True, help="Original raw task/demo_clean tree")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--task-config-path", required=True, help="Exact original demo_clean.yml")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--head-baseline-m", type=float, default=0.06)
    parser.add_argument("--left-wrist-baseline-m", type=float, default=0.02)
    parser.add_argument("--right-wrist-baseline-m", type=float, default=0.02)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--pixel-mae-limit", type=float, default=1.0)
    parser.add_argument("--joint-atol", type=float, default=1e-5)
    replay(parser.parse_args())
