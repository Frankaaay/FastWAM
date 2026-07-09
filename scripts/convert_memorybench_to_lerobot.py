#!/usr/bin/env python
"""Convert MemoryBench RLBench demos into the local LeRobot format used by FastWAM.

The official dataset is expected under:

    raw_hf/data/{train,test}/{put_block_back,rearrange_block,reopen_drawer}.zip

or the same paths already extracted as directories.

This script intentionally writes a simple image-in-parquet LeRobot dataset
(`use_videos=False`) so it does not depend on ffmpeg/video encoding during the
first conversion pass.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
import numpy as np
from PIL import Image

from fastwam.datasets.lerobot.lerobot.lerobot_dataset import CODEBASE_VERSION
from fastwam.datasets.lerobot.lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from fastwam.datasets.lerobot.lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FEATURES,
    DEFAULT_PARQUET_PATH,
    embed_images,
    get_hf_features_from_features,
    write_episode,
    write_episode_stats,
    write_info,
    write_json,
    write_stats,
    write_task,
)


MEMORYBENCH_TASKS = ("put_block_back", "rearrange_block", "reopen_drawer")
TASK_INSTRUCTIONS = {
    "put_block_back": "Put the block to the centre and then back to its initial position while pushing the button in between.",
    "rearrange_block": "Move the block not on the patch to the empty patch, then press the button, then move the block that has not been moved off the patch.",
    "reopen_drawer": "Close the drawer, then reopen the previously opened drawer while pushing the button in between.",
}
QUALITY_TASK = "success"
COARSE_TASK = "MemoryBench short-term memory"


@dataclass(frozen=True)
class CameraSpec:
    output_key: str
    obs_attr: str


def parse_camera_specs(raw: str) -> list[CameraSpec]:
    specs: list[CameraSpec] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            out_key, obs_attr = item.split(":", 1)
        else:
            out_key = item
            obs_attr = item
        specs.append(CameraSpec(output_key=out_key.strip(), obs_attr=obs_attr.strip()))
    if not specs:
        raise ValueError("At least one camera must be provided.")
    return specs


def image_feature_key(camera_key: str) -> str:
    return f"observation.images.{camera_key}"


def action_names(dim: int) -> list[str]:
    base = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    return base[:dim] if dim <= len(base) else [f"action_{i}" for i in range(dim)]


def state_names(dim: int) -> list[str]:
    base = ["x", "y", "z", "roll", "pitch", "yaw", "gripper_open", "gripper_closed"]
    return base[:dim] if dim <= len(base) else [f"state_{i}" for i in range(dim)]


def build_features(
    *,
    cameras: list[CameraSpec],
    image_shape: tuple[int, int, int],
    action_dim: int,
    state_dim: int,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": action_names(action_dim),
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": state_names(state_dim),
        },
    }
    for cam in cameras:
        features[image_feature_key(cam.output_key)] = {
            "dtype": "image",
            "shape": image_shape,
            "names": ["channels", "height", "width"],
        }
    return features


def make_info(*, fps: int, features: dict[str, dict[str, Any]]) -> dict[str, Any]:
    features = {**features, **DEFAULT_FEATURES}
    return {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": "franka_panda_memorybench",
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "total_videos": 0,
        "total_chunks": 0,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": fps,
        "splits": {},
        "data_path": DEFAULT_PARQUET_PATH,
        "video_path": None,
        "features": features,
    }


def find_task_source(raw_root: Path, split: str, task: str) -> Path:
    candidates = [
        raw_root / "data" / split / f"{task}.zip",
        raw_root / split / f"{task}.zip",
        raw_root / "data" / split / task,
        raw_root / split / task,
        raw_root / f"{task}.zip",
        raw_root / task,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find MemoryBench task '{task}' split '{split}' under {raw_root}. "
        f"Tried: {', '.join(str(p) for p in candidates)}"
    )


def prepare_source(path: Path, tmp_root: Path) -> Path:
    if path.is_dir():
        return path
    if path.suffix.lower() != ".zip":
        raise ValueError(f"Unsupported source type: {path}")
    out_dir = tmp_root / path.stem
    if out_dir.exists():
        return out_dir
    out_dir.mkdir(parents=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(out_dir)
    return out_dir


def find_demo_pickles(root: Path) -> list[Path]:
    names = {"low_dim_obs.pkl", "low_dim_obs.pickle", "variation_descriptions.pkl", "variation_descriptions.pickle"}
    candidates = [p for p in root.rglob("*") if p.name in names or (p.suffix in {".pkl", ".pickle"} and "low_dim" in p.name)]
    # RLBench stores one low_dim_obs.pkl per episode. Prefer those over metadata pickles.
    low_dim = [p for p in candidates if "low_dim_obs" in p.name]
    if low_dim:
        return sorted(low_dim)
    return sorted(candidates)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def as_demo_sequence(obj: Any) -> list[Any]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    if hasattr(obj, "_observations"):
        return list(obj._observations)
    if hasattr(obj, "observations"):
        return list(obj.observations)
    raise TypeError(f"Unsupported demo object type: {type(obj)}")


def obs_get(obs: Any, name: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def to_uint8_hwc(image: Any) -> np.ndarray:
    if image is None:
        raise ValueError("Image is None.")
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"))
    else:
        arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def image_to_pil(image: Any) -> Image.Image:
    return Image.fromarray(to_uint8_hwc(image), mode="RGB")


def infer_image_shape(obs: Any, cameras: list[CameraSpec]) -> tuple[int, int, int]:
    img = to_uint8_hwc(obs_get(obs, cameras[0].obs_attr))
    return (3, int(img.shape[0]), int(img.shape[1]))


def vector_from_attrs(obs: Any, names: list[str]) -> np.ndarray | None:
    parts = []
    for name in names:
        value = obs_get(obs, name)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        parts.append(arr)
    if not parts:
        return None
    return np.concatenate(parts, axis=0).astype(np.float32)


def extract_action(obs: Any, *, action_source: str, action_dim: int) -> np.ndarray:
    if action_source == "auto":
        candidates = [
            ["action"],
            ["joint_velocities"],
            ["joint_positions", "gripper_open"],
            ["gripper_pose", "gripper_open"],
        ]
    else:
        candidates = [[item.strip() for item in action_source.split("+") if item.strip()]]
    for names in candidates:
        vec = vector_from_attrs(obs, names)
        if vec is None:
            continue
        if vec.shape[0] >= action_dim:
            return vec[:action_dim].astype(np.float32)
    return np.zeros((action_dim,), dtype=np.float32)


def extract_state(obs: Any, *, state_source: str, state_dim: int) -> np.ndarray:
    if state_source == "auto":
        candidates = [
            ["gripper_pose", "gripper_open"],
            ["joint_positions", "gripper_open"],
            ["gripper_pose"],
            ["joint_positions"],
        ]
    else:
        candidates = [[item.strip() for item in state_source.split("+") if item.strip()]]
    for names in candidates:
        vec = vector_from_attrs(obs, names)
        if vec is None:
            continue
        out = np.zeros((state_dim,), dtype=np.float32)
        take = min(state_dim, vec.shape[0])
        out[:take] = vec[:take]
        return out
    return np.zeros((state_dim,), dtype=np.float32)


def get_instruction(task: str, demo_dir: Path) -> str:
    for name in ("variation_descriptions.pkl", "variation_descriptions.pickle"):
        path = demo_dir / name
        if path.exists():
            try:
                payload = load_pickle(path)
                if isinstance(payload, (list, tuple)) and payload:
                    return str(payload[0])
                if isinstance(payload, str):
                    return payload
            except Exception:
                pass
    return TASK_INSTRUCTIONS[task]


def task_tuple(task: str, instruction: str) -> list[str]:
    return [COARSE_TASK, instruction, QUALITY_TASK, QUALITY_TASK]


def pil_bytes(image: Image.Image) -> dict[str, bytes]:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {"bytes": buf.getvalue(), "path": None}


def compute_numeric_episode_stats(episode_arrays: dict[str, np.ndarray], features: dict[str, dict[str, Any]]) -> dict:
    stats = {}
    for key, data in episode_arrays.items():
        if features[key]["dtype"] in ("image", "video", "string"):
            continue
        stats[key] = {
            "min": np.min(data, axis=0, keepdims=data.ndim == 1),
            "max": np.max(data, axis=0, keepdims=data.ndim == 1),
            "mean": np.mean(data, axis=0, keepdims=data.ndim == 1),
            "std": np.std(data, axis=0, keepdims=data.ndim == 1),
            "count": np.array([len(data)]),
        }
    return stats


def write_dataset(
    *,
    out_root: Path,
    split: str,
    task_sources: dict[str, Path],
    cameras: list[CameraSpec],
    fps: int,
    action_dim: int,
    state_dim: int,
    action_source: str,
    state_source: str,
    limit_episodes: int | None,
    overwrite: bool,
    inspect_only: bool,
) -> None:
    if overwrite and out_root.exists() and not inspect_only:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_episode_stats = []
    info: dict[str, Any] | None = None
    episode_index = 0
    global_index = 0
    task_to_index: dict[str, int] = {}

    with tempfile.TemporaryDirectory(prefix="memorybench_convert_") as tmp:
        tmp_root = Path(tmp)
        for task in MEMORYBENCH_TASKS:
            if task not in task_sources:
                continue
            source_dir = prepare_source(task_sources[task], tmp_root)
            demo_pickles = find_demo_pickles(source_dir)
            if limit_episodes is not None:
                demo_pickles = demo_pickles[:limit_episodes]
            print(f"[{split}:{task}] source={task_sources[task]} extracted={source_dir} demos={len(demo_pickles)}")

            for demo_pickle in demo_pickles:
                demo = as_demo_sequence(load_pickle(demo_pickle))
                if not demo:
                    print(f"  skip empty demo: {demo_pickle}")
                    continue
                instruction = get_instruction(task, demo_pickle.parent)
                obs0 = demo[0]
                if info is None:
                    image_shape = infer_image_shape(obs0, cameras)
                    features = build_features(
                        cameras=cameras,
                        image_shape=image_shape,
                        action_dim=action_dim,
                        state_dim=state_dim,
                    )
                    info = make_info(fps=fps, features=features)
                    if inspect_only:
                        print(json.dumps(info, indent=2, default=str))
                if inspect_only:
                    attrs = sorted(k for k in dir(obs0) if not k.startswith("_"))
                    print(f"  demo={demo_pickle} len={len(demo)} obs_type={type(obs0)} attrs={attrs[:80]}")
                    continue

                assert info is not None
                features = info["features"]
                frame_count = len(demo)
                episode_data: dict[str, list[Any]] = {
                    key: [] for key in features if key not in {"index", "episode_index", "frame_index", "timestamp"}
                }
                episode_data["index"] = []
                episode_data["episode_index"] = []
                episode_data["frame_index"] = []
                episode_data["timestamp"] = []
                tasks = task_tuple(task, instruction)
                for t, obs in enumerate(demo):
                    episode_data["index"].append(global_index)
                    episode_data["episode_index"].append(episode_index)
                    episode_data["frame_index"].append(t)
                    episode_data["timestamp"].append(float(t) / float(fps))
                    episode_data["action"].append(extract_action(obs, action_source=action_source, action_dim=action_dim))
                    episode_data["observation.state"].append(
                        extract_state(obs, state_source=state_source, state_dim=state_dim)
                    )
                    for cam in cameras:
                        episode_data[image_feature_key(cam.output_key)].append(pil_bytes(image_to_pil(obs_get(obs, cam.obs_attr))))
                    global_index += 1

                for task_text in tasks:
                    if task_text not in task_to_index:
                        task_to_index[task_text] = len(task_to_index)
                        write_task(task_to_index[task_text], task_text, out_root)
                task_indices = {
                    "coarse_task_index": task_to_index[tasks[0]],
                    "task_index": task_to_index[tasks[1]],
                    "coarse_quality_index": task_to_index[tasks[2]],
                    "quality_index": task_to_index[tasks[3]],
                }
                for key, value in task_indices.items():
                    episode_data[key] = [value] * frame_count

                hf_features = get_hf_features_from_features(features)
                table_payload = {}
                for key, values in episode_data.items():
                    if key in {"action", "observation.state"}:
                        table_payload[key] = np.stack(values).astype(np.float32)
                    elif key in task_indices or key in {"index", "episode_index", "frame_index"}:
                        table_payload[key] = np.asarray(values, dtype=np.int64)
                    elif key == "timestamp":
                        table_payload[key] = np.asarray(values, dtype=np.float32)
                    else:
                        table_payload[key] = values

                ep_dataset = datasets.Dataset.from_dict(table_payload, features=hf_features, split="train")
                ep_dataset = embed_images(ep_dataset)
                ep_chunk = episode_index // DEFAULT_CHUNK_SIZE
                ep_path = out_root / DEFAULT_PARQUET_PATH.format(
                    episode_chunk=ep_chunk,
                    episode_index=episode_index,
                )
                ep_path.parent.mkdir(parents=True, exist_ok=True)
                ep_dataset.to_parquet(ep_path)

                numeric_arrays = {
                    "action": table_payload["action"],
                    "observation.state": table_payload["observation.state"],
                }
                ep_stats = compute_numeric_episode_stats(numeric_arrays, features)
                write_episode_stats(episode_index, ep_stats, out_root)
                all_episode_stats.append(ep_stats)
                write_episode(
                    {
                        "episode_index": episode_index,
                        "tasks": tasks,
                        "length": frame_count,
                        "raw_file_name": str(demo_pickle),
                    },
                    out_root,
                )
                episode_index += 1

    if inspect_only:
        return
    if info is None:
        raise RuntimeError("No demos were converted.")
    info["total_episodes"] = episode_index
    info["total_frames"] = global_index
    info["total_tasks"] = len(task_to_index)
    info["total_chunks"] = (episode_index + DEFAULT_CHUNK_SIZE - 1) // DEFAULT_CHUNK_SIZE
    info["splits"] = {"train": f"0:{episode_index}"}
    write_info(info, out_root)
    if all_episode_stats:
        write_stats(aggregate_stats(all_episode_stats), out_root)
    write_json(
        {
            "source": "hqfang/memorybench",
            "split": split,
            "tasks": list(task_sources),
            "num_episodes": episode_index,
            "num_frames": global_index,
            "cameras": [cam.__dict__ for cam in cameras],
            "action_source": action_source,
            "state_source": state_source,
        },
        out_root / "meta" / "conversion_info.json",
    )
    print(f"Wrote {episode_index} episodes / {global_index} frames to {out_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True, help="Path to MemoryBench raw_hf root.")
    parser.add_argument("--out-root", type=Path, required=True, help="Output directory for converted LeRobot data.")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--tasks", nargs="*", default=list(MEMORYBENCH_TASKS), choices=MEMORYBENCH_TASKS)
    parser.add_argument(
        "--cameras",
        default="image:front_rgb,wrist_image:wrist_rgb",
        help="Comma-separated output:observation_attr camera mapping.",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument(
        "--action-source",
        default="auto",
        help="Observation fields joined by '+', or 'auto'. Default tries action, joint_velocities, joint_positions+gripper_open, gripper_pose+gripper_open.",
    )
    parser.add_argument(
        "--state-source",
        default="auto",
        help="Observation fields joined by '+', or 'auto'. Default tries gripper_pose+gripper_open, joint_positions+gripper_open.",
    )
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cameras = parse_camera_specs(args.cameras)
    task_sources = {task: find_task_source(args.raw_root, args.split, task) for task in args.tasks}
    write_dataset(
        out_root=args.out_root,
        split=args.split,
        task_sources=task_sources,
        cameras=cameras,
        fps=args.fps,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        action_source=args.action_source,
        state_source=args.state_source,
        limit_episodes=args.limit_episodes,
        overwrite=args.overwrite,
        inspect_only=args.inspect_only,
    )


if __name__ == "__main__":
    main()
