from __future__ import annotations

import json
import logging
import math
import os
import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Callable

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
src_root = project_root / "src"
for import_root in (project_root, src_root):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

os.environ["TOKENIZERS_PARALLELISM"] = "false"


TASK_MODULES = {
    "put_block_back": ("rlbench.tasks.put_block_back", "PutBlockBack"),
    "rearrange_block": ("rlbench.tasks.rearrange_block", "RearrangeBlock"),
    "reopen_drawer": ("rlbench.tasks.reopen_drawer", "ReopenDrawer"),
}


def _register_resolver(name: str, func: Callable) -> None:
    OmegaConf.register_new_resolver(name, func, replace=True)


def _sum_shapes(shape_meta_list):
    if not shape_meta_list:
        return 0
    return sum(int(item["shape"]) for item in shape_meta_list if item["key"] is not None)


def _max_action_dim(embodiment_datasets_cfg):
    max_dim = 0
    for _, dataset_cfg in embodiment_datasets_cfg.items():
        if "shape_meta" in dataset_cfg:
            max_dim = max(max_dim, _sum_shapes(dataset_cfg.shape_meta.action))
    return max_dim


def _max_state_dim(embodiment_datasets_cfg):
    max_dim = 0
    for _, dataset_cfg in embodiment_datasets_cfg.items():
        if "shape_meta" in dataset_cfg:
            max_dim = max(max_dim, _sum_shapes(dataset_cfg.shape_meta.state))
    return max_dim


def _register_default_resolvers() -> None:
    _register_resolver("eval", eval)
    _register_resolver("split", lambda s, idx: s.split("/")[int(idx)])
    _register_resolver("max", lambda x: max(x))
    _register_resolver("round_up", math.ceil)
    _register_resolver("round_down", math.floor)
    _register_resolver("sum_shapes", _sum_shapes)
    _register_resolver("max_action_dim", _max_action_dim)
    _register_resolver("max_state_dim", _max_state_dim)


_register_default_resolvers()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return super().default(obj)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(f"Unsupported mixed_precision: {mixed_precision}")
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _expand_path(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value))))


def _require_path(path: str | os.PathLike[str], *, what: str) -> Path:
    resolved = _expand_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{what} not found: {resolved}")
    return resolved


def _resolve_dataset_stats_path(cfg: DictConfig, ckpt: Path) -> Path:
    explicit = cfg.memorybench_eval.get("dataset_stats_path")
    candidates: list[Path] = []
    if explicit:
        candidates.append(_expand_path(explicit))
    for parent in ckpt.parents[:6]:
        candidates.append(parent / "dataset_stats.json")

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        "Could not locate dataset_stats.json. Pass "
        "++memorybench_eval.dataset_stats_path=/path/to/dataset_stats.json."
    )


def _center_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize((width, height), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _obs_to_image(
    obs: Any,
    *,
    processor: Any,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    image_meta = processor.shape_meta["images"]
    if len(image_meta) != 2:
        raise ValueError(f"MemoryBench eval expects two cameras, got {len(image_meta)}")

    front_h, front_w = int(image_meta[0]["shape"][1]), int(image_meta[0]["shape"][2])
    wrist_h, wrist_w = int(image_meta[1]["shape"][1]), int(image_meta[1]["shape"][2])
    front = _center_resize(np.asarray(obs.front_rgb), width=front_w, height=front_h)
    wrist = _center_resize(np.asarray(obs.wrist_rgb), width=wrist_w, height=wrist_h)
    rgb = np.concatenate([front, wrist], axis=1)
    if rgb.shape[:2] != (height, width):
        raise ValueError(f"Unexpected model image shape {rgb.shape[:2]}, expected {(height, width)}")
    tensor = torch.as_tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    return tensor * (2.0 / 255.0) - 1.0


def _extract_state(obs: Any) -> np.ndarray:
    gripper_pose = np.asarray(obs.gripper_pose, dtype=np.float32).reshape(-1)
    gripper_open = np.asarray([float(obs.gripper_open)], dtype=np.float32)
    state = np.zeros((8,), dtype=np.float32)
    merged = np.concatenate([gripper_pose, gripper_open], axis=0)
    state[: min(len(merged), 8)] = merged[:8]
    return state


def _normalize_proprio(obs: Any, processor: Any) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError("MemoryBench eval expects one merged state key.")
    state_key = state_meta[0]["key"]
    state_batch = {"state": {state_key: torch.as_tensor(_extract_state(obs)).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _denormalize_action(action: torch.Tensor, processor: Any) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("MemoryBench eval expects one merged action key.")
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
    return denorm.numpy()


def _get_cached_text_context(prompt: str, cfg: DictConfig) -> tuple[torch.Tensor, torch.Tensor]:
    cache_dir = cfg.data.train.get("text_embedding_cache_dir")
    if cache_dir is None:
        raise ValueError("data.train.text_embedding_cache_dir is required for closed-loop eval.")
    context_len = int(cfg.data.train.get("context_len", 128))
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_path = _expand_path(cache_dir) / f"{hashed}.t5_len{context_len}.wan22ti2v5b.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing text embedding cache: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu")
    context = payload["context"]
    context_mask = payload["mask"].bool()
    context = context.clone()
    context[~context_mask] = 0.0
    context_mask = torch.ones_like(context_mask)
    return context, context_mask


def _model_action_to_env_action(action: np.ndarray, obs: Any, gripper_strategy: str) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] != 8:
        raise ValueError(f"Expected model action dim 8, got {action.shape}")

    if gripper_strategy == "keep":
        gripper = float(obs.gripper_open)
    elif gripper_strategy == "open":
        gripper = 1.0
    elif gripper_strategy == "close":
        gripper = 0.0
    elif gripper_strategy == "last_dim":
        gripper = float(action[-1] >= 0.5)
    else:
        raise ValueError(f"Unsupported gripper_strategy: {gripper_strategy}")
    env_action = action.copy()
    env_action[-1] = gripper
    return env_action


def _build_obs_config():
    from rlbench.observation_config import ObservationConfig

    obs_config = ObservationConfig()
    obs_config.set_all_high_dim(False)
    obs_config.front_camera.rgb = True
    obs_config.front_camera.image_size = (128, 128)
    obs_config.wrist_camera.rgb = True
    obs_config.wrist_camera.image_size = (128, 128)
    obs_config.joint_velocities = True
    obs_config.joint_positions = True
    obs_config.gripper_open = True
    obs_config.gripper_pose = True
    obs_config.record_ignore_collisions = True
    return obs_config


def _build_env(dataset_root: Path, task_name: str):
    import importlib
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment

    class JointVelocityIgnoreCollisions(JointVelocity):
        def action(self, scene, action, ignore_collisions=True):
            return super().action(scene, action)

    module_name, class_name = TASK_MODULES[task_name]
    task_cls = getattr(importlib.import_module(module_name), class_name)
    action_mode = MoveArmThenGripper(JointVelocityIgnoreCollisions(), Discrete())
    env = Environment(
        action_mode=action_mode,
        dataset_root=str(dataset_root),
        obs_config=_build_obs_config(),
        headless=True,
    )
    return env, task_cls


def _reset_to_demo(task_env, episode_idx: int):
    task_env.set_variation(-1)
    demo = task_env.get_demos(
        1,
        live_demos=False,
        random_selection=False,
        from_episode_number=int(episode_idx),
        image_paths=True,
    )[0]
    task_env.set_variation(demo.variation_number)
    desc, obs = task_env.reset_to_demo(demo)
    return desc, obs


def _predict_action_chunk(
    *,
    obs: Any,
    task_description: str,
    model: torch.nn.Module,
    processor: Any,
    cfg: DictConfig,
    history_buffer: FastWAMOnlineHistoryBuffer,
    current_step: int,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> np.ndarray:
    image = _obs_to_image(
        obs,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )
    proprio = _normalize_proprio(obs, processor)
    history_buffer.record_observation(current_step, image)
    prompt = (
        "A video recorded from a robot's point of view executing the following "
        f"instruction: {task_description}"
    )
    context, context_mask = _get_cached_text_context(prompt, cfg)

    infer_kwargs = {
        "prompt": None,
        "context": context,
        "context_mask": context_mask,
        "input_image": image,
        "action_horizon": action_horizon,
        "proprio": proprio,
        "negative_prompt": str(cfg.memorybench_eval.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.memorybench_eval.get("text_cfg_scale", 1.0)),
        "num_inference_steps": int(cfg.memorybench_eval.get("num_inference_steps", cfg.get("eval_num_inference_steps", 10))),
        "sigma_shift": (
            None
            if cfg.memorybench_eval.get("sigma_shift") is None
            else float(cfg.memorybench_eval.get("sigma_shift"))
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed) + int(current_step),
        "rand_device": str(cfg.memorybench_eval.get("rand_device", "cpu")),
        "tiled": bool(cfg.memorybench_eval.get("tiled", False)),
    }
    if FastWAMOnlineHistoryBuffer.enabled_for_model(model):
        infer_kwargs.update(
            history_buffer.build_condition(
                current_step=current_step,
                device=model_device,
                dtype=model.torch_dtype,
            )
        )

    with torch.no_grad():
        pred = model.infer_action(**infer_kwargs)
    return _denormalize_action(pred["action"][:action_horizon], processor)[0]


def run_episode(
    *,
    task_env,
    episode_idx: int,
    initial_reset: tuple[list[str], Any] | None = None,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    action_horizon: int,
    replan_steps: int,
    max_steps: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> dict[str, Any]:
    if initial_reset is None:
        desc, obs = _reset_to_demo(task_env, episode_idx)
    else:
        desc, obs = initial_reset
    task_description = desc[0] if desc else task_env.get_task_descriptions()[0]
    history_buffer = FastWAMOnlineHistoryBuffer(
        action_dim=int(model.action_expert.action_dim),
        history_action_len=int(getattr(model, "history_action_len", 20)),
        history_video_past_steps=16,
        action_video_freq_ratio=int(cfg.data.train.action_video_freq_ratio),
    )
    gripper_strategy = str(cfg.memorybench_eval.get("gripper_strategy", "last_dim"))
    pending_actions: list[np.ndarray] = []
    errors: list[str] = []
    success = False
    policy_step = 0

    for step in range(max_steps):
        try:
            if not pending_actions:
                chunk = _predict_action_chunk(
                    obs=obs,
                    task_description=task_description,
                    model=model,
                    processor=processor,
                    cfg=cfg,
                    history_buffer=history_buffer,
                    current_step=policy_step,
                    action_horizon=action_horizon,
                    input_w=input_w,
                    input_h=input_h,
                    model_device=model_device,
                )
                pending_actions = [chunk[i] for i in range(min(replan_steps, len(chunk)))]
            else:
                image = _obs_to_image(
                    obs,
                    processor=processor,
                    width=input_w,
                    height=input_h,
                    device=model_device,
                    dtype=model.torch_dtype,
                )
                history_buffer.record_observation(policy_step, image)

            model_action = np.asarray(pending_actions.pop(0), dtype=np.float32)
            env_action = _model_action_to_env_action(model_action, obs, gripper_strategy)
            obs, reward, terminate = task_env.step(env_action)
            history_buffer.record_action(model_action)
            success = bool(reward >= 1.0 or terminate)
            policy_step += 1
            if success:
                break
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            logging.exception("Episode %s step %s failed", episode_idx, step)
            break

    return {
        "episode": int(episode_idx),
        "success": bool(success),
        "steps": int(policy_step),
        "task_description": task_description,
        "errors": errors,
    }


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> dict[str, Any]:
    OmegaConf.set_struct(cfg, False)
    if "memorybench_eval" not in cfg:
        cfg.memorybench_eval = {}

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start_time = time.time()

    ckpt = _require_path(cfg.memorybench_eval.get("ckpt"), what="checkpoint")
    stats_path = _resolve_dataset_stats_path(cfg, ckpt)
    data_root = _expand_path(
        cfg.memorybench_eval.get(
            "data_root",
            os.environ.get("MEMORYBENCH_DATA_ROOT", "/data/shared/offline/datasets/memorybench"),
        )
    )
    dataset_root = _require_path(
        cfg.memorybench_eval.get("rlbench_dataset_root", data_root / "raw_hf" / "data" / "test"),
        what="MemoryBench RLBench test root",
    )
    task_name = str(cfg.memorybench_eval.get("task", "put_block_back"))
    if task_name not in TASK_MODULES:
        raise ValueError(f"Unsupported MemoryBench task: {task_name}")

    video_size = cfg.data.train.get("video_size", [224, 448])
    input_h, input_w = int(video_size[0]), int(video_size[1])
    action_horizon = int(cfg.memorybench_eval.get("action_horizon", int(cfg.data.train.num_frames) - 1))
    replan_steps = int(cfg.memorybench_eval.get("replan_steps", 10))
    max_steps = int(cfg.memorybench_eval.get("max_steps", 400))
    start_episode = int(cfg.memorybench_eval.get("start_episode", 0))
    eval_episodes = int(cfg.memorybench_eval.get("eval_episodes", 25))
    output_dir = _expand_path(
        cfg.memorybench_eval.get(
            "output_dir",
            Path("evaluate_results") / "memorybench_closed_loop" / time.strftime("%Y%m%d_%H%M%S"),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("MemoryBench closed-loop eval")
    logging.info("  task=%s episodes=%d:%d", task_name, start_episode, start_episode + eval_episodes)
    logging.info("  ckpt=%s", ckpt)
    logging.info("  stats=%s", stats_path)
    logging.info("  dataset_root=%s", dataset_root)

    env, task_cls = _build_env(dataset_root, task_name)
    episode_results: list[dict[str, Any]] = []
    try:
        env.launch()
        task_env = env.get_task(task_cls)
        first_reset = None
        if eval_episodes > 0:
            first_reset = _reset_to_demo(task_env, start_episode)
        global FastWAMOnlineHistoryBuffer, torch
        import torch as _torch
        from experiments.fastwam_online_history import FastWAMOnlineHistoryBuffer as _FastWAMOnlineHistoryBuffer
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        from fastwam.utils.pytorch_utils import set_global_seed

        torch = _torch
        FastWAMOnlineHistoryBuffer = _FastWAMOnlineHistoryBuffer
        if cfg.get("seed") is not None:
            set_global_seed(int(cfg.seed), get_worker_init_fn=False)
        dataset_stats = load_dataset_stats_from_json(str(stats_path))
        processor = instantiate(cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(dataset_stats)
        model_device = str(cfg.memorybench_eval.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
        model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
        model.load_checkpoint(str(ckpt))
        model = model.to(model_device).eval()
        logging.info("  device=%s use_history=%s", model_device, FastWAMOnlineHistoryBuffer.enabled_for_model(model))
        for ep in tqdm(range(start_episode, start_episode + eval_episodes), desc=task_name):
            result = run_episode(
                task_env=task_env,
                episode_idx=ep,
                initial_reset=first_reset if ep == start_episode else None,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                replan_steps=replan_steps,
                max_steps=max_steps,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            episode_results.append(result)
            interim = output_dir / f"{task_name}_episodes.jsonl"
            with interim.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, cls=NumpyEncoder) + "\n")
    finally:
        env.shutdown()

    successes = int(sum(bool(r["success"]) for r in episode_results))
    results = {
        "mode": "memorybench_closed_loop_rlbench",
        "task": task_name,
        "checkpoint": str(ckpt),
        "dataset_stats_path": str(stats_path),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "device": model_device,
        "use_history": FastWAMOnlineHistoryBuffer.enabled_for_model(model),
        "action_mode": "JointVelocity+Discrete",
        "gripper_strategy": str(cfg.memorybench_eval.get("gripper_strategy", "last_dim")),
        "action_horizon": action_horizon,
        "replan_steps": replan_steps,
        "max_steps": max_steps,
        "start_episode": start_episode,
        "eval_episodes": eval_episodes,
        "successes": successes,
        "success_rate": successes / max(len(episode_results), 1),
        "duration_sec": time.time() - start_time,
        "episodes": episode_results,
    }
    output_path = output_dir / f"{task_name}_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    logging.info("Saved results: %s", output_path)
    logging.info("Success: %d/%d", successes, len(episode_results))
    return results


if __name__ == "__main__":
    main()
