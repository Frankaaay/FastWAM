import hashlib
import json
import os
from pathlib import Path
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

class RobotVideoDataset(torch.utils.data.Dataset):
    HISTORY_ACTION_LEN = 20
    HISTORY_VIDEO_PAST_STEPS = 16

    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        vae_latent_cache_dir: Optional[str] = None,
        vae_latent_cache_keep_video: bool = False,
        vae_latent_cache_model_id: Optional[str] = None,
        vae_latent_cache_validate_metadata: bool = True,
        vae_latent_cache_precompute_only: bool = False,
    ):
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        self.history_action_len = self.HISTORY_ACTION_LEN
        self.history_video_past_steps = self.HISTORY_VIDEO_PAST_STEPS
        self.future_action_len = num_frames - 1
        self.current_raw_index = self.history_video_past_steps
        self.raw_num_frames = self.history_video_past_steps + num_frames

        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        assert self.history_video_past_steps % self.action_video_freq_ratio == 0, \
            f"history_video_past_steps must be divisible by action_video_freq_ratio, got {self.history_video_past_steps}"
        self.history_video_sample_indices = list(
            range(0, self.current_raw_index + 1, self.action_video_freq_ratio)
        )
        self.video_sample_indices = list(
            range(self.current_raw_index, self.current_raw_index + num_frames, self.action_video_freq_ratio)
        )

        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=self.raw_num_frames,
            past_obs_size=self.history_video_past_steps,
            action_size=self.history_action_len + self.future_action_len,
            past_action_size=self.history_action_len,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
        )

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.vae_latent_cache_dir = (
            str(Path(vae_latent_cache_dir).expanduser()) if vae_latent_cache_dir else None
        )
        self.vae_latent_cache_keep_video = bool(vae_latent_cache_keep_video)
        self.vae_latent_cache_model_id = (
            str(vae_latent_cache_model_id) if vae_latent_cache_model_id else None
        )
        self.vae_latent_cache_validate_metadata = bool(vae_latent_cache_validate_metadata)
        self.vae_latent_cache_precompute_only = bool(vae_latent_cache_precompute_only)

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        self.vae_latent_cache_metadata = self._build_vae_latent_cache_metadata(
            dataset_dirs=dataset_dirs,
            shape_meta=shape_meta,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
        )
        self.vae_latent_cache_fingerprint = self._hash_vae_latent_cache_metadata(
            self.vae_latent_cache_metadata
        )
        if self.vae_latent_cache_dir is not None:
            logger.info(
                "Using VAE latent cache: dir=%s fingerprint=%s keep_video=%s",
                self.vae_latent_cache_dir,
                self.vae_latent_cache_fingerprint,
                self.vae_latent_cache_keep_video,
            )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if hasattr(processor, "num_obs_steps"):
                processor.num_obs_steps = self.raw_num_frames
            processor.future_action_start_step = self.history_action_len
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
        
    def __len__(self):
        return len(self.lerobot_dataset)

    def _build_vae_latent_cache_metadata(
        self,
        *,
        dataset_dirs,
        shape_meta,
        val_set_proportion,
        is_training_set,
        global_sample_stride,
    ) -> dict:
        if isinstance(shape_meta, DictConfig):
            shape_meta_payload = OmegaConf.to_container(shape_meta, resolve=True)
        else:
            shape_meta_payload = shape_meta
        return {
            "schema": "fastwam_robot_video_vae_latents_v1",
            "dataset_dirs": [
                str(Path(str(path)).expanduser().resolve(strict=False))
                for path in dataset_dirs
            ],
            "shape_meta": shape_meta_payload,
            "num_frames": int(self.num_frames),
            "history_video_past_steps": int(self.history_video_past_steps),
            "action_video_freq_ratio": int(self.action_video_freq_ratio),
            "history_video_sample_indices": list(self.history_video_sample_indices),
            "video_sample_indices": list(self.video_sample_indices),
            "video_size": list(self.video_size),
            "camera_key": self.camera_key,
            "concat_multi_camera": self.concat_multi_camera,
            "val_set_proportion": float(val_set_proportion),
            "is_training_set": bool(is_training_set),
            "global_sample_stride": int(global_sample_stride),
        }

    @staticmethod
    def _hash_vae_latent_cache_metadata(metadata: dict) -> str:
        payload = json.dumps(metadata, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def vae_latent_cache_path(self, sample_idx: int, cache_dir: Optional[str | Path] = None) -> Path:
        root = cache_dir if cache_dir is not None else self.vae_latent_cache_dir
        if root is None:
            raise ValueError("vae_latent_cache_dir is not set.")
        sample_idx = int(sample_idx)
        shard = f"{sample_idx // 1000:06d}"
        return (
            Path(root).expanduser()
            / self.vae_latent_cache_fingerprint
            / shard
            / f"{sample_idx:09d}.pt"
        )

    def make_vae_latent_cache_payload(
        self,
        *,
        sample_idx: int,
        input_latents: torch.Tensor,
        history_video_latents: torch.Tensor,
        model_id: Optional[str] = None,
        vae_path: Optional[str] = None,
    ) -> dict:
        if input_latents.ndim != 4:
            raise ValueError(
                f"`input_latents` cache tensor must be 4D [C,T,H,W], got {tuple(input_latents.shape)}"
            )
        if history_video_latents.ndim != 4:
            raise ValueError(
                "`history_video_latents` cache tensor must be 4D [C,T,H,W], "
                f"got {tuple(history_video_latents.shape)}"
            )
        return {
            "schema": "fastwam_robot_video_vae_latents_v1",
            "sample_idx": int(sample_idx),
            "fingerprint": self.vae_latent_cache_fingerprint,
            "metadata": self.vae_latent_cache_metadata,
            "model_id": None if model_id is None else str(model_id),
            "vae_path": None if vae_path is None else str(vae_path),
            "input_latents": input_latents.detach().to("cpu").contiguous(),
            "history_video_latents": history_video_latents.detach().to("cpu").contiguous(),
        }

    def _load_vae_latent_cache(self, sample_idx: int) -> dict[str, torch.Tensor]:
        cache_path = self.vae_latent_cache_path(sample_idx)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing VAE latent cache for sample_idx={sample_idx}: {cache_path}. "
                "Run scripts/precompute_vae_latents.py first or unset vae_latent_cache_dir."
            )
        payload = torch.load(cache_path, map_location="cpu")
        if self.vae_latent_cache_validate_metadata:
            if payload.get("schema") != "fastwam_robot_video_vae_latents_v1":
                raise ValueError(f"Invalid VAE latent cache schema in {cache_path}")
            if int(payload.get("sample_idx", -1)) != int(sample_idx):
                raise ValueError(
                    f"VAE latent cache sample_idx mismatch in {cache_path}: "
                    f"expected {sample_idx}, got {payload.get('sample_idx')}"
                )
            if payload.get("fingerprint") != self.vae_latent_cache_fingerprint:
                raise ValueError(
                    f"VAE latent cache fingerprint mismatch in {cache_path}: "
                    f"expected {self.vae_latent_cache_fingerprint}, got {payload.get('fingerprint')}"
                )
            if self.vae_latent_cache_model_id is not None:
                payload_model_id = payload.get("model_id")
                if payload_model_id != self.vae_latent_cache_model_id:
                    raise ValueError(
                        f"VAE latent cache model_id mismatch in {cache_path}: "
                        f"expected {self.vae_latent_cache_model_id}, got {payload_model_id}"
                    )

        input_latents = payload.get("input_latents")
        history_video_latents = payload.get("history_video_latents")
        if not torch.is_tensor(input_latents) or input_latents.ndim != 4:
            raise ValueError(
                f"Cached input_latents must be 4D tensor [C,T,H,W] in {cache_path}"
            )
        if not torch.is_tensor(history_video_latents) or history_video_latents.ndim != 4:
            raise ValueError(
                f"Cached history_video_latents must be 4D tensor [C,T,H,W] in {cache_path}"
            )
        return {
            "input_latents": input_latents.contiguous(),
            "history_video_latents": history_video_latents.contiguous(),
        }

    def _select_and_format_video(self, pixel_values: torch.Tensor, indices: list[int]) -> torch.Tensor:
        video = pixel_values
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, indices, :, :, :]  # [num_cameras, T_video, C, H, W]
            num_cameras, t_video, c, h, w = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            video = video[indices, :, :, :]  # [T_video, C, H, W]
            t_video, c, h, w = video.shape

        video = video.reshape(num_cameras, t_video, c, h, w)
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )
            bottom = torch.cat([cam_left, cam_right], dim=-1)
            video = torch.cat([cam_top, bottom], dim=-2)
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)

        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)
        return video.permute(1, 0, 2, 3)  # [C, T_video, H, W], range [-1, 1]

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))

        sample_idx = int(sample.get("idx", sample_idx))
        image_is_pad = sample["image_is_pad"]
        history_video_is_pad = image_is_pad[self.history_video_sample_indices]
        image_is_pad = image_is_pad[self.video_sample_indices]

        latent_cache = None
        if self.vae_latent_cache_dir is not None:
            latent_cache = self._load_vae_latent_cache(sample_idx)

        video = None
        history_video = None
        if latent_cache is None or self.vae_latent_cache_keep_video:
            pixel_values = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
            history_video = self._select_and_format_video(pixel_values, self.history_video_sample_indices)
            video = self._select_and_format_video(pixel_values, self.video_sample_indices)

        if self.vae_latent_cache_precompute_only:
            if video is None or history_video is None:
                raise ValueError(
                    "`vae_latent_cache_precompute_only=true` requires videos to be returned. "
                    "Set `vae_latent_cache_keep_video=true` or unset `vae_latent_cache_dir`."
                )
            return {
                "sample_idx": torch.tensor(sample_idx, dtype=torch.long),
                "video": video,
                "history_video": history_video,
            }

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        raw_action = sample["action"] # [history + future, action_dim]
        raw_proprio = sample["proprio"] # [history_obs + future_obs, state_dim]
        history_action = raw_action[:self.history_action_len, :]
        action = raw_action[self.history_action_len:self.history_action_len + self.future_action_len, :]
        proprio = raw_proprio[self.current_raw_index:self.current_raw_index + self.future_action_len, :]
        history_action_is_pad = sample["action_is_pad"][:self.history_action_len]
        action_is_pad = sample["action_is_pad"][self.history_action_len:self.history_action_len + self.future_action_len]
        proprio_is_pad = sample["proprio_is_pad"][self.current_raw_index:self.current_raw_index + self.future_action_len]
        video_transitions = len(self.video_sample_indices) - 1
        if video is not None:
            if video.shape[1] <= 1:
                raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
            video_transitions = int(video.shape[1] - 1)
        if action.shape[0] % video_transitions != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video_transitions}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        
        data = {
            "sample_idx": torch.tensor(sample_idx, dtype=torch.long),
            "action": action,
            "history_action": history_action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "history_video_is_pad": history_video_is_pad,
            "action_is_pad": action_is_pad,
            "history_action_is_pad": history_action_is_pad,
            "proprio_is_pad": proprio_is_pad,
        }
        if video is not None:
            data["video"] = video
        if history_video is not None:
            data["history_video"] = history_video
        if latent_cache is not None:
            data.update(latent_cache)
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            if self.vae_latent_cache_dir is not None:
                raise
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
