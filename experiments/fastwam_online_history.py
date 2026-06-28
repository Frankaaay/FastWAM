from collections import deque
from typing import Any

import numpy as np
import torch


class FastWAMOnlineHistoryBuffer:
    def __init__(
        self,
        *,
        action_dim: int,
        history_action_len: int = 20,
        history_video_past_steps: int = 16,
        action_video_freq_ratio: int = 4,
    ) -> None:
        self.action_dim = int(action_dim)
        self.history_action_len = int(history_action_len)
        self.history_video_past_steps = int(history_video_past_steps)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        if self.history_action_len <= 0:
            raise ValueError(f"`history_action_len` must be positive, got {self.history_action_len}")
        if self.action_dim <= 0:
            raise ValueError(f"`action_dim` must be positive, got {self.action_dim}")
        if self.action_video_freq_ratio <= 0:
            raise ValueError(
                f"`action_video_freq_ratio` must be positive, got {self.action_video_freq_ratio}"
            )
        if self.history_video_past_steps % self.action_video_freq_ratio != 0:
            raise ValueError(
                "`history_video_past_steps` must be divisible by `action_video_freq_ratio`, "
                f"got {self.history_video_past_steps} and {self.action_video_freq_ratio}."
            )

        self.video_offsets = list(
            range(-self.history_video_past_steps, 1, self.action_video_freq_ratio)
        )
        self._video_frames: deque[tuple[int, torch.Tensor]] = deque(
            maxlen=self.history_video_past_steps + 1
        )
        self._actions: deque[torch.Tensor] = deque(maxlen=self.history_action_len)

    def reset(self) -> None:
        self._video_frames.clear()
        self._actions.clear()

    def record_observation(self, step_idx: int, frame: torch.Tensor) -> None:
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]
        if frame.ndim != 3 or frame.shape[0] != 3:
            raise ValueError(
                f"`frame` must have shape [3,H,W] or [1,3,H,W], got {tuple(frame.shape)}"
            )
        self._video_frames.append((int(step_idx), frame.detach().to(device="cpu", dtype=torch.float32)))

    def record_action(self, action: torch.Tensor | np.ndarray | list[float]) -> None:
        action_tensor = torch.as_tensor(action, dtype=torch.float32).detach().cpu()
        if action_tensor.ndim == 2 and action_tensor.shape[0] == 1:
            action_tensor = action_tensor[0]
        if action_tensor.ndim != 1 or action_tensor.shape[0] != self.action_dim:
            raise ValueError(
                f"`action` must have shape [{self.action_dim}], got {tuple(action_tensor.shape)}"
            )
        self._actions.append(action_tensor)

    def build_condition(
        self,
        *,
        current_step: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        frames_by_step: dict[int, torch.Tensor] = {step: frame for step, frame in self._video_frames}
        current_step = int(current_step)
        current_frame = frames_by_step.get(current_step)
        if current_frame is None:
            raise ValueError(
                f"Current observation for step {current_step} is required before building history condition."
            )

        # 缺失的历史帧用最早已记录帧做边缘填充，匹配训练集 LeRobot 的 index clamp 行为
        # （越界 query 被 clamp 到 ep_start，即重复 episode 首帧）。否则 episode 起步时缺失帧
        # 用零填充，会让 VAE 时间压缩后的 current_video latent 与训练不一致（current 永不 dropout）。
        earliest_step = min(frames_by_step)
        earliest_frame = frames_by_step[earliest_step]
        frames = []
        video_is_pad = []
        for offset in self.video_offsets:
            frame = frames_by_step.get(current_step + offset)
            if frame is None:
                frames.append(earliest_frame)
                video_is_pad.append(True)
            else:
                frames.append(frame)
                video_is_pad.append(False)

        history_video = torch.stack(frames, dim=1).to(device=device, dtype=dtype, non_blocking=True)
        history_video_is_pad = torch.as_tensor(video_is_pad, dtype=torch.bool, device=device)

        history_action = torch.zeros(
            (self.history_action_len, self.action_dim),
            dtype=torch.float32,
        )
        history_action_is_pad = torch.ones((self.history_action_len,), dtype=torch.bool)
        actions = list(self._actions)[-self.history_action_len :]
        if actions:
            start = self.history_action_len - len(actions)
            history_action[start:] = torch.stack(actions, dim=0)
            history_action_is_pad[start:] = False

        return {
            "history_video": history_video,
            "history_action": history_action.to(device=device, dtype=dtype, non_blocking=True),
            "history_video_is_pad": history_video_is_pad,
            "history_action_is_pad": history_action_is_pad.to(device=device, non_blocking=True),
        }

    @staticmethod
    def enabled_for_model(model: Any) -> bool:
        return bool(getattr(model, "enable_mem_stage_v4", False))
