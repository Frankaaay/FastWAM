from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class FastWAM(torch.nn.Module):
    """MoT world model with video/action experts."""
    SOURCE_HISTORY_VIDEO = 0
    SOURCE_CURRENT_VIDEO = 1
    SOURCE_HISTORY_ACTION = 2
    SOURCE_FUTURE_ACTION = 3
    SOURCE_FUTURE_VIDEO = 4
    HISTORY_ACTION_LEN = 20
    CURRENT_TIMELINE_INDEX = 20
    VIDEO_LATENT_TIMELINE_STRIDE = 4
    HISTORY_CONDITION_DROPOUT = 0.2

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.history_action_len = self.HISTORY_ACTION_LEN
        self.current_timeline_index = self.CURRENT_TIMELINE_INDEX
        self.history_condition_dropout = self.HISTORY_CONDITION_DROPOUT
        self.enable_mem_stage_v4 = self.__class__ is FastWAM

        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "FastWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )

        history_video = sample.get("history_video", None) if self.enable_mem_stage_v4 else None
        history_video_is_pad = sample.get("history_video_is_pad", None) if self.enable_mem_stage_v4 else None
        if history_video is not None:
            if history_video.ndim != 5:
                raise ValueError(
                    f"`sample['history_video']` must be 5D [B, 3, T, H, W], got shape {tuple(history_video.shape)}"
                )
            if history_video.shape[0] != batch_size or history_video.shape[1] != 3:
                raise ValueError(
                    "`sample['history_video']` shape mismatch: "
                    f"got {tuple(history_video.shape)} vs expected batch={batch_size}, channels=3"
                )
            if history_video.shape[3] != height or history_video.shape[4] != width:
                raise ValueError(
                    "`sample['history_video']` spatial shape must match `sample['video']`, "
                    f"got {tuple(history_video.shape[3:])} vs {(height, width)}"
                )
            if history_video.shape[2] % 4 != 1:
                raise ValueError(
                    f"`sample['history_video']` T must satisfy T % 4 == 1, got T={history_video.shape[2]}"
                )
            if history_video_is_pad is not None:
                if history_video_is_pad.ndim != 2:
                    raise ValueError(
                        "`sample['history_video_is_pad']` must be 2D [B, T], "
                        f"got shape {tuple(history_video_is_pad.shape)}"
                    )
                if history_video_is_pad.shape != (batch_size, history_video.shape[2]):
                    raise ValueError(
                        "`sample['history_video_is_pad']` shape mismatch: "
                        f"got {tuple(history_video_is_pad.shape)} vs expected {(batch_size, history_video.shape[2])}"
                    )

        history_action = sample.get("history_action", None) if self.enable_mem_stage_v4 else None
        history_action_is_pad = sample.get("history_action_is_pad", None) if self.enable_mem_stage_v4 else None
        if history_action is not None:
            if history_action.ndim != 3:
                raise ValueError(
                    f"`sample['history_action']` must be 3D [B, T, a_dim], got shape {tuple(history_action.shape)}"
                )
            if history_action.shape[0] != batch_size or history_action.shape[2] != action.shape[2]:
                raise ValueError(
                    "`sample['history_action']` shape mismatch: "
                    f"got {tuple(history_action.shape)} vs batch={batch_size}, action_dim={action.shape[2]}"
                )
            if history_action_is_pad is not None:
                if history_action_is_pad.ndim != 2:
                    raise ValueError(
                        "`sample['history_action_is_pad']` must be 2D [B, T], "
                        f"got shape {tuple(history_action_is_pad.shape)}"
                    )
                if history_action_is_pad.shape != history_action.shape[:2]:
                    raise ValueError(
                        "`sample['history_action_is_pad']` shape mismatch: "
                        f"got {tuple(history_action_is_pad.shape)} vs expected {tuple(history_action.shape[:2])}"
                    )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)
        history_video_latents = None
        if history_video is not None:
            history_video_input = history_video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            history_video_latents = self._encode_video_latents(history_video_input, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if history_action is not None:
            history_action = history_action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if history_video_is_pad is not None:
            history_video_is_pad = history_video_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if history_action_is_pad is not None:
            history_action_is_pad = history_action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "history_video_latents": history_video_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "history_action": history_action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
            "history_video_is_pad": history_video_is_pad,
            "history_action_is_pad": history_action_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def _action_source_ids(
        self,
        batch_size: int,
        seq_len: int,
        source_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.full((batch_size, seq_len), int(source_id), dtype=torch.long, device=device)

    def _action_position_ids(self, seq_len: int, start: int, device: torch.device) -> torch.Tensor:
        return torch.arange(start, start + seq_len, dtype=torch.long, device=device)

    def _history_video_source_and_position_ids(
        self,
        num_latent_frames: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_latent_frames <= 0:
            raise ValueError(f"`num_latent_frames` must be positive, got {num_latent_frames}")
        source_ids = torch.full(
            (num_latent_frames,),
            self.SOURCE_HISTORY_VIDEO,
            dtype=torch.long,
            device=device,
        )
        source_ids[-1] = self.SOURCE_CURRENT_VIDEO
        if num_latent_frames == 1:
            position_ids = torch.tensor([self.current_timeline_index], dtype=torch.long, device=device)
        else:
            position_ids = torch.linspace(
                4,
                self.current_timeline_index,
                steps=num_latent_frames,
                device=device,
            ).round().to(dtype=torch.long)
            position_ids[-1] = self.current_timeline_index
        return source_ids, position_ids

    def _combined_video_source_and_position_ids(
        self,
        clean_prefix_latent_frames: int,
        total_latent_frames: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if clean_prefix_latent_frames <= 0:
            raise ValueError(
                f"`clean_prefix_latent_frames` must be positive, got {clean_prefix_latent_frames}"
            )
        if total_latent_frames < clean_prefix_latent_frames:
            raise ValueError(
                "`total_latent_frames` must be >= clean prefix, "
                f"got total={total_latent_frames}, clean_prefix={clean_prefix_latent_frames}"
            )

        prefix_source_ids, prefix_position_ids = self._history_video_source_and_position_ids(
            num_latent_frames=clean_prefix_latent_frames,
            device=device,
        )
        future_latent_frames = total_latent_frames - clean_prefix_latent_frames
        if future_latent_frames == 0:
            return prefix_source_ids, prefix_position_ids

        future_source_ids = torch.full(
            (future_latent_frames,),
            self.SOURCE_FUTURE_VIDEO,
            dtype=torch.long,
            device=device,
        )
        future_position_ids = self.current_timeline_index + self.VIDEO_LATENT_TIMELINE_STRIDE * torch.arange(
            1,
            future_latent_frames + 1,
            dtype=torch.long,
            device=device,
        )
        source_ids = torch.cat([prefix_source_ids, future_source_ids], dim=0)
        position_ids = torch.cat([prefix_position_ids, future_position_ids], dim=0)
        return source_ids, position_ids

    def _latent_valid_from_raw_pad(
        self,
        raw_is_pad: Optional[torch.Tensor],
        num_latent_frames: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if raw_is_pad is None:
            return torch.ones((batch_size, num_latent_frames), dtype=torch.bool, device=device)
        if raw_is_pad.ndim != 2:
            raise ValueError(f"`raw_is_pad` must be 2D [B,T], got shape {tuple(raw_is_pad.shape)}")
        if raw_is_pad.shape[0] != batch_size:
            raise ValueError(
                f"`raw_is_pad` batch mismatch: got {raw_is_pad.shape[0]} vs expected {batch_size}"
            )
        raw_is_pad = raw_is_pad.to(device=device, dtype=torch.bool)
        if raw_is_pad.shape[1] == num_latent_frames:
            return ~raw_is_pad

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if raw_is_pad.shape[1] < 1 or (raw_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align raw padding mask with latent video steps: "
                f"raw_steps={raw_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )
        latent_steps = 1 + (raw_is_pad.shape[1] - 1) // temporal_factor
        if latent_steps != num_latent_frames:
            raise ValueError(
                "Raw padding mask latent length mismatch: "
                f"mask_latent_steps={latent_steps} vs latent_frames={num_latent_frames}."
            )
        tail_is_pad = raw_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(raw_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        latent_is_pad = torch.cat([raw_is_pad[:, :1], latent_tail_is_pad], dim=1)
        return ~latent_is_pad

    @staticmethod
    def _latent_valid_to_token_valid(latent_valid: torch.Tensor, tokens_per_frame: int) -> torch.Tensor:
        if latent_valid.ndim != 2:
            raise ValueError(f"`latent_valid` must be 2D [B,F], got shape {tuple(latent_valid.shape)}")
        if tokens_per_frame <= 0:
            raise ValueError(f"`tokens_per_frame` must be positive, got {tokens_per_frame}")
        return latent_valid.repeat_interleave(tokens_per_frame, dim=1)

    def _condition_dropout_masks(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training and self.history_condition_dropout > 0:
            drop_video = torch.rand((batch_size,), device=device) < self.history_condition_dropout
            drop_action = torch.rand((batch_size,), device=device) < self.history_condition_dropout
        else:
            drop_video = torch.zeros((batch_size,), dtype=torch.bool, device=device)
            drop_action = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        return drop_video, drop_action

    @staticmethod
    def _concat_kv_caches(*caches: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        caches = [cache for cache in caches if cache]
        if not caches:
            raise ValueError("At least one KV cache is required.")
        num_layers = len(caches[0])
        for cache in caches:
            if len(cache) != num_layers:
                raise ValueError("All KV caches must have the same number of layers.")
        combined: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            combined.append(
                {
                    "k": torch.cat([cache[layer_idx]["k"] for cache in caches], dim=1),
                    "v": torch.cat([cache[layer_idx]["v"] for cache in caches], dim=1),
                }
            )
        return combined

    @staticmethod
    def _slice_kv_cache(cache: list[dict[str, torch.Tensor]], key_len: int) -> list[dict[str, torch.Tensor]]:
        if key_len <= 0:
            raise ValueError(f"`key_len` must be positive, got {key_len}")
        sliced: list[dict[str, torch.Tensor]] = []
        for layer_idx, layer_cache in enumerate(cache):
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(f"`cache[{layer_idx}]` must contain `k` and `v`.")
            if layer_cache["k"].shape[1] < key_len or layer_cache["v"].shape[1] < key_len:
                raise ValueError(
                    f"`cache[{layer_idx}]` shorter than requested key_len={key_len}: "
                    f"k={tuple(layer_cache['k'].shape)}, v={tuple(layer_cache['v'].shape)}"
                )
            sliced.append(
                {
                    "k": layer_cache["k"][:, :key_len],
                    "v": layer_cache["v"][:, :key_len],
                }
            )
        return sliced

    def _compute_action_loss(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        timestep_action: torch.Tensor,
    ) -> torch.Tensor:
        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        return (action_loss_per_sample * action_weight).mean()

    def _training_loss_v4(self, inputs: dict[str, Any]):
        input_latents = inputs["input_latents"]
        history_video_latents = inputs["history_video_latents"]
        history_action = inputs["history_action"]
        if history_video_latents is None or history_action is None:
            raise ValueError("mem-stage-v4 training requires both `history_video` and `history_action`.")

        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        history_video_is_pad = inputs["history_video_is_pad"]
        history_action_is_pad = inputs["history_action_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]
        if input_latents.shape[2] <= 1:
            raise ValueError("mem-stage-v4 video loss requires at least one future video latent frame.")
        clean_prefix_latent_frames = int(history_video_latents.shape[2])
        if clean_prefix_latent_frames <= 0:
            raise ValueError("mem-stage-v4 requires non-empty `history_video_latents`.")
        future_noisy_latents = latents[:, :, 1:]
        target_video = target_video[:, :, 1:]
        combined_video_latents = torch.cat([history_video_latents, future_noisy_latents], dim=2)

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        future_source_ids = self._action_source_ids(
            batch_size=batch_size,
            seq_len=action.shape[1],
            source_id=self.SOURCE_FUTURE_ACTION,
            device=action.device,
        )
        future_position_ids = self._action_position_ids(
            seq_len=action.shape[1],
            start=self.current_timeline_index,
            device=action.device,
        )

        video_source_ids, video_position_ids = self._combined_video_source_and_position_ids(
            clean_prefix_latent_frames=clean_prefix_latent_frames,
            total_latent_frames=combined_video_latents.shape[2],
            device=combined_video_latents.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=combined_video_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            source_ids=video_source_ids,
            temporal_position_ids=video_position_ids,
            allow_missing_action_condition=True,
            clean_prefix_latent_frames=clean_prefix_latent_frames,
        )

        video_attention_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            clean_prefix_latent_frames=clean_prefix_latent_frames,
        )

        history_video_latent_valid = self._latent_valid_from_raw_pad(
            raw_is_pad=history_video_is_pad,
            num_latent_frames=clean_prefix_latent_frames,
            batch_size=batch_size,
            device=history_video_latents.device,
        )
        input_video_latent_valid = self._latent_valid_from_raw_pad(
            raw_is_pad=image_is_pad,
            num_latent_frames=input_latents.shape[2],
            batch_size=batch_size,
            device=input_latents.device,
        )
        future_video_latent_valid = input_video_latent_valid[:, 1:]
        combined_video_latent_valid = torch.cat(
            [history_video_latent_valid, future_video_latent_valid],
            dim=1,
        )
        drop_history_video, drop_history_action = self._condition_dropout_masks(
            batch_size=batch_size,
            device=action.device,
        )
        video_history_latent = video_source_ids.eq(self.SOURCE_HISTORY_VIDEO).view(1, -1)
        combined_video_read_latent_valid = combined_video_latent_valid & (
            ~video_history_latent | ~drop_history_video.view(-1, 1)
        )
        combined_video_read_token_valid = self._latent_valid_to_token_valid(
            combined_video_read_latent_valid,
            tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
        )
        video_cache, video_tokens = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=video_attention_mask,
            video_key_valid_mask=combined_video_read_token_valid,
        )
        pred_combined_video = self.video_expert.post_dit(video_tokens, video_pre)
        pred_video = pred_combined_video[:, :, clean_prefix_latent_frames:]
        if pred_video.shape[2] != target_video.shape[2]:
            raise ValueError(
                "Combined v4 video target length mismatch: "
                f"pred={pred_video.shape[2]}, target={target_video.shape[2]}"
            )
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=False,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        history_source_ids = self._action_source_ids(
            batch_size=batch_size,
            seq_len=history_action.shape[1],
            source_id=self.SOURCE_HISTORY_ACTION,
            device=history_action.device,
        )
        history_position_ids = self._action_position_ids(
            seq_len=history_action.shape[1],
            start=0,
            device=history_action.device,
        )
        clean_action_timestep = torch.zeros((batch_size,), device=self.device, dtype=history_action.dtype)
        history_action_pre = self.action_expert.pre_dit(
            action_tokens=history_action,
            timestep=clean_action_timestep,
            context=context,
            context_mask=context_mask,
            source_ids=history_source_ids,
            position_ids=history_position_ids,
        )
        if history_action_is_pad is None:
            history_action_valid = torch.ones(
                (batch_size, history_action.shape[1]),
                dtype=torch.bool,
                device=history_action.device,
            )
        else:
            history_action_valid = ~history_action_is_pad

        history_action_read_valid = history_action_valid & ~drop_history_action.view(-1, 1)

        history_video_token_len = clean_prefix_latent_frames * int(video_pre["meta"]["tokens_per_frame"])
        history_video_cache = self._slice_kv_cache(video_cache, history_video_token_len)
        history_video_read_token_valid = combined_video_read_token_valid[:, :history_video_token_len]
        history_action_attention_mask = torch.ones(
            (history_action.shape[1], history_action.shape[1]),
            dtype=torch.bool,
            device=history_action.device,
        )
        history_action_cache = self.mot.prefill_action_cache(
            action_tokens=history_action_pre["tokens"],
            action_freqs=history_action_pre["freqs"],
            action_t_mod=history_action_pre["t_mod"],
            action_context_payload={
                "context": history_action_pre["context"],
                "mask": history_action_pre["context_mask"],
            },
            action_attention_mask=history_action_attention_mask,
            action_key_valid_mask=history_action_valid,
        )
        condition_cache = self._concat_kv_caches(history_video_cache, history_action_cache)
        condition_key_valid = torch.cat(
            [history_video_read_token_valid, history_action_read_valid],
            dim=1,
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
            source_ids=future_source_ids,
            position_ids=future_position_ids,
        )
        if action_is_pad is None:
            action_key_valid = torch.ones(
                (batch_size, action.shape[1]),
                dtype=torch.bool,
                device=action.device,
            )
        else:
            action_key_valid = ~action_is_pad
        action_tokens = self.mot.forward_action_with_condition_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            condition_kv_cache=condition_cache,
            condition_key_valid_mask=condition_key_valid,
            action_key_valid_mask=action_key_valid,
        )
        pred_action = self.action_expert.post_dit(action_tokens, action_pre)
        loss_action = self._compute_action_loss(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=action_is_pad,
            timestep_action=timestep_action,
        )

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        if self.enable_mem_stage_v4 and (
            inputs["history_video_latents"] is not None or inputs["history_action"] is not None
        ):
            return self._training_loss_v4(inputs)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def _predict_action_noise_with_condition_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        condition_kv_cache: list[dict[str, torch.Tensor]],
        condition_key_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = latents_action.shape[0]
        future_source_ids = self._action_source_ids(
            batch_size=batch_size,
            seq_len=latents_action.shape[1],
            source_id=self.SOURCE_FUTURE_ACTION,
            device=latents_action.device,
        )
        future_position_ids = self._action_position_ids(
            seq_len=latents_action.shape[1],
            start=self.current_timeline_index,
            device=latents_action.device,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
            source_ids=future_source_ids,
            position_ids=future_position_ids,
        )
        action_key_valid = torch.ones(
            (batch_size, latents_action.shape[1]),
            dtype=torch.bool,
            device=latents_action.device,
        )
        action_tokens = self.mot.forward_action_with_condition_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            condition_kv_cache=condition_kv_cache,
            condition_key_valid_mask=condition_key_valid_mask,
            action_key_valid_mask=action_key_valid,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        history_video: Optional[torch.Tensor] = None,
        history_action: Optional[torch.Tensor] = None,
        history_video_is_pad: Optional[torch.Tensor] = None,
        history_action_is_pad: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        history_requested = history_video is not None or history_action is not None
        if history_requested and not self.enable_mem_stage_v4:
            raise ValueError("mem-stage-v4 history inference is only enabled for base FastWAM.")
        use_history_condition = self.enable_mem_stage_v4 and history_requested
        if use_history_condition and (history_video is None or history_action is None):
            raise ValueError("v4 history inference requires both `history_video` and `history_action`.")

        condition_kv_cache = None
        condition_key_valid_mask = None
        attention_mask = None
        video_kv_cache = None
        video_seq_len = 0
        if use_history_condition:
            if history_video.ndim == 4:
                history_video = history_video.unsqueeze(0)
            if history_video.ndim != 5 or history_video.shape[0] != 1 or history_video.shape[1] != 3:
                raise ValueError(
                    "`history_video` must have shape [3,T,H,W] or [1,3,T,H,W], "
                    f"got {tuple(history_video.shape)}"
                )
            if history_video.shape[3] != height or history_video.shape[4] != width:
                raise ValueError(
                    "`history_video` spatial shape must match `input_image`, "
                    f"got {tuple(history_video.shape[3:])} vs {(height, width)}"
                )
            if history_video.shape[2] % 4 != 1:
                raise ValueError(f"`history_video` T must satisfy T % 4 == 1, got {history_video.shape[2]}")
            if history_action.ndim == 2:
                history_action = history_action.unsqueeze(0)
            if history_action.ndim != 3 or history_action.shape[0] != 1:
                raise ValueError(
                    "`history_action` must have shape [T,D] or [1,T,D], "
                    f"got {tuple(history_action.shape)}"
                )
            if history_action.shape[2] != self.action_expert.action_dim:
                raise ValueError(
                    f"`history_action` last dim must be {self.action_expert.action_dim}, got {history_action.shape[2]}"
                )
            if history_video_is_pad is not None:
                if history_video_is_pad.ndim == 1:
                    history_video_is_pad = history_video_is_pad.unsqueeze(0)
                if history_video_is_pad.shape != (1, history_video.shape[2]):
                    raise ValueError(
                        "`history_video_is_pad` shape mismatch: "
                        f"got {tuple(history_video_is_pad.shape)} vs expected {(1, history_video.shape[2])}"
                    )
                history_video_is_pad = history_video_is_pad.to(device=self.device, dtype=torch.bool)
            if history_action_is_pad is not None:
                if history_action_is_pad.ndim == 1:
                    history_action_is_pad = history_action_is_pad.unsqueeze(0)
                if history_action_is_pad.shape != history_action.shape[:2]:
                    raise ValueError(
                        "`history_action_is_pad` shape mismatch: "
                        f"got {tuple(history_action_is_pad.shape)} vs expected {tuple(history_action.shape[:2])}"
                    )
                history_action_is_pad = history_action_is_pad.to(device=self.device, dtype=torch.bool)

            history_video = history_video.to(device=self.device, dtype=self.torch_dtype)
            history_action = history_action.to(device=self.device, dtype=self.torch_dtype)
            history_video_latents = self._encode_video_latents(history_video, tiled=tiled)
            clean_video_timestep = torch.zeros((1,), device=self.device, dtype=history_video_latents.dtype)
            video_source_ids, video_position_ids = self._history_video_source_and_position_ids(
                num_latent_frames=history_video_latents.shape[2],
                device=history_video_latents.device,
            )
            history_video_pre = self.video_expert.pre_dit(
                x=history_video_latents,
                timestep=clean_video_timestep,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
                source_ids=video_source_ids,
                temporal_position_ids=video_position_ids,
                allow_missing_action_condition=True,
            )
            history_video_attention_mask = self.video_expert.build_video_to_video_mask(
                video_seq_len=history_video_pre["tokens"].shape[1],
                video_tokens_per_frame=int(history_video_pre["meta"]["tokens_per_frame"]),
                device=history_video_pre["tokens"].device,
            )
            history_video_latent_valid = self._latent_valid_from_raw_pad(
                raw_is_pad=history_video_is_pad,
                num_latent_frames=history_video_latents.shape[2],
                batch_size=1,
                device=history_video_latents.device,
            )
            history_video_token_valid = self._latent_valid_to_token_valid(
                history_video_latent_valid,
                tokens_per_frame=int(history_video_pre["meta"]["tokens_per_frame"]),
            )
            history_video_cache, _history_video_tokens = self.mot.prefill_video_cache(
                video_tokens=history_video_pre["tokens"],
                video_freqs=history_video_pre["freqs"],
                video_t_mod=history_video_pre["t_mod"],
                video_context_payload={
                    "context": history_video_pre["context"],
                    "mask": history_video_pre["context_mask"],
                },
                video_attention_mask=history_video_attention_mask,
                video_key_valid_mask=history_video_token_valid,
            )

            history_source_ids = self._action_source_ids(
                batch_size=1,
                seq_len=history_action.shape[1],
                source_id=self.SOURCE_HISTORY_ACTION,
                device=history_action.device,
            )
            history_position_ids = self._action_position_ids(
                seq_len=history_action.shape[1],
                start=0,
                device=history_action.device,
            )
            clean_action_timestep = torch.zeros((1,), device=self.device, dtype=history_action.dtype)
            history_action_pre = self.action_expert.pre_dit(
                action_tokens=history_action,
                timestep=clean_action_timestep,
                context=context,
                context_mask=context_mask,
                source_ids=history_source_ids,
                position_ids=history_position_ids,
            )
            if history_action_is_pad is None:
                history_action_valid = torch.ones(
                    (1, history_action.shape[1]),
                    dtype=torch.bool,
                    device=history_action.device,
                )
            else:
                history_action_valid = ~history_action_is_pad
            history_action_attention_mask = torch.ones(
                (history_action.shape[1], history_action.shape[1]),
                dtype=torch.bool,
                device=history_action.device,
            )
            history_action_cache = self.mot.prefill_action_cache(
                action_tokens=history_action_pre["tokens"],
                action_freqs=history_action_pre["freqs"],
                action_t_mod=history_action_pre["t_mod"],
                action_context_payload={
                    "context": history_action_pre["context"],
                    "mask": history_action_pre["context_mask"],
                },
                action_attention_mask=history_action_attention_mask,
                action_key_valid_mask=history_action_valid,
            )
            condition_kv_cache = self._concat_kv_caches(history_video_cache, history_action_cache)
            condition_key_valid_mask = torch.cat([history_video_token_valid, history_action_valid], dim=1)
        else:
            timestep_video = torch.zeros(
                (first_frame_latents.shape[0],),
                dtype=first_frame_latents.dtype,
                device=self.device,
            )
            video_pre = self.video_expert.pre_dit(
                x=first_frame_latents,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            video_seq_len = int(video_pre["tokens"].shape[1])
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=latents_action.shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_pre["tokens"].device,
            )
            video_kv_cache, _video_tokens = self.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            if use_history_condition:
                pred_action_posi = self._predict_action_noise_with_condition_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    condition_kv_cache=condition_kv_cache,
                    condition_key_valid_mask=condition_key_valid_mask,
                )
            else:
                pred_action_posi = self._predict_action_noise_with_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=video_kv_cache,
                    attention_mask=attention_mask,
                    video_seq_len=video_seq_len,
                )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "mem_stage_v4": True,
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            missing_keys, unexpected_keys = self.mot.load_state_dict(payload["mot"], strict=False)
            if payload.get("mem_stage_v4", False) and (missing_keys or unexpected_keys):
                raise ValueError(
                    "mem-stage-v4 checkpoint did not restore MoT weights strictly: "
                    f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                    f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
                )
            if missing_keys or unexpected_keys:
                logger.warning(
                    "Loaded MoT checkpoint with non-strict key diff: missing=%s unexpected=%s",
                    missing_keys[:10],
                    unexpected_keys[:10],
                )
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
