"""Use the pinned OpenWAM readers without reimplementing their data semantics."""
from pathlib import Path
import subprocess

import numpy as np
import torch
from omegaconf import OmegaConf

from .lerobot.robot_video_dataset import RobotVideoDataset


OPENWAM_SHA = "4d7f642b841913ea3c176010b692b1de3d5ef180"


class OpenWAMDataset(RobotVideoDataset):
    """Adapt only the sample container; inherit FastWAM's T5 cache reader.

    No random-sample recovery, renormalization, camera remapping, or action
    conversion is performed here. OpenWAM owns all of those data semantics.
    """

    def __init__(self, benchmark, dataset_dirs, text_embedding_cache_dir, processor,
                 context_len=512):
        if benchmark not in ("libero", "robodojo"):
            raise ValueError(f"Unsupported aligned FastWAM benchmark: {benchmark}")
        if len(dataset_dirs) != 1:
            raise ValueError("Each benchmark must bind exactly one canonical data root")
        if processor.action_output_dim != 80 or processor.proprio_output_dim != 80:
            raise ValueError("The OpenWAM reader requires unified action/state width 80")
        # Import lazily so existing FastWAM tasks do not depend on OpenWAM.
        import openwam
        from openwam.dataloader import build_dataset

        source_root = Path(openwam.__file__).resolve().parents[1]
        sha = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip()
        if sha != OPENWAM_SHA:
            raise ValueError(f"OpenWAM source revision mismatch: {sha}")
        dirty = subprocess.check_output(
            ["git", "-C", str(source_root), "status", "--porcelain", "--",
             "openwam/dataloader", f"configs/dataloader/{benchmark}.yaml"], text=True
        ).strip()
        if dirty:
            raise ValueError(f"OpenWAM reader/config has local changes: {dirty}")
        config = OmegaConf.load(source_root / "configs" / "dataloader" / f"{benchmark}.yaml")
        if config.type != benchmark or not config.unify_action or config.action_mode != "eef":
            raise ValueError("Unexpected OpenWAM benchmark/action contract")
        config.dataset_dir = str(Path(dataset_dirs[0]).resolve(strict=True))
        self.reader = build_dataset(config, split="train")
        self.text_embedding_cache_dir = str(text_embedding_cache_dir)
        self.context_len = int(context_len)
        if self.context_len != 512:
            raise ValueError("Aligned OpenWAM text context length must be 512")

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        sample = self.reader[index]
        action, mask = sample["action"], sample["action_mask"]
        if action.shape != (32, 80) or mask.shape != action.shape or mask.dtype != torch.bool:
            raise ValueError("Expected normalized action and bool validity mask [32,80]")
        proprio = sample["proprio"]
        if proprio.shape != (1, 80):
            raise ValueError("Expected normalized proprio [1,80]")
        frames = np.stack([np.asarray(frame) for frame in sample["video"]])
        if frames.shape != (9, 384, 320, 3) or frames.dtype != np.uint8:
            raise ValueError("Expected nine RGB uint8 OpenWAM canvases at 384x320")
        video_mask = sample["video_mask"]
        if video_mask.shape != (9,) or video_mask.dtype != torch.bool:
            raise ValueError("Expected nine bool video validity entries")
        if not torch.isfinite(action).all() or not torch.isfinite(proprio).all():
            raise ValueError("Non-finite normalized action/proprio")
        context, context_mask = self._get_cached_text_context(sample["prompt"])
        context = context.clone()
        context[~context_mask] = 0
        return {
            "video": torch.from_numpy(frames.copy()).permute(3, 0, 1, 2).float() / 127.5 - 1,
            "action": action,
            "action_is_pad": ~mask,
            "image_is_pad": ~video_mask,
            "proprio": proprio,
            "prompt": sample["prompt"],
            "context": context,
            "context_mask": torch.ones_like(context_mask),
        }
