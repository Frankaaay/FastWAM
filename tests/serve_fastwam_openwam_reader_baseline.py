"""Legacy FastWAM adaptation using an OpenWAM reader, not an OpenWAM model.

Run in the FastWAM environment with the pinned OpenWAM checkout on PYTHONPATH.
Not accepted as a provenance-aligned native benchmark baseline.
Only transport is implemented here; OpenWAM owns layout, normalization and
unified action mapping, and its benchmark clients own simulator conversion.
"""
import argparse
import asyncio
import base64
from collections import deque
import hashlib
import io
import json
from pathlib import Path
import time

from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch

from fastwam.datasets.openwam_dataset import OPENWAM_SHA
from fastwam.utils.config_resolvers import register_default_resolvers


class BaselinePolicy:
    def __init__(self, training_dir, checkpoint, benchmark, denoise_steps):
        register_default_resolvers()
        root = Path(training_dir).resolve(strict=True)
        checkpoint = Path(checkpoint).resolve(strict=True)
        if not checkpoint.is_relative_to(root):
            raise ValueError("Checkpoint must belong to the selected training directory")
        cfg = OmegaConf.load(root / "config.yaml")
        if (cfg.data.train._target_ != "fastwam.datasets.openwam_dataset.OpenWAMDataset"
                or cfg.data.train.benchmark != benchmark):
            raise ValueError("Saved dataset/benchmark does not match the requested server")
        if cfg.model._target_ != "fastwam.runtime.create_fastwam":
            raise ValueError("This smoke server only supports the original FastWAM model")
        if denoise_steps <= 0:
            raise ValueError("denoise_steps must be positive")
        self.dataset = instantiate(cfg.data.train)  # checks pinned reader SHA and clean config
        import openwam
        from openwam.dataloader.transforms.normalize import Normalizer, YAML_TO_NORM_MODE, load_mode_stats
        from openwam.dataloader.utils.unify_action import parse_unify_spec

        source = Path(openwam.__file__).resolve().parents[1]
        self.dl = OmegaConf.load(source / "configs/dataloader" / f"{benchmark}.yaml")
        self.dl.dataset_dir = cfg.data.train.dataset_dirs[0]
        stats_source = Path(self.dataset.reader.normalization_stats_path).resolve(strict=True)
        artifact = root / "canonical_deployment"
        artifact.mkdir(exist_ok=True)
        stats_bytes = stats_source.read_bytes()
        stats_path = artifact / "normalization_stats.npy"
        if stats_path.exists():
            if stats_path.read_bytes() != stats_bytes:
                raise ValueError("Normalization statistics changed since deployment binding")
        else:
            with stats_path.open("xb") as handle:
                handle.write(stats_bytes)
        stats = load_mode_stats(str(stats_path), "eef")
        if stats is None:
            raise ValueError("Canonical EEF normalization statistics are missing")
        self.normalizer = Normalizer(mode=YAML_TO_NORM_MODE[self.dl.normalize_mode], stats=stats)
        self.indices = parse_unify_spec(self.dl.unify_action_map, 80)
        self.raw_dim = 10 if benchmark == "libero" else 20
        if len(self.indices) != self.raw_dim:
            raise ValueError("Unexpected raw EEF action width")
        self.denoise_steps = denoise_steps
        self.actions = deque()
        self.step = 0
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if int(payload.get("step", 0)) < 2 or "mot" not in payload or "proprio_encoder" not in payload:
            raise ValueError("Expected a completed training-smoke checkpoint with proprio weights")
        self.model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda:0")
        self.model.mot.load_state_dict(payload["mot"], strict=True)
        self.model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        self.model.eval()
        self.contract = {"representation": "eef", "benchmark": benchmark, "action_dim": self.raw_dim}
        (artifact / "binding.json").write_text(json.dumps({
            "checkpoint": str(checkpoint), "step": int(payload["step"]),
            "openwam_sha": OPENWAM_SHA, "benchmark": benchmark,
            "stats_source": str(stats_source), "stats_sha256": hashlib.sha256(stats_bytes).hexdigest(),
            "denoise_steps": denoise_steps, "action_horizon": 32,
            "inference_horizon": 32, "dataloader": OmegaConf.to_container(self.dl),
        }, indent=2))
        print("STRICT_CHECKPOINT_LOADED", checkpoint, payload["step"], flush=True)

    def reset(self):
        self.actions.clear()
        self.step = 0

    @torch.inference_mode()
    def act(self, obs):
        from openwam.dataloader.transforms.multiview import assemble_multiview_layout
        from openwam.dataloader.utils.unify_action import map_to_unify, unmap_from_unify

        state = np.asarray(obs["state"], dtype=np.float32)
        if state.shape != (self.raw_dim,) or not np.isfinite(state).all():
            raise ValueError("Invalid raw EEF proprio shape or values")
        prompt = obs["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty canonical prompt is required")
        if not self.actions:
            cameras = {}
            for key, camera in zip(("head_camera", "left_wrist_camera", "right_wrist_camera"), self.dl.camera_layout):
                value = obs["images"].get(key)
                if camera is None:
                    if value is not None:
                        raise ValueError(f"Unexpected camera in unused slot {key}")
                    continue
                if value is None:
                    raise ValueError(f"Missing required camera {key}")
                cameras[camera] = Image.open(io.BytesIO(base64.b64decode(value, validate=True))).convert("RGB")
            canvas = assemble_multiview_layout(cameras, list(self.dl.camera_layout), 384, 320)
            image = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float() / 127.5 - 1
            state, _ = map_to_unify(self.normalizer.normalize(state), self.indices, 80)
            context, mask = self.dataset._get_cached_text_context(prompt)
            context = context.clone()
            context[~mask] = 0
            result = self.model.infer_action(
                prompt=None, input_image=image, action_horizon=32,
                proprio=torch.from_numpy(state), context=context,
                context_mask=torch.ones_like(mask), num_inference_steps=self.denoise_steps,
            )["action"].numpy()
            if result.shape != (32, 80) or not np.isfinite(result).all():
                raise ValueError("Invalid model action chunk")
            actions = self.normalizer.unnormalize(unmap_from_unify(result, self.indices))
            if actions.shape != (32, self.raw_dim) or not np.isfinite(actions).all():
                raise ValueError("Invalid decoded EEF action chunk")
            self.actions.extend(actions)
            print("INFERENCE_CHUNK_OK", self.step, actions.shape, flush=True)
        self.step += 1
        return self.actions.popleft().tolist()


async def serve(policy, port):
    from websockets.asyncio.server import serve as websocket_serve

    active = False

    async def handler(connection):
        nonlocal active
        if active:
            await connection.close(code=1013, reason="Single smoke client only")
            return
        active = True
        policy.reset()
        try:
            async for raw in connection:
                try:
                    message = json.loads(raw)
                    kind = message["type"]
                    if kind == "ping":
                        response = {"type": "pong", **policy.contract}
                    elif kind == "reset":
                        policy.reset()
                        response = {"type": "reset_ack"}
                    elif kind == "obs":
                        start = time.monotonic()
                        action = policy.act(message)
                        response = {"type": "action", "action": action, "step": policy.step,
                                    "latency_ms": 1000 * (time.monotonic() - start)}
                    else:
                        raise ValueError(f"Unsupported request type: {kind}")
                    await connection.send(json.dumps(response))
                except Exception as exc:
                    await connection.send(json.dumps({"type": "error", "code": "baseline_contract_error", "message": str(exc)}))
                    raise
        finally:
            active = False
            policy.reset()

    async with websocket_serve(handler, "127.0.0.1", port, max_size=32 * 1024 * 1024):
        print("BASELINE_SERVER_READY", port, policy.contract, flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark", choices=["libero", "robodojo"], required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--denoise-steps", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(serve(BaselinePolicy(args.training_dir, args.checkpoint, args.benchmark, args.denoise_steps), args.port))
