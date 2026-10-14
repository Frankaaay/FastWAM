"""Single-environment smoke bridge: pinned RoboDojo protocol to EEF20 policy WS."""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

import numpy as np


def require_revision(root, revision):
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=root, text=True)
    if actual != revision or dirty.strip():
        raise RuntimeError(f"Source contract mismatch: {root}: {actual}, dirty={dirty!r}")


class Bridge:
    def __init__(self, url, adapter):
        from benchmarks.utils import WSPolicyClient
        from openwam.dataloader.robodojo_contract import arx_x5_calibration
        from openwam.dataloader.transforms.multiview import format_prompt_for_inference
        from openwam.dataloader.utils import poses

        self._poses = poses
        self._calibration = arx_x5_calibration()
        self._format_prompt = format_prompt_for_inference
        self.default_instruction = ""
        # Bind only canonical conversion methods. Never initialize a dummy policy.
        self._state_to_eef20 = adapter._state_to_eef20.__get__(self)
        self._encode_obs = adapter._encode_obs.__get__(self)
        self._to_native = adapter._eef20_chunk_to_native.__get__(self)
        self._client = WSPolicyClient(url, timeout=300)
        pong = self._client.ping()
        if pong.get("type") != "pong" or pong.get("representation") != "eef":
            raise RuntimeError(f"Expected an EEF policy service, got {pong}")
        self._payload = None
        self._env_idx = None
        self.reset()

    def update_obs(self, obs):
        from benchmarks.utils import encode_numpy_b64
        from XPolicyLab.policy.OpenWAM.model import _decode_instruction

        if not _decode_instruction(obs.get("instruction", obs.get("task_instruction")), ""):
            raise ValueError("RoboDojo observation must supply the task instruction")
        env_idx = int(obs.get("env_idx", 0))
        if self._env_idx is not None and self._env_idx != env_idx:
            raise ValueError("Reset is required before switching simulator environments")
        self._env_idx = env_idx
        payload = self._encode_obs(obs)
        payload["images"] = {
            key: encode_numpy_b64(np.asarray(value)) for key, value in payload["images"].items()
        }
        self._payload = payload

    def update_obs_batch(self, observations):
        if not isinstance(observations, list) or len(observations) != 1:
            raise ValueError("This smoke bridge requires exactly one simulator environment")
        self.update_obs(observations[0])

    def get_action(self):
        if self._payload is None:
            raise RuntimeError("update_obs must precede get_action")
        # Both services use full 32-action chunks. No retry on ambiguous action calls.
        chunk = np.asarray([
            self._client.predict_once(self._payload)["action"] for _ in range(32)
        ], dtype=np.float32)
        if chunk.shape != (32, 20) or not np.isfinite(chunk).all():
            raise ValueError(f"Expected finite raw EEF20 chunk, got {chunk.shape}")
        actions = self._to_native(chunk)
        if not all(np.isfinite(value).all() for action in actions for value in action.values()):
            raise ValueError("Non-finite simulator action after canonical conversion")
        print("ROBODOJO_POLICY_CHUNK_OK", self._env_idx, chunk.shape, flush=True)
        return actions

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is not None and list(np.asarray(env_idx_list).reshape(-1)) != [self._env_idx]:
            raise ValueError("Requested environment does not match the stored observation")
        return [self.get_action()]

    def reset(self):
        if self._client.reset().get("type") != "reset_ack":
            raise RuntimeError("Policy reset failed")
        self._payload = None
        self._env_idx = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo-root", type=Path, required=True)
    parser.add_argument("--openwam-root", type=Path, required=True)
    parser.add_argument("--policy-url", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    xpolicy = args.robodojo_root / "XPolicyLab"
    require_revision(xpolicy, "53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2")
    require_revision(args.openwam_root, "4d7f642b841913ea3c176010b692b1de3d5ef180")
    sys.path[:0] = [str(args.openwam_root), str(args.robodojo_root), str(xpolicy)]
    from XPolicyLab.policy.OpenWAM.model import Model
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    bridge = Bridge(args.policy_url, Model)
    server = PolicyServer(bridge, PolicyServerConfig(host="127.0.0.1", port=args.port))
    try:
        asyncio.run(server.serve_forever())
    finally:
        bridge._client.close()


if __name__ == "__main__":
    main()
