"""Diagnose whether added stereo cameras change an original left-eye replay.

Experimental branch only. This writes diagnostics, never verified.json, and
cannot produce an accepted stereo training asset. Rendering settings, saved
trajectory and canonical comparison thresholds remain unchanged.
"""
import argparse
import importlib
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("robotwin-root", "source-root", "output-root", "task-config-path",
                 "canonical-archives-root", "stereo-episode"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    root = Path(args.robotwin_root).resolve(strict=True)
    stereo = Path(args.stereo_episode).resolve(strict=True)
    if stereo.name != "episode5" or stereo.parent.parent.name != "open_laptop":
        raise ValueError("This diagnostic only supports open_laptop episode5")
    os.chdir(root)
    sys.path.insert(0, str(root))
    camera = importlib.import_module("envs.camera.camera").Camera
    original_init = camera.__init__

    def mono_init(self, *positional, **options):
        options["stereo"] = {"enabled": False}
        original_init(self, *positional, **options)

    camera.__init__ = mono_init
    script = Path(__file__).resolve().parents[1] / "scripts/render_robotwin_stereo.py"
    spec = importlib.util.spec_from_file_location("mono_replay_diagnostic", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class DiagnosticComplete(Exception):
        pass

    def diagnose(original, replay, baselines, pixel_mae_limit, joint_atol, rgb_cache):
        import cv2
        import h5py
        import numpy as np

        assert pixel_mae_limit == 1 and joint_atol == 1e-5
        keys = ("left_arm_joint_states", "left_ee_joint_states",
                "right_arm_joint_states", "right_ee_joint_states")
        with h5py.File(original, "r") as source, h5py.File(replay, "r") as target:
            states = np.concatenate([source[f"state/{key}"][:] for key in keys], axis=1)
            actions = np.concatenate([source[f"action/{key}"][:] for key in keys], axis=1)
            canonical_q = np.concatenate((states, actions[-1:]))
            mono_q = target["joint_action/vector"][:]
            np.testing.assert_allclose(mono_q, canonical_q, atol=joint_atol, rtol=0)
            assert "head_camera_right" not in target["observation"]
            report = dict(diagnostic_only=True, stereo_asset_accepted=False,
                          canonical_joint_max_abs=float(np.abs(mono_q - canonical_q).max()),
                          frames=len(mono_q), cameras={})
            for left, canonical in (("head_camera", "cam_head"),
                                    ("left_camera", "cam_left_wrist"),
                                    ("right_camera", "cam_right_wrist")):
                images = source[f"vision/{canonical}/colors"]
                assert len(images) + 1 == len(mono_q)
                values, raw_differences = [], []
                for index, encoded in enumerate(images):
                    with (Path(rgb_cache) / f"{index}.pkl").open("rb") as stream:
                        mono_rgb = pickle.load(stream)["observation"][left]["rgb"]
                    with (stereo / ".cache/episode5" / f"{index}.pkl").open("rb") as stream:
                        stereo_rgb = pickle.load(stream)["observation"][left]["rgb"]
                    canonical_rgb = cv2.imdecode(np.frombuffer(bytes(encoded), np.uint8),
                                                cv2.IMREAD_COLOR)[..., ::-1]
                    ok, jpeg = cv2.imencode(".jpg", mono_rgb[..., ::-1])
                    assert ok
                    mono_jpeg = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)[..., ::-1]
                    assert canonical_rgb.shape == mono_rgb.shape == stereo_rgb.shape
                    values.append(float(np.abs(canonical_rgb.astype(np.float32) - mono_jpeg).mean()))
                    raw_differences.append(float(np.abs(mono_rgb.astype(np.float32) - stereo_rgb).max()))
                report["cameras"][left] = dict(
                    canonical_max_mae=max(values),
                    first_above_one=next((i for i, value in enumerate(values) if value > 1), None),
                    mono_vs_stereo_raw_max_abs=max(raw_differences),
                    sampled_mae={str(i): values[i] for i in (0, 20, 22, 30, 60)})
        (Path(replay).parent.parent / "mono-replay-diagnostics.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        # Exit before production replay can write a stereo acceptance marker.
        raise DiagnosticComplete()

    module.verify_episode = diagnose
    options = argparse.Namespace(**vars(args), tasks=["open_laptop"], replan_missing=False,
                                 head_baseline_m=0.06, left_wrist_baseline_m=0.02,
                                 right_wrist_baseline_m=0.02, num_shards=50, shard_index=5,
                                 pixel_mae_limit=1.0, joint_atol=1e-5)
    try:
        module.replay(options)
    except DiagnosticComplete:
        pass


if __name__ == "__main__":
    main()
