"""Real HDF5/parquet/video checks for the raw terminal-frame contract.

Small test outputs remain in the configured personal TMPDIR for inspection.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("augmentation", ROOT / "scripts/augment_robotwin_lerobot_stereo.py")
augmentation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(augmentation)
spec = importlib.util.spec_from_file_location("renderer", ROOT / "scripts/render_robotwin_stereo.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


class TestTerminalObservation(unittest.TestCase):
    def run_case(self, raw_frames, wrong_action=False):
        import cv2
        root = Path(tempfile.mkdtemp(prefix="stereo-augmentation-"))
        source, replay = root / "official", root / "replay"
        for path in (source / "meta", source / "data", source / "videos", replay / "data"):
            path.mkdir(parents=True)
        features = {f"observation.images.{left}": dict(dtype="video", shape=[8, 8, 3])
                    for left, _ in augmentation.RIGS}
        info = dict(total_episodes=1, chunks_size=1000, fps=15, total_videos=3,
                    features=features, data_path="data/episode_{episode_index:06d}.parquet",
                    video_path="videos/{video_key}/episode_{episode_index:06d}.mp4")
        (source / "meta/info.json").write_text(json.dumps(info))
        (source / "meta/episodes.jsonl").write_text(json.dumps(dict(episode_index=0, length=3)) + "\n")
        (source / "meta/episodes_stats.jsonl").write_text(json.dumps(dict(episode_index=0, stats={})) + "\n")
        qpos = np.arange(raw_frames * 14, dtype=np.float32).reshape(raw_frames, 14) / 100
        targets = qpos[1:4]
        if len(targets) == 2:
            targets = np.concatenate((targets, qpos[-1:]))
        if wrong_action:
            targets = targets.copy()
            targets[0, 0] += 0.1
        table = pa.table({"observation.state": qpos[:3].tolist(), "action": targets.tolist(),
                          "timestamp": (np.arange(3) / 15).tolist()})
        pq.write_table(table, source / "data/episode_000000.parquet")
        image = np.full((8, 8, 3), 64, np.uint8)
        ok, jpeg = cv2.imencode(".jpg", image)
        assert ok
        images = np.array([jpeg.tobytes()] * raw_frames)
        with h5py.File(replay / "data/episode0.hdf5", "w") as f:
            f.create_dataset("joint_action/vector", data=qpos)
            for left, right in augmentation.RIGS:
                f.create_dataset(f"observation/{right.removesuffix('_right')}/rgb", data=images)
                f.create_dataset(f"observation/{right}/rgb", data=images)
                augmentation.encode_right(images[:3], source / f"videos/observation.images.{left}/episode_000000.mp4", 15)
        marker = dict(frames=raw_frames, episode=0, baselines={"head_baseline_m": 0.06}, joint_atol=1e-5)
        (replay / "verified.json").write_text(json.dumps(marker))
        mapping = root / "mapping.json"
        mapping.write_text(json.dumps({"0": str(replay)}))
        return source, root / "augmented", mapping

    def test_terminal_frame_is_omitted_without_shifting_left_or_actions(self):
        source, output, mapping = self.run_case(4)
        augmentation.augment(source, output, mapping)
        self.assertEqual((source / "data/episode_000000.parquet").read_bytes(),
                         (output / "data/episode_000000.parquet").read_bytes())
        provenance = json.loads((output / "meta/stereo_provenance.json").read_text())
        self.assertTrue(provenance["replay_verification"]["0"]["terminal_observation_omitted"])
        for left, _ in augmentation.RIGS:
            path = f"videos/observation.images.{left}/episode_000000.mp4"
            self.assertEqual((source / path).read_bytes(), (output / path).read_bytes())
        stats = json.loads((output / "meta/episodes_stats.jsonl").read_text())["stats"]
        self.assertTrue(all(value["count"] == [3] for value in stats.values()))

    def test_equal_length_terminal_action_replication(self):
        source, output, mapping = self.run_case(3)
        augmentation.augment(source, output, mapping)

    def test_wrong_action_is_rejected(self):
        source, output, mapping = self.run_case(4, wrong_action=True)
        with self.assertRaises(AssertionError):
            augmentation.augment(source, output, mapping)

    def test_more_than_one_extra_frame_is_rejected(self):
        source, output, mapping = self.run_case(5)
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            augmentation.augment(source, output, mapping)

    def test_canonical_hdf5_next_targets_and_jpeg_colors(self):
        import cv2
        _, _, mapping = self.run_case(4)
        replay = Path(json.loads(mapping.read_text())["0"]) / "data/episode0.hdf5"
        canonical = replay.parent / "canonical.hdf5"
        baselines = dict(head_baseline_m=0.06, left_wrist_baseline_m=0.02, right_wrist_baseline_m=0.02)
        with h5py.File(replay, "r+") as target, h5py.File(canonical, "w") as source:
            source.create_dataset("data_format_version", data="v1.0")
            qpos = target["joint_action/vector"][:]
            for key, section in zip(("left_arm_joint_states", "left_ee_joint_states", "right_arm_joint_states", "right_ee_joint_states"),
                                    (slice(0, 6), slice(6, 7), slice(7, 13), slice(13, 14))):
                source.create_dataset(f"state/{key}", data=qpos[:-1, section])
                source.create_dataset(f"action/{key}", data=qpos[1:, section])
            for (left, right, key), camera in zip(renderer.PAIRS, ("cam_head", "cam_left_wrist", "cam_right_wrist")):
                rgb = np.full((8, 8, 3), [32, 80, 160], np.uint8)
                _, jpeg_rgb = cv2.imencode(".jpg", rgb)
                _, jpeg_bgr = cv2.imencode(".jpg", rgb[..., ::-1])
                target[f"observation/{left}/rgb"][:] = np.array([jpeg_rgb.tobytes()] * 4)
                target[f"observation/{right}/rgb"][:] = np.array([jpeg_rgb.tobytes()] * 4)
                source.create_dataset(f"vision/{camera}/colors", data=np.array([jpeg_bgr.tobytes()] * 3))
                for name in (left, right):
                    target.create_dataset(f"observation/{name}/intrinsic_cv", data=np.tile(np.eye(3), (4, 1, 1)))
                    extrinsic = np.tile(np.eye(4)[:3], (4, 1, 1))
                    if name == right:
                        extrinsic[:, 0, 3] = -baselines[key]
                    target.create_dataset(f"observation/{name}/extrinsic_cv", data=extrinsic)
        result = renderer.verify_episode(canonical, replay, baselines)
        self.assertEqual((result["frames"], result["observed_source_frames"]), (4, 3))
        self.assertTrue(result["canonical_source"])
        with h5py.File(canonical, "r+") as source:
            source["action/left_arm_joint_states"][-1, 0] += 0.1
        with self.assertRaises(AssertionError):
            renderer.verify_episode(canonical, replay, baselines)


if __name__ == "__main__":
    unittest.main()
