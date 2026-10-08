"""Add verified right-eye videos to a copy of the official FastWAM LeRobot data.

The explicit episode map is a JSON object mapping every LeRobot episode index
to a verified replay episode directory. Original parquet/action/state/text and
left videos are copied unchanged. No trajectory conversion or action shifting.
"""
import argparse
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor
import io
import zipfile
from contextlib import ExitStack
import shutil
from fractions import Fraction
from pathlib import Path

import numpy as np

RIGS = (("cam_high", "head_camera_right"),
        ("cam_left_wrist", "left_camera_right"),
        ("cam_right_wrist", "right_camera_right"))


def encode_right(images, destination, fps):
    import av
    import cv2
    destination.parent.mkdir(parents=True, exist_ok=True)
    channel_min = np.full(3, np.inf)
    channel_max = np.full(3, -np.inf)
    channel_sum = np.zeros(3)
    channel_square_sum = np.zeros(3)
    count = 0
    with av.open(str(destination), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(str(fps)))
        stream.pix_fmt = "yuv444p"
        stream.thread_count = 1
        stream.options = {"crf": "0", "preset": "fast"}
        shape = None
        for index, jpeg in enumerate(images):
            # RoboTwin writes RGB arrays directly using cv2.imencode; imdecode
            # returns that same channel order. A BGR->RGB conversion would invert it.
            rgb = cv2.imdecode(np.frombuffer(bytes(jpeg), np.uint8), cv2.IMREAD_COLOR)
            if rgb is None or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError(f"Invalid right-eye JPEG at frame {index}")
            if shape is None:
                shape = rgb.shape
                stream.height, stream.width = shape[:2]
            if rgb.shape != shape:
                raise ValueError("Right-eye resolution changed within episode")
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 1) / Fraction(str(fps))
            for packet in stream.encode(frame):
                container.mux(packet)
            # Exact full-frame RGB statistics, normalized to [0,1].
            pixels = rgb.reshape(-1, 3).astype(np.float64) / 255
            channel_min = np.minimum(channel_min, pixels.min(0))
            channel_max = np.maximum(channel_max, pixels.max(0))
            channel_sum += pixels.sum(0)
            channel_square_sum += (pixels * pixels).sum(0)
            count += len(pixels)
        for packet in stream.encode():
            container.mux(packet)
    if not count:
        raise ValueError("Empty right-eye video")
    mean = channel_sum / count
    variance = np.maximum(channel_square_sum / count - mean * mean, 0)
    stats = {key: value.reshape(3, 1, 1).tolist() for key, value in
             dict(min=channel_min, max=channel_max, mean=mean, std=np.sqrt(variance)).items()}
    stats["count"] = [len(images)]
    # Verify the encoded stream frame count before publishing its metadata.
    with av.open(str(destination)) as container:
        if sum(1 for _ in container.decode(video=0)) != len(images):
            raise ValueError(f"Encoded frame count mismatch: {destination}")
    return stats, shape


def verify_left_video(video, images, mae_limit, canonical_bgr=False):
    import av
    import cv2
    maximum = 0.0
    with av.open(str(video)) as container:
        decoded = iter(container.decode(video=0))
        for index, jpeg in enumerate(images):
            try:
                rgb = next(decoded).to_ndarray(format="rgb24")
            except StopIteration as error:
                raise ValueError(f"Official left video is too short: {video}") from error
            raw = cv2.imdecode(np.frombuffer(bytes(jpeg), np.uint8), cv2.IMREAD_COLOR)
            if raw is None or raw.shape != rgb.shape:
                raise ValueError(f"Official/raw left image shape mismatch: {video}/{index}")
            if canonical_bgr:
                raw = raw[..., ::-1]
            mae = float(np.abs(raw.astype(np.float32) - rgb).mean())
            maximum = max(maximum, mae)
            if mae > mae_limit:
                raise ValueError(f"Official/raw left frame or color mismatch: {video}/{index}, MAE={mae}")
        if next(decoded, None) is not None:
            raise ValueError(f"Official left video has extra frames: {video}")
    return maximum


def _augment_episode(job):
    import h5py
    import pyarrow.parquet as pq
    source, destination, info, episode, statistics, folder, marker, left_video_mae_limit = job
    index = episode["episode_index"]
    chunk = index // info["chunks_size"]
    raw = folder / "data" / f"episode{marker['episode']}.hdf5"
    table = pq.read_table(source / info["data_path"].format(episode_chunk=chunk, episode_index=index), use_threads=False)
    state = np.array(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.array(table["action"].to_pylist(), dtype=np.float32)
    timestamp = np.array(table["timestamp"].to_pylist(), dtype=np.float64)
    np.testing.assert_allclose(timestamp, np.arange(episode["length"]) / info["fps"], atol=1e-4, rtol=0)
    with ExitStack() as stack:
        data = stack.enter_context(h5py.File(raw, "r"))
        canonical = None
        if "canonical_reference" in marker:
            reference = marker["canonical_reference"]
            if not marker.get("canonical_source") or marker["pixel_mae_limit"] > 1.0:
                raise ValueError(f"Missing strict canonical replay verification: {index}")
            archive = stack.enter_context(zipfile.ZipFile(reference["archive"]))
            canonical = stack.enter_context(h5py.File(io.BytesIO(archive.read(reference["member"])), "r"))
            keys = ("left_arm_joint_states", "left_ee_joint_states", "right_arm_joint_states", "right_ee_joint_states")
            np.testing.assert_allclose(state, np.concatenate([canonical[f"state/{key}"][:] for key in keys], 1), atol=1e-5, rtol=0)
            np.testing.assert_allclose(action, np.concatenate([canonical[f"action/{key}"][:] for key in keys], 1), atol=1e-5, rtol=0)
        qpos = data["joint_action/vector"][:]
        frames = episode["length"]
        if qpos.shape != (marker["frames"], 14) or state.shape != (frames, 14) or action.shape != state.shape:
            raise ValueError(f"Official/replayed state shape mismatch: {index}")
        if not np.isfinite(state).all() or not np.isfinite(action).all() or not np.isfinite(qpos).all():
            raise ValueError(f"Nonfinite official/replayed states: {index}")
        np.testing.assert_allclose(state, qpos[:frames], atol=marker["joint_atol"], rtol=0)
        # LeRobot stores observation q[t] and its target q[t+1]. Raw
        # RoboTwin includes that terminal observation; it is not a new
        # training sample. Equal-length recordings replicate the last target.
        targets = qpos[1:frames + 1]
        if len(targets) == frames - 1:
            targets = np.concatenate((targets, qpos[-1:]))
        np.testing.assert_allclose(action, targets, atol=marker["joint_atol"], rtol=0)
        marker["official_frames"] = frames
        marker["terminal_observation_omitted"] = len(qpos) == frames + 1
        marker["official_left_video_max_mae"] = {}
        for left, right in RIGS:
            original_camera = right.removesuffix("_right")
            original_key = f"observation.images.{left}"
            original_video = source / info["video_path"].format(
                episode_chunk=chunk, episode_index=index, video_key=original_key)
            images = data[f"observation/{original_camera}/rgb"][:frames]
            if canonical is not None:
                camera = {"cam_high": "cam_head", "cam_left_wrist": "cam_left_wrist", "cam_right_wrist": "cam_right_wrist"}[left]
                images = canonical[f"vision/{camera}/colors"][:]
                if len(images) != frames:
                    raise ValueError(f"Canonical image count mismatch: {index}/{left}")
            marker["official_left_video_max_mae"][left] = verify_left_video(
                original_video, images, left_video_mae_limit, canonical_bgr=canonical is not None)
            key = f"observation.images.{left}_right"
            video = destination / info["video_path"].format(episode_chunk=chunk, episode_index=index, video_key=key)
            if video.exists():
                raise FileExistsError(video)
            stats, shape = encode_right(data[f"observation/{right}/rgb"][:frames], video, info["fps"])
            statistics["stats"][key] = stats
            if key not in info["features"]:
                feature = copy.deepcopy(info["features"][f"observation.images.{left}"])
                if tuple(feature["shape"]) != shape:
                    raise ValueError(f"Official/replayed image shape mismatch: {key}")
                feature["info"] = {"video.fps": info["fps"], "video.height": shape[0],
                                   "video.width": shape[1], "video.channels": 3, "video.codec": "h264",
                                   "video.pix_fmt": "yuv444p", "video.is_depth_map": False, "has_audio": False}
                info["features"][key] = feature
    features = {f"observation.images.{left}_right": info["features"][f"observation.images.{left}_right"]
                for left, _ in RIGS}
    return index, statistics, marker, features


def augment(source, destination, episode_map, left_video_mae_limit=3.0):
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be disjoint")
    if destination.exists():
        raise FileExistsError(destination)
    if not np.isfinite(left_video_mae_limit) or left_video_mae_limit < 0:
        raise ValueError("Invalid official/raw left video alignment threshold")
    mapping_path = Path(episode_map).resolve(strict=True)
    mapping = json.loads(mapping_path.read_text())
    info = json.loads((source / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines()]
    indices = [item["episode_index"] for item in episodes]
    if not indices or indices != list(range(info["total_episodes"])) or set(mapping) != {str(i) for i in indices}:
        raise ValueError("Episode map must cover every official clean episode exactly once")
    statistics = [json.loads(line) for line in (source / "meta/episodes_stats.jsonl").read_text().splitlines()]
    stats_by_episode = {item["episode_index"]: item for item in statistics}
    if set(stats_by_episode) != set(indices):
        raise ValueError("Incomplete official episode statistics")
    resolved = {}
    for episode in episodes:
        index = episode["episode_index"]
        folder = Path(mapping[str(index)])
        folder = (mapping_path.parent / folder).resolve(strict=True) if not folder.is_absolute() else folder.resolve(strict=True)
        marker = json.loads((folder / "verified.json").read_text())
        if marker["frames"] not in (episode["length"], episode["length"] + 1):
            raise ValueError(f"Replay/official episode length mismatch: {index}")
        resolved[index] = folder, marker
    if len({str(folder) for folder, _ in resolved.values()}) != len(indices):
        raise ValueError("Multiple official episodes map to the same replay")
    baselines = resolved[indices[0]][1]["baselines"]
    if any(marker["baselines"] != baselines for _, marker in resolved.values()):
        raise ValueError("Replay shards used inconsistent stereo baselines")
    for left, _ in RIGS:
        feature = info["features"][f"observation.images.{left}"]
        if feature["dtype"] != "video":
            raise ValueError("Expected official video-backed LeRobot cameras")
    destination.mkdir(parents=True)
    # Preserve official split inputs, prompts, parquet state/actions and left video bytes.
    for name in ("meta", "data", "videos"):
        shutil.copytree(source / name, destination / name)
    if (source / "annotations").is_dir():
        shutil.copytree(source / "annotations", destination / "annotations")
    worker_info = copy.deepcopy(info)
    jobs = [(source, destination, worker_info, episode, stats_by_episode[episode["episode_index"]],
             *resolved[episode["episode_index"]], left_video_mae_limit) for episode in episodes]
    executor = ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1, len(episodes)))
    try:
        for index, statistics, marker, features in executor.map(_augment_episode, jobs):
            stats_by_episode[index] = statistics
            resolved[index] = resolved[index][0], marker
            info["features"].update(features)
            print(f"Augmented official episode {index}/{len(episodes)}", flush=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    info["total_videos"] += 3 * len(episodes)
    (destination / "meta/info.json").write_text(json.dumps(info, indent=2))
    (destination / "meta/episodes_stats.jsonl").write_text(
        "".join(json.dumps(stats_by_episode[i]) + "\n" for i in indices))
    # Legacy stats.json exists in some releases; append the new eyes only.
    legacy_path = destination / "meta/stats.json"
    if legacy_path.exists():
        legacy = json.loads(legacy_path.read_text())
        for left, _ in RIGS:
            key = f"observation.images.{left}_right"
            entries = [stats_by_episode[i]["stats"][key] for i in indices]
            weights = np.array([e["count"][0] for e in entries], dtype=np.float64)
            means = np.array([e["mean"] for e in entries])
            deviations = np.array([e["std"] for e in entries])
            mean = np.average(means, axis=0, weights=weights)
            std = np.sqrt(np.average(deviations**2 + (means - mean)**2, axis=0, weights=weights))
            legacy[key] = dict(min=np.min([e["min"] for e in entries], axis=0).tolist(),
                               max=np.max([e["max"] for e in entries], axis=0).tolist(),
                               mean=mean.tolist(), std=std.tolist(), count=[int(weights.sum())])
        legacy_path.write_text(json.dumps(legacy, indent=2))
    (destination / "meta/stereo_provenance.json").write_text(json.dumps(
        dict(source=str(source), episode_map={str(i): str(resolved[i][0]) for i in indices},
             replay_verification={str(i): resolved[i][1] for i in indices},
             official_left_video_mae_limit=left_video_mae_limit), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Official clean LeRobot root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--episode-map", required=True)
    parser.add_argument("--left-video-mae-limit", type=float, default=3.0)
    args = parser.parse_args()
    augment(args.source, args.output, args.episode_map, args.left_video_mae_limit)
