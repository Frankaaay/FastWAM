"""Publish same-replay left/right videos with official actions, states and text.

The episode map selects completed replays by their original LeRobot episode ID.
Local IDs are contiguous; provenance retains IDs for the original train/val split.
"""
import argparse
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
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


def _augment_episode(job):
    import h5py
    import pyarrow.parquet as pq
    source, destination, info, episode, statistics, folder, marker, output_index, output_offset = job
    index = episode["episode_index"]
    chunk = index // info["chunks_size"]
    raw = folder / "data" / f"episode{marker['episode']}.hdf5"
    table = pq.read_table(source / info["data_path"].format(episode_chunk=chunk, episode_index=index), use_threads=False)
    state = np.array(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.array(table["action"].to_pylist(), dtype=np.float32)
    timestamp = np.array(table["timestamp"].to_pylist(), dtype=np.float64)
    np.testing.assert_allclose(timestamp, np.arange(episode["length"]) / info["fps"], atol=1e-4, rtol=0)
    with h5py.File(raw, "r") as data:
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
        marker["pairing"] = "fresh-left+fresh-right"
        marker["canonical_episode"] = index
        for left, right in RIGS:
            original_camera = right.removesuffix("_right")
            kl, kr = (data[f"observation/{camera}/intrinsic_cv"][:] for camera in (original_camera, right))
            el, er = (data[f"observation/{camera}/extrinsic_cv"][:] for camera in (original_camera, right))
            if kl.shape != (len(qpos), 3, 3) or kr.shape != kl.shape or el.shape != (len(qpos), 3, 4) or er.shape != el.shape:
                raise ValueError(f"Invalid stereo calibration shape: {index}/{left}")
            if any(not np.isfinite(matrix).all() for matrix in (kl, kr, el, er)):
                raise ValueError(f"Nonfinite stereo calibration: {index}/{left}")
            np.testing.assert_allclose(kl, kr, atol=1e-6, rtol=0)
            rotation = el[:, :3, :3]
            np.testing.assert_allclose(rotation, er[:, :3, :3], atol=1e-5, rtol=0)
            np.testing.assert_allclose(rotation @ rotation.transpose(0, 2, 1), np.broadcast_to(np.eye(3), rotation.shape), atol=1e-5, rtol=0)
            np.testing.assert_allclose(np.linalg.det(rotation), 1, atol=1e-5, rtol=0)
            center_l = -np.einsum("nji,nj->ni", rotation, el[:, :3, 3])
            center_r = -np.einsum("nji,nj->ni", rotation, er[:, :3, 3])
            offset = np.einsum("nij,nj->ni", rotation, center_r - center_l)
            expected = np.zeros_like(offset)
            baseline_key = {"cam_high": "head_baseline_m", "cam_left_wrist": "left_wrist_baseline_m", "cam_right_wrist": "right_wrist_baseline_m"}[left]
            expected[:, 0] = marker["baselines"][baseline_key]
            np.testing.assert_allclose(offset, expected, atol=1e-5, rtol=0)
            for suffix, camera in (("", original_camera), ("_right", right)):
                images = data[f"observation/{camera}/rgb"]
                if len(images) != len(qpos):
                    raise ValueError(f"Stereo image count mismatch: {index}/{camera}")
                key = f"observation.images.{left}{suffix}"
                video = destination / info["video_path"].format(episode_chunk=output_index // info["chunks_size"], episode_index=output_index, video_key=key)
                if video.exists():
                    raise FileExistsError(video)
                stats, shape = encode_right(images[:frames], video, info["fps"])
                statistics["stats"][key] = stats
                feature = copy.deepcopy(info["features"][f"observation.images.{left}"])
                if tuple(feature["shape"]) != shape:
                    raise ValueError(f"Official/replayed image shape mismatch: {key}")
                feature["info"] = {"video.fps": info["fps"], "video.height": shape[0],
                                   "video.width": shape[1], "video.channels": 3, "video.codec": "h264",
                                   "video.pix_fmt": "yuv444p", "video.is_depth_map": False, "has_audio": False}
                info["features"][key] = feature
    # Only dataset indexing changes. Action/state/timestamps/tasks stay exact.
    for column, values in (("episode_index", np.full(frames, output_index)), ("index", np.arange(output_offset, output_offset + frames))):
        if column in table.column_names:
            import pyarrow as pa
            field = table.schema.field(column)
            table = table.set_column(table.schema.get_field_index(column), field, pa.array(values, type=field.type))
    parquet = destination / info["data_path"].format(episode_chunk=output_index // info["chunks_size"], episode_index=output_index)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet)
    statistics["episode_index"] = output_index
    features = {f"observation.images.{left}{suffix}": info["features"][f"observation.images.{left}{suffix}"]
                for left, _ in RIGS for suffix in ("", "_right")}
    return output_index, statistics, marker, features


def augment(source, destination, episode_map):
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be disjoint")
    if destination.exists():
        raise FileExistsError(destination)
    mapping_path = Path(episode_map).resolve(strict=True)
    mapping = json.loads(mapping_path.read_text())
    info = json.loads((source / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines()]
    indices = [item["episode_index"] for item in episodes]
    if not indices or indices != list(range(info["total_episodes"])) or not mapping or not set(mapping).issubset({str(i) for i in indices}):
        raise ValueError("Episode map must select unique official clean episode IDs")
    statistics = [json.loads(line) for line in (source / "meta/episodes_stats.jsonl").read_text().splitlines()]
    stats_by_episode = {item["episode_index"]: item for item in statistics}
    if set(stats_by_episode) != set(indices):
        raise ValueError("Incomplete official episode statistics")
    episodes = [episode for episode in episodes if str(episode["episode_index"]) in mapping]
    indices = [episode["episode_index"] for episode in episodes]
    source_total_episodes = info["total_episodes"]
    resolved = {}
    for episode in episodes:
        index = episode["episode_index"]
        folder = Path(mapping[str(index)])
        folder = (mapping_path.parent / folder).resolve(strict=True) if not folder.is_absolute() else folder.resolve(strict=True)
        marker = json.loads((folder / "verified.json").read_text())
        if marker["joint_atol"] != 1e-5:
            raise ValueError(f"Expected unchanged joint tolerance: {index}")
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
    (destination / "meta").mkdir()
    shutil.copyfile(source / "meta/tasks.jsonl", destination / "meta/tasks.jsonl")
    worker_info = copy.deepcopy(info)
    jobs = []
    offset = 0
    for output_index, episode in enumerate(episodes):
        jobs.append((source, destination, worker_info, episode, stats_by_episode[episode["episode_index"]],
                     *resolved[episode["episode_index"]], output_index, offset))
        offset += episode["length"]
    output_stats, output_markers = {}, {}
    executor = ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1, len(episodes)),
                                   mp_context=get_context("spawn"))
    try:
        for index, statistics, marker, features in executor.map(_augment_episode, jobs):
            output_stats[index] = statistics
            output_markers[index] = marker
            info["features"].update(features)
            print(f"Augmented official episode {index}/{len(episodes)}", flush=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    info.update(total_videos=6 * len(episodes), total_episodes=len(episodes), total_frames=offset,
                total_chunks=(len(episodes) + info["chunks_size"] - 1) // info["chunks_size"], splits={"train": f"0:{len(episodes)}"})
    (destination / "meta/info.json").write_text(json.dumps(info, indent=2))
    (destination / "meta/episodes_stats.jsonl").write_text(
        "".join(json.dumps(output_stats[i]) + "\n" for i in range(len(episodes))))
    (destination / "meta/episodes.jsonl").write_text("".join(
        json.dumps(dict(episode, episode_index=i)) + "\n" for i, episode in enumerate(episodes)))
    # Legacy stats.json exists in some releases; append the new eyes only.
    legacy_path = destination / "meta/stats.json"
    if (source / "meta/stats.json").exists():
        legacy = json.loads((source / "meta/stats.json").read_text())
        for key in (f"observation.images.{left}{suffix}" for left, _ in RIGS for suffix in ("", "_right")):
            entries = [output_stats[i]["stats"][key] for i in range(len(episodes))]
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
             replay_verification={str(i): output_markers[i] for i in range(len(episodes))},
             source_episode_indices=indices, source_total_episodes=source_total_episodes,
             pairing="fresh-left+fresh-right"), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Official clean LeRobot root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--episode-map", required=True)
    args = parser.parse_args()
    augment(args.source, args.output, args.episode_map)
