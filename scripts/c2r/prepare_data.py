"""Reuse the fixed 2461 fresh stereo replays, without rendering or selection."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
import torch
from fastwam_c2r.data import CAMERAS, ROTATION_DIMS, read_eef
from fastwam_c2r.vendor.openwam.multiview import format_prompt_for_inference
from fastwam_c2r.vendor.openwam.training_utils import build_cosine_scheduler


def prepare(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Preparation output must be empty")
    provenance_path = Path(args.fresh_root) / 'meta/stereo_provenance.json'
    provenance = json.loads(provenance_path.read_text())
    source = json.loads(Path(args.source_manifest).read_text())
    instructions = {}
    for episode in source['episodes']:
        index = int(Path(episode['path']).stem.removeprefix('episode'))
        instructions[(episode['task'], index)] = episode['seen_instructions']
    ids = provenance['source_episode_indices']
    if len(ids) != 2461 or len(set(ids)) != 2461:
        raise ValueError('Expected the existing fixed 2461 episode set')
    episodes, frames = [], 0
    lo, hi = np.full(20, np.inf), np.full(20, -np.inf)
    root = Path(args.data_root).resolve()
    for output_index, canonical in enumerate(ids):
        folder = Path(provenance['episode_map'][str(canonical)]).resolve()
        # episode_map uses original IDs; replay_verification uses compact output IDs.
        marker = provenance['replay_verification'][str(output_index)]
        task, index = marker['task'], int(marker['episode'])
        raw = folder / 'data' / f'episode{index}.hdf5'
        if not raw.is_relative_to(root):
            raise ValueError(f'Replay outside the specified data root: {raw}')
        with h5py.File(raw, 'r') as handle:
            n = len(handle['joint_action/vector'])
            eef = read_eef(handle, 0, n)
            if n < 2 or eef.shape != (n, 20) or not np.isfinite(eef).all():
                raise ValueError(f'Invalid EEF sequence: {raw}')
            for camera in CAMERAS:
                for name in (camera, camera + '_right'):
                    if len(handle[f'observation/{name}/rgb']) != n:
                        raise ValueError(f'Frame count mismatch: {raw} {name}')
            lo = np.minimum(lo, eef.min(0))
            hi = np.maximum(hi, eef.max(0))
        episodes.append(dict(path=str(raw.relative_to(root)), task=task, length=n,
            source_episode_index=canonical, seen_instructions=instructions[(task, index)]))
        frames += n
        if len(episodes) % 250 == 0:
            print('PREPARED', len(episodes), frames, flush=True)
    lo[ROTATION_DIMS], hi[ROTATION_DIMS] = -1, 1
    windows = sum(e['length'] - 1 for e in episodes)
    updates = math.ceil(windows / 256)
    total = updates * 5
    manifest = dict(representation=source['representation'], native_quaternion='wxyz',
        pairing='fresh-left+fresh-right', episodes=episodes, frames=frames, windows=windows,
        normalization=dict(min=lo.tolist(), max=hi.tolist(),
            source='same fixed 2461 fresh episodes; rotations fixed +/-1'))
    (output / 'manifest.json').write_text(json.dumps(manifest) + '\n')
    prompts = sorted({format_prompt_for_inference(p) for ep in episodes for p in ep['seen_instructions']})
    (output / 'prompts.json').write_text(json.dumps(prompts, ensure_ascii=False) + '\n')
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))], lr=1.)
    cfg = SimpleNamespace(training=SimpleNamespace(learning_rate=1., warmup_ratio=.05, lr_min_ratio=.01))
    # Preserve ruhu's reference construction, including its 64 scheduler ticks/update.
    scheduler = build_cosine_scheduler(optimizer, total_opt_steps=total, cfg=cfg, num_processes=64)
    factors = [optimizer.param_groups[0]['lr']]
    for _ in range(total):
        optimizer.step()
        for _ in range(64):
            scheduler.step()
        factors.append(optimizer.param_groups[0]['lr'])
    (output / 'lr.json').write_text(json.dumps(dict(lr_factors=factors)) + '\n')
    report = dict(episodes=len(episodes), tasks=len({e['task'] for e in episodes}), frames=frames,
        windows=windows, updates_per_epoch=updates, total_steps=total, global_batch=256, epochs=5,
        excluded=sorted(set(range(2500)) - set(ids)), prompts=len(prompts),
        provenance_sha256=hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
        source_manifest_sha256=hashlib.sha256(Path(args.source_manifest).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((output / 'manifest.json').read_bytes()).hexdigest())
    (output / 'prepared.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh-root', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--output', required=True)
    prepare(parser.parse_args())
