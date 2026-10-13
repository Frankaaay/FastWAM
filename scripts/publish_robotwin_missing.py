"""Publish one C2R-compatible 2500-episode manifest without copying old data.

The input manifest is the already validated 2461-episode EEF20 dataset. Its
min/max statistics are merged exactly with the new 39, with rotation bounds
remaining [-1, 1]. No image of the original 2461 is re-rendered or overwritten.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

from collect_robotwin_missing import DATA, digest, verify_raw, write_json


def wait_for_collection(plan, run, deadline):
    """Bounded batch finalizer, not an additional GPU job."""
    until = time.monotonic() + deadline
    while True:
        ready = [list((Path(plan['output'])/f"canonical-{r['canonical_episode']:04d}").glob('attempt-*/accepted.json'))
                 for r in plan['rows']]
        if all(len(paths) == 1 for paths in ready):
            return
        if any(len(paths) > 1 for paths in ready):
            raise ValueError('Ambiguous accepted replacement')
        for worker in range(8):
            log = Path(run) / f'launcher-{worker}.log'
            if log.exists():
                with log.open('rb') as stream:
                    stream.seek(max(0, log.stat().st_size-8000))
                    tail = stream.read().decode(errors='replace')
                if 'Traceback (most recent call last)' in tail:
                    raise RuntimeError(f'Worker {worker} stopped with an exception; retain all outputs')
        if time.monotonic() >= until:
            raise TimeoutError('Collection deadline reached; incomplete version is not published')
        time.sleep(30)


def validate_readers(output, manifest, reader_root):
    import torch
    torch.set_num_threads(2)
    reader_root = Path(reader_root).resolve(strict=True)
    sha = subprocess.check_output(['git', '-C', str(reader_root), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != '633981aaa4d5b9e697774887b7cba4bc3289ea89':
        raise ValueError('Reader revision differs from the reviewed C2R reader')
    if subprocess.check_output(['git', '-C', str(reader_root), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Reader clone is dirty')
    sys.path.insert(0, str(reader_root/'src'))
    from fastwam_c2r.data import C2RDataset
    from fastwam_c2r.vendor.openwam.multiview import format_prompt_for_inference
    mono = C2RDataset(DATA, output/'manifest.json', augment=False, require_context=False, stereo=False)
    stereo = C2RDataset(DATA, output/'manifest.json', augment=False, require_context=False, stereo=True)
    selected = set()
    for index, ep in enumerate(manifest['episodes']):
        if ep['recovery_kind'] == 'new_demonstration' or index in (0, 1250, 2499):
            selected.update((mono.cumulative[index], mono.cumulative[index+1]-1))
    for index in sorted(selected):
        random.seed(42); a = mono[index]
        random.seed(42); b = stereo[index]
        for key in ('video', 'action', 'proprio', 'action_is_pad', 'image_is_pad'):
            if not torch.equal(a[key], b[key]):
                raise ValueError(f'Matched reader mismatch at {index}/{key}')
        if a['prompt'] != b['prompt']:
            raise ValueError('Matched reader prompt mismatch')
        if b['video'].shape != (3,9,384,320) or b['video_right'].shape != b['video'].shape or b['action'].shape != (32,20):
            raise ValueError('Reader tensor ABI mismatch')
        for key in ('video','video_right','action','proprio'):
            if not torch.isfinite(b[key]).all():
                raise ValueError(f'Nonfinite reader output: {key}')
        if torch.equal(b['video'], b['video_right']):
            raise ValueError('Reader returned duplicate eyes')
    prompts = sorted({format_prompt_for_inference(s) for ep in manifest['episodes'] for s in ep['seen_instructions']})
    write_json(output/'prompts.json', prompts)
    report = dict(passed=True,reader_sha=sha,manifest_sha256=digest(output/'manifest.json'),
        windows=len(mono),samples=len(selected),new_episode_first_and_last=True,
        left_action_proprio_prompt_exact_match=True,context_cache_checked=False,training_started=False)
    write_json(output/'reader-accepted.json', report)
    return report


def merge_manifest(existing, replacements):
    episodes = copy.deepcopy(existing['episodes'])
    ids = [e['source_episode_index'] for e in episodes]
    new_ids = [e['source_episode_index'] for e in replacements]
    if len(ids) != 2461 or len(set(ids)) != 2461 or len(new_ids) != 39 or len(set(new_ids)) != 39:
        raise ValueError('Expected unique 2461 existing and 39 replacement episodes')
    if set(ids) & set(new_ids) or set(ids + new_ids) != set(range(2500)):
        raise ValueError('Existing/replacement IDs must partition all 2500 original slots')
    for e in episodes:
        e['recovery_kind'] = 'existing_fresh_replay'
    episodes.extend(copy.deepcopy(replacements))
    episodes.sort(key=lambda e: e['source_episode_index'])
    counts = Counter(e['task'] for e in episodes)
    if len(counts) != 50 or set(counts.values()) != {50}:
        raise ValueError('Expected exactly 50 demonstrations for each of 50 tasks')
    for e in episodes:
        if e['length'] < 2 or not e['seen_instructions']:
            raise ValueError('Empty episode or instruction pool')
    out = copy.deepcopy(existing)
    out.update(episodes=episodes, frames=sum(e['length'] for e in episodes),
               windows=sum(e['length']-1 for e in episodes))
    return out


def main(args):
    import h5py
    plan_path, old_path = Path(args.plan), Path(args.existing_manifest)
    plan = json.loads(plan_path.read_text())
    if args.wait_run:
        wait_for_collection(plan, args.wait_run, 6*3600)
    existing = json.loads(old_path.read_text())
    if existing['representation'] != 'absolute_xyz_rot6d_columns_gripper20' or existing['native_quaternion'] != 'wxyz':
        raise ValueError('Unexpected EEF ABI')
    expected_old = set(range(2500)) - {r['canonical_episode'] for r in plan['rows']}
    if {e['source_episode_index'] for e in existing['episodes']} != expected_old:
        raise ValueError('Prepared 2461 manifest does not match frozen provenance')
    lower = np.array(existing['normalization']['min'], dtype=np.float32)
    upper = np.array(existing['normalization']['max'], dtype=np.float32)
    rot = [3,4,5,6,7,8,13,14,15,16,17,18]
    if lower.shape != (20,) or upper.shape != (20,) or not np.all(lower[rot] == -1) or not np.all(upper[rot] == 1):
        raise ValueError('Unexpected normalization contract')
    replacements, evidence = [], []
    for row in plan['rows']:
        matches = list((Path(plan['output'])/f"canonical-{row['canonical_episode']:04d}").glob('attempt-*/accepted.json'))
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous accepted sample: {row['canonical_episode']}")
        marker_path = matches[0]
        marker = json.loads(marker_path.read_text())
        if marker['plan_sha256'] != digest(plan_path) or marker['task'] != row['task'] or marker['canonical_episode'] != row['canonical_episode']:
            raise ValueError('Replacement provenance mismatch')
        raw = Path(marker['path']).resolve(strict=True)
        if Path(plan['output']).resolve() not in raw.parents or digest(raw) != marker['sha256']:
            raise ValueError('Replacement path/hash mismatch')
        n = verify_raw(raw)
        if n != marker['length']:
            raise ValueError('Length mismatch')
        with h5py.File(raw, 'r') as f:
            for side, offset in (('left',0),('right',10)):
                pose = np.asarray(f[f'endpose/{side}_endpose'][:], dtype=np.float32)
                grip = np.asarray(f[f'endpose/{side}_gripper'][:], dtype=np.float32).reshape(-1)
                lower[offset:offset+3] = np.minimum(lower[offset:offset+3], pose[:,:3].min(0))
                upper[offset:offset+3] = np.maximum(upper[offset:offset+3], pose[:,:3].max(0))
                lower[offset+9] = min(lower[offset+9], grip.min())
                upper[offset+9] = max(upper[offset+9], grip.max())
        replacements.append(dict(path=str(raw.relative_to(DATA)), task=row['task'], length=n,
            source_episode_index=row['canonical_episode'], seen_instructions=marker['seen_instructions'],
            recovery_kind='new_demonstration', original_episode=row['episode'], original_seed=row['seed'],
            replacement_seed=marker['seed'], sha256=marker['sha256']))
        evidence.append(dict(canonical_episode=row['canonical_episode'], marker=str(marker_path),
                             marker_sha256=digest(marker_path), original=row))
    manifest = merge_manifest(existing, replacements)
    manifest['normalization'] = dict(min=lower.tolist(), max=upper.tolist(),
        source='exact union of frozen 2461 min/max and 39 new demonstrations; rotations fixed +/-1')
    manifest['dataset_version'] = 'robotwin-fresh-stereo-2500-fill39-20261013'
    manifest['provenance'] = dict(existing_manifest=str(old_path), existing_manifest_sha256=digest(old_path),
        plan=str(plan_path), plan_sha256=digest(plan_path), original_stereo_provenance_sha256=plan['provenance_sha256'],
        existing_episodes=2461, newly_collected_episodes=39, original_demonstrations_recovered=0,
        replacement_mapping=evidence)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'manifest.json', manifest)
    write_json(output/'seen-instructions.json', sorted({s for e in manifest['episodes'] for s in e['seen_instructions']}))
    summary = dict(version=manifest['dataset_version'], data_root=str(DATA), manifest=str(output/'manifest.json'),
        manifest_sha256=digest(output/'manifest.json'), episodes=len(manifest['episodes']),
        tasks=dict(sorted(Counter(e['task'] for e in manifest['episodes']).items())),
        frames=manifest['frames'], windows=manifest['windows'],
        old_frames=existing['frames'], old_windows=existing['windows'],
        new_frames=sum(e['length'] for e in replacements), new_windows=sum(e['length']-1 for e in replacements),
        normalization_bound_to_manifest=True, training_started=False, reader_validation_pending=True)
    write_json(output/'published.json', summary)
    report = validate_readers(output, manifest, args.reader_root)
    summary['reader_validation_pending'] = False
    write_json(output/'ready.json', dict(**summary, reader_validation=report))
    print(json.dumps(summary))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--existing-manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--reader-root', required=True)
    p.add_argument('--wait-run')
    main(p.parse_args())
