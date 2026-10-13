"""Prepare the published 2500 stereo episodes without re-rendering or selection."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
from fastwam_c2r.vendor.openwam.training_utils import build_cosine_scheduler


def prepare(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Preparation output must be empty")
    published = Path(args.published_root)
    manifest_path = published / 'manifest.json'
    ready = json.loads((published / 'ready.json').read_text())
    reader = json.loads((published / 'reader-accepted.json').read_text())
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if not reader['passed'] or ready['reader_validation_pending'] or any(
        r['manifest_sha256'] != digest for r in (ready, reader)
    ):
        raise ValueError('Published manifest and reader evidence disagree')
    manifest = json.loads(manifest_path.read_text())
    episodes = manifest['episodes']
    ids = [e['source_episode_index'] for e in episodes]
    if len(ids) != 2500 or set(ids) != set(range(2500)):
        raise ValueError('Expected all 2500 unique canonical episodes')
    frames = sum(e['length'] for e in episodes)
    windows = sum(e['length'] - 1 for e in episodes)
    if windows != manifest['windows'] or frames != manifest['frames']:
        raise ValueError('Published window/frame count mismatch')
    updates = math.ceil(windows / 256)
    total = updates * 5
    # Reference the accepted metadata without copying any trajectory data.
    (output / 'manifest.json').symlink_to(manifest_path.resolve())
    prompts = json.loads((published / 'prompts.json').read_text())
    (output / 'prompts.json').symlink_to((published / 'prompts.json').resolve())
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
        publication_ready_sha256=hashlib.sha256((published / "ready.json").read_bytes()).hexdigest(),
        source_manifest_sha256=digest,
        manifest_sha256=hashlib.sha256((output / 'manifest.json').read_bytes()).hexdigest())
    (output / 'prepared.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--published-root', required=True)
    parser.add_argument('--output', required=True)
    prepare(parser.parse_args())
