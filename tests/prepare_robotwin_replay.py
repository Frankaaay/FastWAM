"""Prepare one adjust_bottle replay in a fresh personal test directory.

Shared raw observations/trajectories remain read-only symlinks. Only the Aloha
robot and bottle assets are extracted; no randomization or original data changes.
"""
import argparse
import getpass
import json
import subprocess
import zipfile
from pathlib import Path


def prepare(args):
    repo = Path(__file__).resolve().parents[1]
    personal = Path('/gpfs/jiuquyun/projects') / getpass.getuser()
    destination = args.output.resolve()
    if not destination.is_relative_to(personal):
        raise ValueError('Test output must be in the current user project directory')
    reference = args.reference.resolve(strict=True)
    expected = 'bf44be51cf5717a5595ce59447f2cf5263d2aa95'
    actual = subprocess.check_output(['git', '-C', str(reference), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != expected:
        raise ValueError('RoboTwin reference revision mismatch')
    if subprocess.check_output(['git', '-C', str(reference), 'status', '--porcelain'], text=True):
        raise ValueError('RoboTwin reference must be clean')
    source = args.raw.resolve(strict=True) / 'adjust_bottle/aloha-agilex_clean_50'
    inputs = [source / name for name in ('seed.txt', 'data/episode0.hdf5', '_traj_data/episode0.pkl')]
    if any(not p.is_file() for p in inputs):
        raise FileNotFoundError(inputs)
    destination.mkdir(parents=True, exist_ok=False)
    environment = destination / 'environment'
    environment.mkdir()
    for name in ('envs', 'script', 'description'):
        (environment / name).symlink_to(repo / 'third_party/RoboTwin' / name, target_is_directory=True)
    (environment / 'task_config').symlink_to(reference / 'task_config', target_is_directory=True)
    assets = environment / 'assets'
    assets.mkdir()
    extracted = []
    for archive, prefix in (('embodiments.zip', 'embodiments/aloha-agilex/'),
                            ('objects.zip', 'objects/001_bottle/'),
                            ('objects.zip', 'objects/objaverse/list.json'),
                            ('objects.zip', 'objects/same.json')):
        archive_path = args.archives.resolve(strict=True) / archive
        with zipfile.ZipFile(archive_path) as z:
            selected = [info for info in z.infolist() if info.filename.startswith(prefix)]
            if not selected:
                raise ValueError(f'Asset prefix missing: {archive}/{prefix}')
            for info in selected:
                target = (assets / info.filename).resolve()
                if not target.is_relative_to(assets.resolve()) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Unsafe archive member')
                z.extract(info, assets)
            extracted.append(dict(archive=str(archive_path), prefix=prefix,
                                  files=len(selected), bytes=sum(i.file_size for i in selected)))
    index = destination / 'source-index/adjust_bottle/demo_clean'
    (index / 'data').mkdir(parents=True)
    (index / '_traj_data').mkdir()
    (index / 'seed.txt').symlink_to(inputs[0])
    (index / 'data/episode0.hdf5').symlink_to(inputs[1])
    (index / '_traj_data/episode0.pkl').symlink_to(inputs[2])
    report = dict(reference_sha=actual, raw=str(source),
                  inputs=[str(p) for p in inputs], extracted=extracted,
                  source_index=str(destination / 'source-index'), environment=str(environment))
    (destination / 'preparation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args())
