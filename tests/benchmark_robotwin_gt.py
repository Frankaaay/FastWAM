"""Matched four-episode end-to-end throughput at 1/2/4 renderers per GPU."""
import concurrent.futures
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

repo = Path(__file__).resolve().parents[1]
root = Path('/gpfs/jiuquyun/projects/Frank/robotwin-stereo-gt-20261014')
width = (1, 2, 4)[int(os.environ['SLURM_PROCID'])]
output = root / 'benchmarks' / os.environ['SLURM_JOB_ID'] / f'concurrency-{width}'
output.mkdir(parents=True, exist_ok=False)
ids = (0, 16, 1, 2)


def run(index):
    with (output / f'{index:04d}.log').open('x') as log:
        subprocess.run([sys.executable, str(repo / 'scripts/h100_robotwin_gt.py'), 'episode',
                        '--index', str(index), '--output-root', str(output)],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    return json.loads((output / 'episodes' / f'{index:04d}/accepted.json').read_text())


start = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=width) as executor:
    rows = list(executor.map(run, ids))
seconds = time.monotonic() - start
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
report = dict(concurrency=width, source_sha=rows[0]['source_sha'], ids=ids,
              frames=sum(r['length'] for r in rows), seconds=seconds,
              frames_per_second=sum(r['length'] for r in rows)/seconds,
              user_cpu_seconds=usage.ru_utime, system_cpu_seconds=usage.ru_stime,
              records=[{k:r[k] for k in ('source_episode_index','length','seconds','timings','device')} for r in rows])
with (output / 'result.json').open('x') as f:
    json.dump(report, f, indent=2)
print(json.dumps(report), flush=True)
