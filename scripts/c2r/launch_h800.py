"""Single-node, eight-GPU DLC entry for C2R preparation and matched training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['text','wan','stereotok'],required=True)
    parser.add_argument('--source-sha',required=True)
    parser.add_argument('--steps',type=int)
    parser.add_argument('--resume')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2]
    actual=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if actual!=args.source_sha or subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip():
        raise RuntimeError('Expected the published clean source SHA')
    base=Path('/mnt/workdir/frank')
    job=os.environ['DLC_JOB_ID']
    mode='smoke' if args.steps is not None or args.variant=='text' else 'train'
    output=base/'runs'/f'fastwam-c2r-{args.variant}-{mode}-20261013-{job}'
    output.mkdir(parents=True,exist_ok=False)
    prep=base/'runs/c2r-prep-2500-20261013/prepared'
    shared=base/'checkpoints/fastwam-c2r-shared-20261013'
    env=dict(os.environ)
    libs=json.loads((base/'runs/fastwam-h800-smoke-20261008/system-libs-prepared.json').read_text())
    py=base/'envs/fastwam-original-cu128/bin/python'
    overlays=[base/'envs/c2r-ds0189-20261013',base/'envs/fastwam-c2r-overlay-20261013',base/'envs/fastwam-stereotok-overlay-20261009']
    env.update(PYTHONPATH=':'.join(map(str,[repo/'src',*overlays])),
        LD_LIBRARY_PATH=libs['ld_library_path'],PATH=str(py.parent)+':'+env.get('PATH',''),
        C2R_PREP_DIR=str(prep),C2R_OUTPUT_DIR=str(output),C2R_RUN_NAME=output.name,
        C2R_TEXT_CACHE=str(base/'envs/c2r-text-cache-20261013'),
        C2R_ACTION_INIT=str(base/'checkpoints/fastwam-shared-clean15fps/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt'),
        STEREOTOK_STUDENT_CKPT=str(base/'checkpoints/stereotok/incoming-u15853/student-u15853.pt'),
        DIFFSYNTH_MODEL_BASE_PATH=str(shared),DIFFSYNTH_SKIP_DOWNLOAD='true',
        WANDB_MODE='online',WANDB_DIR=str(output),WANDB_CACHE_DIR=str(base/'envs/wandb-cache'),
        WANDB_CONFIG_DIR=str(base/'envs/wandb-config'),WANDB_DATA_DIR=str(base/'envs/wandb-data'),
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    env['WANDB_API_KEY']=(base/'envs/credentials/wandb-api-key').read_text().strip()
    for key,folder in [('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('CUDA_CACHE_PATH','cuda'),('TORCH_EXTENSIONS_DIR','extensions')]:
        env[key]=str(base/'envs/c2r-cache-20261013'/job/folder)
    if args.variant=='text':
        cmd=[str(py),'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
            str(repo/'scripts/c2r/cache_text.py'),'--prep',str(prep),'--cache',env['C2R_TEXT_CACHE']]
    else:
        cmd=[str(py),'-m','accelerate.commands.launch','--config_file',str(repo/'scripts/accelerate_configs/accelerate_zero2_ds.yaml'),
            '--deepspeed_config_file',str(repo/'configs/c2r/ds_zero2.json'),'--num_machines','1','--num_processes','8',
            str(repo/'scripts/c2r/run_training.py'),'--config',str(repo/'configs/c2r/train.yaml'),
            '--variant',args.variant,'--initialization',str(prep/'common-initialization-v2.pt')]
        if args.steps is not None:cmd+=['--steps',str(args.steps)]
        if args.resume:cmd+=['--resume',args.resume]
    (output/'launch.json').write_text(json.dumps(dict(source_sha=actual,command=cmd,variant=args.variant),indent=2)+'\n')
    with (output/'training.log').open('x') as log:
        child=subprocess.Popen(cmd,cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in child.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        code=child.wait()
    (output/'exit.json').write_text(json.dumps(dict(exit_code=code))+'\n')
    sys.exit(code)


if __name__=='__main__':main()
