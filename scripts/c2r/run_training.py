"""One C2R training entry for the matched Wan and StereoTok experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from fastwam.utils.logging_config import setup_logging
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam_c2r.data import C2RDataset
from fastwam_c2r.training import C2RTrainer


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--variant',choices=['wan','stereotok'],required=True)
    parser.add_argument('--initialization',required=True,help='Shared proprio initialization published by the Wan run')
    parser.add_argument('--steps',type=int,help='Smoke-only stop limit; retains the full five-epoch LR schedule')
    parser.add_argument('--resume')
    parser.add_argument('--input-mode', choices=['stereo','mono-left'], default='stereo')
    args=parser.parse_args()
    setup_logging(log_level='INFO')
    from einops._torch_specific import allow_ops_in_compiled_graph
    allow_ops_in_compiled_graph()
    rank=int(os.environ.get('RANK',0))
    local=int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local)
    cfg=OmegaConf.load(args.config)
    prep=Path(os.environ['C2R_PREP_DIR'])
    report=json.loads((prep/'prepared.json').read_text())
    cfg.c2r.total_steps=report['total_steps']
    cfg.max_steps=report['total_steps'] if args.steps is None else args.steps
    cfg.save_every=2500
    cfg.resume=args.resume
    cfg.c2r.dataset.stereo=args.variant=='stereotok' and args.input_mode=='stereo'
    if args.variant=='stereotok':
        cfg.model.stereotok_checkpoint=os.environ['STEREOTOK_STUDENT_CKPT']
        cfg.model.stereotok_input_mode=args.input_mode
    out=Path(cfg.output_dir)
    if args.resume:
        if not (Path(args.resume)/'c2r_complete.json').is_file():
            raise ValueError('Only complete full state can resume')
    elif (out/'progress.json').exists():
        raise FileExistsError('Refuse fresh initialization into an existing run')
    out.mkdir(parents=True,exist_ok=True)
    set_global_seed(int(cfg.seed))
    native_load=torch.nn.Module.load_state_dict
    audit=[]
    def strict_audit(module,state_dict,*a,**kw):
        result=native_load(module,state_dict,*a,**kw)
        if module.__class__.__name__ in ('WanVideoDiT','WanVideoVAE38','WanTextEncoder','ActionDiT'):
            if result.missing_keys or result.unexpected_keys:
                raise ValueError((module.__class__.__name__,result))
            audit.append({'class':module.__class__.__name__,'keys':len(state_dict)})
        return result
    torch.nn.Module.load_state_dict=strict_audit
    try:
        model=instantiate(cfg.model,model_dtype=torch.bfloat16,device=f'cuda:{local}')
    finally:
        torch.nn.Module.load_state_dict=native_load
    assert model.proprio_dim==20 and model.action_expert.action_dim==20
    reference=Path(args.initialization)
    contract={'seed':int(cfg.seed),'manifest_sha256':report['manifest_sha256'],
              'action_init':str(cfg.model.action_dit_pretrained_path)}
    # Codec constructors consume different RNG streams. Keep Wan's original
    # initialization and share every randomly initialized head with StereoTok.
    heads = {'proprio': model.proprio_encoder,
             'action_encoder': model.action_expert.action_encoder,
             'action_head': model.action_expert.head}
    if args.variant=='stereotok':
        payload=torch.load(reference,map_location='cpu',weights_only=True)
        if payload['contract']!=contract: raise ValueError('Initialization contract mismatch')
        for name, module in heads.items():
            module.load_state_dict(payload['heads'][name], strict=True)
    fingerprints={}
    for name,module in [('video',model.video_expert),('action',model.action_expert),('proprio',model.proprio_encoder)]:
        first=next(module.parameters()).detach().contiguous().cpu()
        fingerprints[name]=hashlib.sha256(first.view(torch.uint8).numpy().tobytes()).hexdigest()
    for name,module in heads.items():
        digest=hashlib.sha256()
        for key,value in module.state_dict().items():
            digest.update(key.encode())
            digest.update(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
        fingerprints[name]=digest.hexdigest()
    if args.variant=='wan' and rank==0:
        payload={'contract':contract,'heads':{name:{k:v.detach().cpu() for k,v in module.state_dict().items()} for name,module in heads.items()},'fingerprints':fingerprints}
        if reference.exists():
            saved=torch.load(reference,map_location='cpu',weights_only=True)
            if saved['contract']!=contract or saved['fingerprints']!=fingerprints:
                raise ValueError('Existing common initialization differs')
        else:
            reference.parent.mkdir(parents=True,exist_ok=True)
            torch.save(payload,reference)
    if args.variant=='stereotok' and payload['fingerprints']!=fingerprints:
        raise ValueError('Trainable initialization differs between tokenizers')
    (out/f'initialization-rank{rank}.json').write_text(json.dumps(dict(contract=contract,fingerprints=fingerprints,audit=audit),indent=2)+'\n')
    dataset=C2RDataset(**OmegaConf.to_container(cfg.c2r.dataset,resolve=True))
    if rank==0: OmegaConf.save(cfg,out/'config.yaml',resolve=True)
    trainer=C2RTrainer(model,dataset,None,cfg=cfg)
    assert trainer.accelerator.num_processes in (4,8)
    assert trainer.batch_size*trainer.accelerator.num_processes*trainer.gradient_accumulation_steps==256
    if rank==0:
        if trainer.wandb_run is None or trainer.wandb_run.settings.mode!='online':
            raise RuntimeError('W&B online initialization required')
        (out/'run.json').write_text(json.dumps(dict(variant=args.variant,world_size=trainer.accelerator.num_processes,global_batch=256,
            total_steps=report['total_steps'],stop_step=cfg.max_steps,save_every=cfg.save_every,
            wandb_id=trainer.wandb_run.id,wandb_url=trainer.wandb_run.url,
            model_paths=model.model_paths,vae_frozen=all(not p.requires_grad for p in model.vae.parameters())),indent=2)+'\n')
    trainer.train()
    trainer.accelerator.wait_for_everyone()
    if rank==0:
        (out/'completed.json').write_text(json.dumps({'step':trainer.global_step,'target':cfg.max_steps})+'\n')
    trainer._finish_wandb()


if __name__=='__main__':main()
