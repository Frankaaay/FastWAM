"""RoboTwin EEF policy adapter. FastWAM inference itself remains official."""
from collections import deque
import os
from pathlib import Path
import time
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate
import torch
from fastwam.models.wan22.stereotok_vae import StereoTokVAE
from .data import observation_state, compose_observation, eef_to_native
from .vendor.openwam.multiview import format_prompt_for_inference
from .vendor.openwam.normalize import Normalizer
import json

class C2RPolicy:
    def __init__(self, checkpoint, root, device='cuda'):
        root=Path(root)
        cfg=OmegaConf.load(root/'config.yaml')
        os.environ['DIFFSYNTH_MODEL_BASE_PATH']=str(cfg.c2r.assets_root)
        os.environ['DIFFSYNTH_SKIP_DOWNLOAD']='true'
        self.inference_seed=int(cfg.c2r.evaluation.seed)
        cfg.model.load_text_encoder=True
        cfg.model.skip_dit_load_from_pretrain=True
        cfg.model.action_dit_pretrained_path=None
        self.model=instantiate(cfg.model,model_dtype=torch.bfloat16,device=device)
        payload=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
        if payload.get('tokenizer') != getattr(self.model.vae, 'binding', None):
            raise ValueError('Policy/tokenizer checkpoint binding mismatch')
        self.model.mot.load_state_dict(payload['mot'],strict=True)
        self.model.proprio_encoder.load_state_dict(payload['proprio_encoder'],strict=True)
        self.model.eval().requires_grad_(False)
        stats=json.loads((Path(cfg.c2r.dataset.manifest)).read_text())['normalization']
        self.normalizer=Normalizer('min_max',stats={k:np.asarray(stats[k],dtype=np.float32) for k in ('min','max')})
        self.pending=deque()
        self.inference_seconds=[]
        self._prompt = None
        self._context = None
    def should_request_observation(self):
        return not self.pending
    def step(self,env,observation):
        if not self.pending:
            if observation is None: raise ValueError('Fresh observation required for replanning')
            image=compose_observation(observation)
            image=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1)[None].to(self.model.device,torch.float32 if isinstance(self.model.vae, StereoTokVAE) else self.model.torch_dtype)/127.5-1
            extra = {}
            if getattr(self.model.vae, "requires_stereo", False):
                right = compose_observation(observation, right=True)
                extra["input_image_right"] = torch.from_numpy(np.asarray(right).copy()).permute(2,0,1)[None].to(self.model.device,torch.float32)/127.5-1
                image = torch.from_numpy(np.asarray(compose_observation(observation)).copy()).permute(2,0,1)[None].to(self.model.device,torch.float32)/127.5-1
            state=torch.from_numpy(self.normalizer.normalize(observation_state(observation)))[None]
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.no_grad():
                prompt = format_prompt_for_inference(env.get_instruction())
                if prompt != self._prompt:
                    self._context = self.model.encode_prompt(prompt)
                    self._prompt = prompt
                result=self.model.infer_action(prompt=None,context=self._context[0],context_mask=self._context[1],input_image=image,
                    action_horizon=32,proprio=state,num_inference_steps=10,sigma_shift=5.0,text_cfg_scale=1.0,
                    negative_prompt='',seed=self.inference_seed,rand_device='cpu',tiled=False,**extra)
            torch.cuda.synchronize();self.inference_seconds.append(time.perf_counter()-start)
            action=result['action'].detach().float().cpu().numpy().reshape(-1,20)
            native=eef_to_native(self.normalizer.unnormalize(action))
            assert len(native)==32 and np.isfinite(native).all()
            self.pending.extend(native)
        env.take_action(self.pending.popleft(),action_type='ee')
    def reset(self):
        self.pending.clear()
        self.inference_seconds.clear()

def get_model(usr_args):
    return C2RPolicy(usr_args['checkpoint_path'],usr_args['experiment_root'],usr_args.get('device','cuda'))
def eval(TASK_ENV,model,observation):
    model.step(TASK_ENV,observation)
def reset_model(model):
    model.reset()
