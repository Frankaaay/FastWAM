from bisect import bisect_right
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random

import cv2
import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset
from fastwam.utils.samplers import ResumableEpochSampler
from .vendor.openwam.multiview import assemble_multiview_layout, format_prompt_for_inference
from .vendor.openwam.video import VideoColorJitter
from .vendor.openwam.normalize import Normalizer
from .vendor.openwam.rotation import quat_xyzw_to_rotation_6d, _rotation_6d_to_matrix

CAMERAS = ("head_camera", "left_camera", "right_camera")
ROTATION_DIMS = [3,4,5,6,7,8,13,14,15,16,17,18]

def read_eef(handle, start, stop):
    arms = []
    for side in ("left", "right"):
        pose = np.asarray(handle[f"endpose/{side}_endpose"][start:stop], dtype=np.float64)
        grip = np.asarray(handle[f"endpose/{side}_gripper"][start:stop], dtype=np.float64).reshape(-1,1)
        # H100 RoboTwin raw poses are xyz+wxyz. OpenWAM's helper accepts xyzw.
        r6 = quat_xyzw_to_rotation_6d(pose[:, [4,5,6,3]])
        arms.append(np.concatenate((pose[:,:3],r6,grip),axis=-1))
    return np.concatenate(arms,axis=-1).astype(np.float32)

def eef_to_native(eef):
    eef = np.asarray(eef, dtype=np.float64)
    result=[]
    for off in (0,10):
        arm=eef[...,off:off+10]
        matrix=_rotation_6d_to_matrix(arm[...,3:9])
        quat=Rotation.from_matrix(matrix.reshape(-1,3,3)).as_quat().reshape(*arm.shape[:-1],4)
        result.append(np.concatenate((arm[...,:3],quat[...,[3,0,1,2]],np.clip(arm[...,9:10],0,1)),axis=-1))
    return np.concatenate(result,axis=-1).astype(np.float32)

def observation_state(observation):
    arms=[]
    for side in ("left","right"):
        pose=np.asarray(observation["endpose"][side+"_endpose"],dtype=np.float64)
        grip=np.asarray(observation["endpose"][side+"_gripper"],dtype=np.float64).reshape(1)
        arms.append(np.concatenate((pose[:3],quat_xyzw_to_rotation_6d(pose[[4,5,6,3]]),grip)))
    return np.concatenate(arms).astype(np.float32)

def compose_observation(observation, right=False):
    frames={c:Image.fromarray(np.asarray(observation["observation"][c + ("_right" if right else "")]["rgb"],dtype=np.uint8)) for c in CAMERAS}
    return assemble_multiview_layout(frames,list(CAMERAS),384,320)

def text_key(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

class C2RDataset(Dataset):
    def __init__(self, root, manifest, text_cache=None, augment=True, require_context=True, stereo=False):
        self.stereo = bool(stereo)
        self.root=Path(root)
        self.manifest=json.loads(Path(manifest).read_text())
        self.episodes=self.manifest["episodes"]
        self.normalizer=Normalizer("min_max",stats={k:np.asarray(self.manifest["normalization"][k],dtype=np.float32) for k in ("min","max")})
        self.cumulative=[0]
        for ep in self.episodes:
            assert ep["seen_instructions"], ep["path"]
            self.cumulative.append(self.cumulative[-1]+ep["length"]-1)
        self.text_cache=None if text_cache is None else Path(text_cache)
        self.require_context=require_context
        self.augment=augment
        self.jitter=VideoColorJitter(.2,.2,.2,0)
        self.frame_offsets=np.arange(0,33,4)
        if len(self) != self.manifest["windows"]:
            raise ValueError("Manifest window count does not match episode lengths")
    def __len__(self):
        return self.cumulative[-1]
    @lru_cache(maxsize=16)
    def context(self,prompt):
        payload=torch.load(self.text_cache/(text_key(prompt)+".pt"),map_location="cpu",weights_only=True)
        assert payload["context_len"]==512 and payload["prompt"]==prompt
        compact=payload["context"]
        assert compact.ndim==2 and compact.shape[1]==4096 and len(compact)<=512
        context=torch.zeros(512,4096,dtype=torch.bfloat16)
        context[:len(compact)]=compact
        # Official FastWAM zeros padded embeddings and marks all context positions visible.
        return context,torch.ones(512,dtype=torch.bool)
    def __getitem__(self,index):
        if not 0<=index<len(self): raise IndexError(index)
        epi=bisect_right(self.cumulative,index)-1
        ep=self.episodes[epi]
        start=index-self.cumulative[epi]
        n=ep["length"]
        v_indices=start+self.frame_offsets
        frames=[]
        right_frames=[]
        with h5py.File(self.root/ep["path"],"r") as handle:
            eef=read_eef(handle,start,min(start+33,n))
            eef=eef[np.minimum(np.arange(33),len(eef)-1)]
            for fi in np.minimum(v_indices,n-1):
                images={}
                for camera in CAMERAS:
                    # Match the official OpenWAM reader: RoboTwin JPEG bytes were
                    # encoded from RGB with OpenCV; imdecode already restores RGB.
                    image=cv2.imdecode(np.frombuffer(bytes(handle[f"observation/{camera}/rgb"][int(fi)]),np.uint8),cv2.IMREAD_COLOR)
                    if image is None: raise ValueError(f"Undecodable image {ep['path']} {fi} {camera}")
                    images[camera]=Image.fromarray(image)
                frames.append(assemble_multiview_layout(images,list(CAMERAS),384,320))
                if self.stereo:
                    right_images = {}
                    for camera in CAMERAS:
                        image = cv2.imdecode(np.frombuffer(bytes(handle[f"observation/{camera}_right/rgb"][int(fi)]),np.uint8),cv2.IMREAD_COLOR)
                        if image is None: raise ValueError(f"Undecodable right image {ep['path']} {fi} {camera}")
                        right_images[camera] = Image.fromarray(image)
                    right_frames.append(assemble_multiview_layout(right_images,list(CAMERAS),384,320))
        if self.augment:
            # One draw for both eyes and all timesteps; left pixels and the
            # subsequent prompt RNG stream remain identical to the Wan branch.
            augmented = self.jitter.apply({"video":frames + right_frames})["video"]
            frames, right_frames = augmented[:9], augmented[9:]
        video=torch.stack([torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/127.5-1 for im in frames],dim=1)
        normalized=torch.from_numpy(self.normalizer.normalize(eef))
        prompt=format_prompt_for_inference(random.choice(ep["seen_instructions"]))
        sample={"video":video,"action":normalized[1:].clone(),"proprio":normalized[:1].clone(),
                "action_is_pad":torch.from_numpy(start+np.arange(1,33)>=n),
                "image_is_pad":torch.from_numpy(v_indices>=n),"prompt":prompt,
                "sample_index":index,"episode_index":epi,"start_frame":start}
        if self.stereo:
            sample["video_right"] = torch.stack([torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/127.5-1 for im in right_frames],dim=1)
        if self.require_context:
            sample["context"],sample["context_mask"]=self.context(prompt)
        return sample

class C2RSampler(ResumableEpochSampler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.total=math.ceil(len(self.dataset)/256)*256
    def _offset(self):
        return self.resume_batch_offset*self.batch_size*self.num_processes if self.epoch==0 else 0
    def __iter__(self):
        generator=torch.Generator().manual_seed(self.seed+self.epoch+self.epoch_offset)
        order=torch.randperm(len(self.dataset),generator=generator).tolist()
        order.extend(order[:self.total-len(order)])
        return iter(order[self._offset():])
    def __len__(self):
        return self.total-self._offset()

