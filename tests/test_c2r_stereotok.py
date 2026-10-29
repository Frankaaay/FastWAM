"""CPU checks for matched C2R data, augmentation, and temporal targets."""
import json
from pathlib import Path
import random
import tempfile
import unittest
import ast
import cv2
import h5py
import numpy as np
import torch
from fastwam_c2r.data import CAMERAS, C2RDataset, C2RSampler

class MatchedDataTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='c2r-contract-'))  # retained, no recursive deletion
        n=35
        with h5py.File(self.root/'episode.hdf5','w') as h:
            for side in ('left','right'):
                pose=np.zeros((n,7));pose[:,0]=np.arange(n)/100;pose[:,3]=1
                h[f'endpose/{side}_endpose']=pose
                h[f'endpose/{side}_gripper']=np.arange(n)/100
            encoded=[]
            for t in range(n):
                image=np.full((12,16,3),20+t,np.uint8);image[:,8:]=[80,100,120]
                encoded.append(cv2.imencode('.jpg',image)[1].tobytes())
            values=np.asarray(encoded,dtype=f'S{max(map(len,encoded))}')
            for camera in CAMERAS:
                h[f'observation/{camera}/rgb']=values
                h[f'observation/{camera}_right/rgb']=values
        manifest={'windows':n-1,'episodes':[{'path':'episode.hdf5','length':n,'seen_instructions':['one','two']}],
                  'normalization':{'min':[-1]*20,'max':[1]*20}}
        (self.root/'manifest.json').write_text(json.dumps(manifest))
    def dataset(self,stereo,augment=True):
        return C2RDataset(self.root,self.root/'manifest.json',augment=augment,require_context=False,stereo=stereo)
    def test_matched_left_actions_prompts_and_rng(self):
        wan,stereo=self.dataset(False),self.dataset(True)
        for index in (0,17,33):
            random.seed(42);a=wan[index];rng=random.getstate()
            random.seed(42);b=stereo[index]
            self.assertEqual(rng,random.getstate());self.assertEqual(a['prompt'],b['prompt'])
            for key in ('video','action','proprio','image_is_pad','action_is_pad'):
                self.assertTrue(torch.equal(a[key],b[key]),key)
            self.assertTrue(torch.equal(b['video'],b['video_right']))
    def test_future_eef_and_tail_padding(self):
        ds=self.dataset(False,False);sample=ds[0]
        for value,expected in [(sample['proprio'][0,0],0),(sample['action'][0,0],.01),(sample['action'][-1,0],.32)]:
            self.assertAlmostEqual(value.item(),expected,places=6)
        tail=ds[33];self.assertFalse(tail['action_is_pad'][0]);self.assertTrue(tail['action_is_pad'][1:].all())
        self.assertTrue(torch.allclose(tail['action'][:,0],torch.full((32,),.34)))
        self.assertEqual(tuple(tail['video'].shape),(3,9,384,320))
    def test_missing_right_fails(self):
        with h5py.File(self.root/'left.hdf5','w') as dst,h5py.File(self.root/'episode.hdf5','r') as src:
            src.copy('endpose',dst)
            for camera in CAMERAS:src.copy(f'observation/{camera}',dst.require_group('observation'),name=camera)
        payload=json.loads((self.root/'manifest.json').read_text());payload['episodes'][0]['path']='left.hdf5'
        (self.root/'left.json').write_text(json.dumps(payload))
        ds=C2RDataset(self.root,self.root/'left.json',require_context=False,stereo=True)
        with self.assertRaises(KeyError):ds[0]
    def test_sampler_padding_resume(self):
        sampler=C2RSampler(range(513),42,8,8);order=list(sampler)
        self.assertEqual(len(order),768);self.assertEqual(len(set(order)),513);self.assertEqual(order[513:],order[:255])
        sampler.set_resume_batch_offset(4);self.assertEqual(list(sampler),order[256:])
        sampler.clear_resume_batch_offset();sampler.set_epoch(1);self.assertNotEqual(list(sampler),order)
class CheckpointTests(unittest.TestCase):
    def test_mono_stereo_input_contract_and_frozen_precision(self):
        import hashlib
        path=Path(__file__).resolve().parents[1]/'src/fastwam/models/wan22/stereotok_vae.py'
        tree=ast.parse(path.read_text());tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef)]
        class Codec(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.ones(1));self.fusion=type('Fusion',(),{'max_disparity':112})()
            def encode_posterior(self,left,*args,**kw):
                self.right=args[0] if args else None
                return torch.zeros(len(left),48,1,2,2),None
        scope=dict(torch=torch,nn=torch.nn,Path=Path,hashlib=hashlib,load_student=lambda _:Codec(),
                   latent_scale=lambda _: (torch.zeros(48),torch.ones(48)))
        exec(compile(tree,str(path),'exec'),scope)
        out=Path(tempfile.mkdtemp(prefix='codec-mode-'))/'metadata.pt';torch.save({'format':'stereotok-wan-training-v1'},out)
        image=torch.zeros(1,3,1,32,32)
        mono=scope['StereoTokVAE'](out,'cpu',input_mode='mono-left')
        stereo=scope['StereoTokVAE'](out,'cpu',input_mode='stereo')
        torch.testing.assert_close(mono.encode(image,'cpu'),stereo.encode(image,'cpu',right_videos=image))
        self.assertIsNone(mono.model.right);self.assertIsNotNone(stereo.model.right)
        with self.assertRaises(ValueError):mono.encode(image,'cpu',right_videos=image)
        with self.assertRaises(ValueError):stereo.encode(image,'cpu')
        mono.train().to(dtype=torch.bfloat16)
        self.assertFalse(mono.training);self.assertFalse(mono.model.weight.requires_grad)
        self.assertEqual(mono.model.weight.dtype,torch.float32)
        self.assertNotEqual(mono.binding,stereo.binding)

    def test_stage3_keeps_inference_weights_and_rejects_missing_keys(self):
        # Exercise the production loader with tiny modules, without LAS/GPU dependencies.
        path=Path(__file__).resolve().parents[1]/'src/fastwam/models/wan22/stereotok/checkpoints.py'
        tree=ast.parse(path.read_text())
        tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
        class VAE(torch.nn.Module):
            def __init__(self):
                super().__init__();self.encoder=torch.nn.Linear(2,2);self.decoder=torch.nn.Linear(2,2)
        scope=dict(torch=torch,Path=Path,make_vae=VAE,
                   LAS2GuidedFusion=lambda *a,**k:torch.nn.Linear(2,2),
                   GeometryDecoder=lambda *a:torch.nn.Linear(2,2))
        exec(compile(tree,str(path),'exec'),scope)
        model=VAE();model.fusion=torch.nn.Linear(2,2);model.decoder.geometry_branch=torch.nn.Linear(2,2)
        geometry=dict(scope['DECODER_ARCHITECTURE'],encoder_geometry={'mode':'early'},fusion_repeats=[0,0],max_disparity=112)
        weights={'vae.'+k:v for k,v in model.state_dict().items()}
        weights['vae.semantic_head.layers.0.weight']=torch.ones(2,2)
        weights['vae.latent_geometry_head.layers.0.weight']=torch.ones(2,2)
        payload=dict(format='stereotok-wan-training-v1',architecture='stereo-v3',config=dict(stage=3,geometry=geometry),state_dict=weights)
        out=Path(tempfile.mkdtemp(prefix='c2r-stage3-'))/'test.pt'
        torch.save(payload,out)
        actual=scope['load_student'](str(out))
        for key,value in model.state_dict().items():torch.testing.assert_close(actual.state_dict()[key],value,rtol=0,atol=0)
        self.assertFalse(any(p.requires_grad for p in actual.parameters()))
        out=out.with_name('missing.pt')
        del weights['vae.encoder.weight'];torch.save(payload,out)
        with self.assertRaises(RuntimeError):scope['load_student'](str(out))
        out=out.with_name('stage1.pt')
        payload['config']['stage']=1;torch.save(payload,out)
        with self.assertRaisesRegex(ValueError,'Stage2 or Stage3'):scope['load_student'](str(out))

if __name__=='__main__':unittest.main()
