"""CPU checks for matched C2R data, augmentation, and temporal targets."""
import json
from pathlib import Path
import random
import tempfile
import unittest
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
if __name__=='__main__':unittest.main()
