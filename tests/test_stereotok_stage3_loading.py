"""Stage3 export admission and strict-load contracts without large model assets."""
import ast
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]


class Stage3Tests(unittest.TestCase):
    def test_stage3_metric_head_strict_roundtrip_and_stage_guard(self):
        tree=ast.parse((ROOT/'src/fastwam/models/wan22/stereotok/student.py').read_text())
        head=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MetricHead')
        scope={'nn':nn}
        exec(compile(ast.Module(body=[head],type_ignores=[]),'student.py','exec'),scope)
        MetricHead=scope['MetricHead']
        def make_vae():
            model=nn.Module();model.encoder=nn.Linear(2,2);model.decoder=nn.Linear(2,2)
            return model
        original=make_vae();original.fusion=nn.Linear(2,2)
        original.decoder.geometry_branch=nn.Linear(2,2);original.metric_head=MetricHead()
        payload={'format':'stereotok-student-v3','stage':3,'semantic':False,
            'geometry':{'encoder_geometry':{'mode':'early'},'fusion_repeats':[0,0],
                        'max_disparity':112,'metric_disparity':{'shallow_head_weight':1}},
            'state_dict':original.state_dict()}
        tree=ast.parse((ROOT/'src/fastwam/models/wan22/stereotok/checkpoints.py').read_text())
        loader=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_student')
        scope.update(torch=torch,Path=Path,hashlib=hashlib,make_vae=make_vae,
            EXPORT_FORMAT='stereotok-student-v3',SELECTED_SHA256='old',DECODER_ARCHITECTURE={},
            GeometryDecoder=lambda decoder:nn.Linear(2,2),LAS2GuidedFusion=lambda *a,**kw:nn.Linear(2,2))
        exec(compile(ast.Module(body=[loader],type_ignores=[]),'checkpoints.py','exec'),scope)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'export.pth'
            def save():
                torch.save(payload,path)
                scope['STAGE3_U8000_SHA256']=hashlib.sha256(path.read_bytes()).hexdigest()
            save();loaded=scope['load_student'](str(path))
            for k,v in original.state_dict().items():self.assertTrue(torch.equal(v,loaded.state_dict()[k]),k)
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
            self.assertEqual(tuple(loaded.metric_head(torch.zeros(1,48,2,3,4)).shape),(1,1,2,48,64))
            payload['stage']=2;save()
            with self.assertRaisesRegex(ValueError,'stage'):scope['load_student'](str(path))
            payload['stage']=3;payload['state_dict'].pop('metric_head.layers.0.bias');save()
            with self.assertRaises(RuntimeError):scope['load_student'](str(path))


if __name__=='__main__':unittest.main()
