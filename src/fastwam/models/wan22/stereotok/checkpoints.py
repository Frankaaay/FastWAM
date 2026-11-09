"""Load the existing StereoTok student-v3 deployment format, without teachers."""

from pathlib import Path
import hashlib

import torch

from .student import GeometryDecoder, LAS2GuidedFusion, MetricHead, make_vae

EXPORT_FORMAT = "stereotok-student-v3"
SELECTED_SHA256 = "61c6057b49bec9c52f96793a20247d3746978939f0551c37142bdbda6b252eae"
STAGE3_U8000_SHA256 = "4c024b1dc9d9280e94a753ab9ccb1de15cc3c24659a1ec5aa86f94a3805f3503"
DECODER_ARCHITECTURE = dict(stereo_version=3, decoder_split="after_middle",
                            depth_decoder="wan_tail", depth_head="causal_subpixel")


def load_student(path: str):
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    expected_stage = {SELECTED_SHA256: 2, STAGE3_U8000_SHA256: 3}.get(digest.hexdigest())
    if expected_stage is None:
        raise ValueError("Expected the selected Stage2 u15853 or clean Stage3 u8000 export SHA256")
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if payload.get("format") != EXPORT_FORMAT:
        raise ValueError("Expected a complete stereotok-student-v3 deployment export")
    if payload.get("stage") != expected_stage:
        raise ValueError("Student stage does not match the admitted export SHA256")
    geometry = payload["geometry"]
    if any(geometry.get(k) != v for k, v in DECODER_ARCHITECTURE.items()):
        raise ValueError("StereoTok decoder architecture mismatch")
    spec = geometry.get("encoder_geometry", {})
    if spec.get("mode") not in ("early", "late") or geometry.get("fusion_repeats") != [0, 0]:
        raise ValueError("This backend requires the LAS2-guided StereoTok architecture")
    if payload.get("semantic"):
        raise ValueError("This RGB backend requires an export without a semantic readout")
    weights = payload["state_dict"]
    vae = make_vae()
    extra = ("fusion.", "decoder.geometry_branch.", "metric_head.")
    vae.load_state_dict({k: v for k, v in weights.items() if not k.startswith(extra)},
                        strict=True, assign=True)
    with torch.random.fork_rng(devices=[]):
        vae.fusion = LAS2GuidedFusion(vae.encoder, spec, geometry["max_disparity"], initialize=False)
        vae.decoder.geometry_branch = GeometryDecoder(vae.decoder)
        if geometry.get("metric_disparity", {}).get("shallow_head_weight", 0) > 0:
            vae.metric_head = MetricHead()
    vae.load_state_dict(weights, strict=True)
    return vae.eval().requires_grad_(False)
