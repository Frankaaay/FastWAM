"""Load the existing StereoTok student-v3 deployment format, without teachers."""

from pathlib import Path
import hashlib

import torch

from .student import GeometryDecoder, LAS2GuidedFusion, make_vae

EXPORT_FORMAT = "stereotok-student-v3"
SELECTED_SHA256 = "61c6057b49bec9c52f96793a20247d3746978939f0551c37142bdbda6b252eae"
DECODER_ARCHITECTURE = dict(stereo_version=3, decoder_split="after_middle",
                            depth_decoder="wan_tail", depth_head="causal_subpixel")


def load_student(path: str):
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() not in (SELECTED_SHA256, "722511a4622860ed5061cd871171e5ebd4ada7ea8066b71141d401878e6cf588"):
        raise ValueError("This experiment requires an admitted u8000 or u15853 export SHA256")
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if payload.get("format") != EXPORT_FORMAT:
        raise ValueError("Expected a complete stereotok-student-v3 deployment export")
    if payload.get("stage") != 2:
        raise ValueError("This experiment requires a Stage2 student export")
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
    extra = ("fusion.", "decoder.geometry_branch.")
    vae.load_state_dict({k: v for k, v in weights.items() if not k.startswith(extra)},
                        strict=True, assign=True)
    with torch.random.fork_rng(devices=[]):
        vae.fusion = LAS2GuidedFusion(vae.encoder, spec, geometry["max_disparity"], initialize=False)
        vae.decoder.geometry_branch = GeometryDecoder(vae.decoder)
    vae.load_state_dict(weights, strict=True)
    return vae.eval().requires_grad_(False)
