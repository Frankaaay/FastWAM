"""Load the existing StereoTok student-v3 deployment format, without teachers."""

from pathlib import Path

import torch

from .student import GeometryDecoder, LAS2GuidedFusion, make_vae

EXPORT_FORMAT = "stereotok-student-v3"
DECODER_ARCHITECTURE = dict(stereo_version=3, decoder_split="after_middle",
                            depth_decoder="wan_tail", depth_head="causal_subpixel")


def load_student(path: str):
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if payload.get("format") != EXPORT_FORMAT:
        raise ValueError("Expected a complete stereotok-student-v3 deployment export")
    if payload.get("stage") != 2 or payload.get("provenance", {}).get("generator_updates") != 8000:
        raise ValueError("This experiment requires the selected Stage2/u8000 student export")
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
