"""Load the existing StereoTok student-v3 deployment format, without teachers."""

from pathlib import Path

import torch

from .student import GeometryDecoder, LAS2GuidedFusion, make_vae

EXPORT_FORMAT = "stereotok-student-v3"
DECODER_ARCHITECTURE = dict(stereo_version=3, decoder_split="after_middle",
                            depth_decoder="wan_tail", depth_head="causal_subpixel")


def load_student(path: str):
    # Trusted training checkpoints also contain Python/NumPy optimizer/RNG metadata.
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False, mmap=True)
    if payload.get("format") == "stereotok-wan-training-v1":
        if payload.get("architecture") != "stereo-v3":
            raise ValueError("Expected a stereo-v3 training checkpoint")
        geometry = payload["config"]["geometry"]
        stage = payload["config"]["stage"]
        weights = {k.removeprefix("vae."): v for k, v in payload["state_dict"].items()
                   if k.startswith("vae.")}
    elif payload.get("format") == EXPORT_FORMAT:
        geometry, stage, weights = payload["geometry"], payload["stage"], payload["state_dict"]
    else:
        raise ValueError("Expected a complete stereotok-student-v3 deployment export")
    if stage not in (2, 3):
        raise ValueError("Expected a Stage2 or Stage3 student")
    if any(geometry.get(k) != v for k, v in DECODER_ARCHITECTURE.items()):
        raise ValueError("StereoTok decoder architecture mismatch")
    spec = geometry.get("encoder_geometry", {})
    if spec.get("mode") not in ("early", "late") or geometry.get("fusion_repeats") != [0, 0]:
        raise ValueError("This backend requires the LAS2-guided StereoTok architecture")
    # These pointwise loss readouts are not called by RGB encode/decode.
    weights = {k: v for k, v in weights.items()
               if not k.startswith(("semantic_head.", "latent_geometry_head."))}
    vae = make_vae()
    extra = ("fusion.", "decoder.geometry_branch.")
    vae.load_state_dict({k: v for k, v in weights.items() if not k.startswith(extra)},
                        strict=True, assign=True)
    with torch.random.fork_rng(devices=[]):
        vae.fusion = LAS2GuidedFusion(vae.encoder, spec, geometry["max_disparity"], initialize=False)
        vae.decoder.geometry_branch = GeometryDecoder(vae.decoder)
    vae.load_state_dict(weights, strict=True)
    return vae.eval().requires_grad_(False)
