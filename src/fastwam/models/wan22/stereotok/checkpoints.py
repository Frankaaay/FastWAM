"""Load the existing StereoTok student-v3 deployment format, without teachers."""

from pathlib import Path
import hashlib

import torch

from .student import GeometryDecoder, LAS2GuidedFusion, MetricHead, make_vae

EXPORT_FORMAT = "stereotok-student-v3"
SELECTED_SHA256 = "61c6057b49bec9c52f96793a20247d3746978939f0551c37142bdbda6b252eae"
STAGE3_U8000_SHA256 = "4c024b1dc9d9280e94a753ab9ccb1de15cc3c24659a1ec5aa86f94a3805f3503"
LEGACY_STAGE3_U8000_SHA256 = "5388b175591e0045f06b52fb0cfa5f769f1c03c79086c0d71661808e6b8df5e1"
DECODER_ARCHITECTURE = dict(stereo_version=3, decoder_split="after_middle",
                            depth_decoder="wan_tail", depth_head="causal_subpixel")


def load_student(path: str):
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    checksum = digest.hexdigest()
    expected_stage = {SELECTED_SHA256: 2, STAGE3_U8000_SHA256: 3,
                      LEGACY_STAGE3_U8000_SHA256: 3}.get(checksum)
    if expected_stage is None:
        raise ValueError("Expected a selected Stage2 u15853 or Stage3 u8000 checkpoint SHA256")
    legacy = checksum == LEGACY_STAGE3_U8000_SHA256
    # Only the exact previously validated training checkpoint admits RNG/optimizer metadata.
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=not legacy)
    if legacy:
        if payload.get("format") != "stereotok-wan-training-v1" or payload.get("architecture") != "stereo-v3":
            raise ValueError("Expected the legacy stereo-v3 training checkpoint")
        if payload["config"].get("stage") != 3 or payload.get("generator_updates") != 8000:
            raise ValueError("Expected legacy Stage3 at generator update 8000")
        geometry = payload["config"]["geometry"]
        # Match the previously published RGB loader: these loss heads are never called.
        weights = {k.removeprefix("vae."): v for k, v in payload["state_dict"].items()
                   if k.startswith("vae.") and not k.startswith(
                       ("vae.semantic_head.", "vae.latent_geometry_head."))}
    else:
        if payload.get("format") != EXPORT_FORMAT:
            raise ValueError("Expected a complete stereotok-student-v3 deployment export")
        if payload.get("stage") != expected_stage:
            raise ValueError("Student stage does not match the admitted export SHA256")
        if payload.get("semantic"):
            raise ValueError("This RGB backend requires an export without a semantic readout")
        geometry, weights = payload["geometry"], payload["state_dict"]
    if any(geometry.get(k) != v for k, v in DECODER_ARCHITECTURE.items()):
        raise ValueError("StereoTok decoder architecture mismatch")
    spec = geometry.get("encoder_geometry", {})
    if spec.get("mode") not in ("early", "late") or geometry.get("fusion_repeats") != [0, 0]:
        raise ValueError("This backend requires the LAS2-guided StereoTok architecture")
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
