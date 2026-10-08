"""Export the selected full Stage2/u8000 checkpoint without training dependencies."""
import argparse
import hashlib
from pathlib import Path

import torch


def export(source, destination, expected_updates=8000):
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if source == destination or destination.exists():
        raise FileExistsError(destination)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    # Only use trusted, user-owned training checkpoints (Python/NumPy RNG state).
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("format") != "stereotok-wan-training-v1":
        raise ValueError("Expected a full StereoTok training checkpoint")
    if payload.get("generator_updates") != expected_updates or payload["config"]["stage"] != 2:
        raise ValueError("Checkpoint must be Stage2 at the selected generator update")
    geometry = payload["config"]["geometry"]
    architecture = {key: geometry[key] for key in
                    ("fusion_dim", "fusion_heads", "max_disparity", "fusion_repeats")}
    architecture.update(stereo_version=3, decoder_split="after_middle",
                        depth_decoder="wan_tail", depth_head="causal_subpixel",
                        encoder_geometry=dict(geometry["encoder_geometry"]))
    state = {key.removeprefix("vae."): value for key, value in payload["state_dict"].items()
             if key.startswith("vae.") and not key.startswith("vae.latent_geometry_head.")}
    if not state or not any(key.startswith("fusion.matcher.") for key in state):
        raise ValueError("Checkpoint lacks complete student/LAS matcher weights")
    if payload["config"].get("semantic"):
        raise ValueError("Expected the RGB student without semantic readout")
    student = dict(format="stereotok-student-v3", geometry=architecture, semantic=False,
                   stage=2, state_dict=state,
                   provenance=dict(generator_updates=expected_updates, source_sha256=digest.hexdigest()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    with temporary.open("xb") as stream:
        torch.save(student, stream)
    temporary.rename(destination)
    print(f"Exported Stage2/u{expected_updates} to {destination}; source SHA256={digest.hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-updates", type=int, default=8000)
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.expected_updates)
