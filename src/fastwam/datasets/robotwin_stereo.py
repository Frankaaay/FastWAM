"""Shared training/rollout preprocessing for three RoboTwin stereo rigs."""
import torch
import torchvision.transforms.functional as F

LEFT_CAMERAS = ("head_camera", "left_camera", "right_camera")
RIGHT_CAMERAS = ("head_camera_right", "left_camera_right", "right_camera_right")
LEROBOT_LEFT = ("cam_high", "cam_left_wrist", "cam_right_wrist")
LEROBOT_RIGHT = tuple(key + "_right" for key in LEROBOT_LEFT)


def compose_mosaic(cameras):
    """[3,T,3,H,W] in [0,1] -> [3,T,384,320] in [-1,1]."""
    if cameras.ndim != 5 or cameras.shape[0] != 3 or cameras.shape[2] != 3:
        raise ValueError("Expected exactly three RGB cameras")
    tiles = [F.resize(camera, size, interpolation=F.InterpolationMode.BILINEAR,
                      antialias=True) for camera, size in zip(cameras, ([256, 320], [128, 160], [128, 160]))]
    return (torch.cat((tiles[0], torch.cat(tiles[1:], -1)), -2) * 2 - 1).permute(1, 0, 2, 3)


def observation_mosaics(observation):
    result = []
    for names in (LEFT_CAMERAS, RIGHT_CAMERAS):
        cameras = []
        for name in names:
            image = torch.as_tensor(observation["observation"][name]["rgb"])
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != torch.uint8:
                raise ValueError(f"{name} must supply HWC uint8 RGB")
            image = image.permute(2, 0, 1).float().unsqueeze(0) / 255
            # Match the official processor's resize before mosaic construction.
            cameras.append(F.resize(image, [240, 320], antialias=True))
        result.append(compose_mosaic(torch.stack(cameras)).squeeze(1).unsqueeze(0))
    return tuple(result)
