"""Isolated six-camera RT batching capability and same-scene buffer comparison."""
import json
import time
import numpy as np
import sapien
import torch

sapien.render.set_camera_shader_dir('rt')
sapien.render.set_ray_tracing_samples_per_pixel(32)
sapien.render.set_ray_tracing_path_depth(8)
sapien.render.set_ray_tracing_denoiser('oidn')
device = sapien.Device('cuda:0')
renderer = sapien.render.RenderSystem(device=device)
scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), renderer])
scene.add_ground(-1)
scene.set_ambient_light([.5, .5, .5])
scene.add_directional_light([0, 1, -1], [1, 1, 1])
builder = scene.create_actor_builder()
builder.add_box_visual(half_size=[.1, .1, .1])
actor = builder.build_kinematic(name='probe')
actor.set_pose(sapien.Pose([1, 0, 0]))
cameras = [scene.add_camera(f'camera-{i}', 320, 240, 1, .01, 10) for i in range(6)]
for i, camera in enumerate(cameras):
    camera.set_local_pose(sapien.Pose([0, (i-3)*.01, 0]))
scene.step(); scene.update_render()
names = ('Color', 'Position', 'Segmentation')


def serial():
    for camera in cameras: camera.take_picture()
    return [[camera.get_picture(name).copy() for name in names] for camera in cameras]


serial()
start = time.monotonic()
for _ in range(10): reference = serial()
serial_seconds = time.monotonic()-start
repeat = serial()
systems = sapien.render.RenderSystemGroup([renderer])
group = systems.create_camera_group(cameras, list(names))
group.take_picture()
def grouped():
    group.take_picture()
    buffers = [torch.as_tensor(group.get_picture_cuda(name), device='cuda').cpu().numpy().copy() for name in names]
    return [[buffer[i] for buffer in buffers] for i in range(len(cameras))]
current = grouped()
print('GROUP_SHAPES', [[value.shape for value in camera] for camera in current], flush=True)
start = time.monotonic()
for _ in range(10):
    current = grouped()
group_seconds = time.monotonic()-start
errors = {}
for j, name in enumerate(names):
    if name == 'Position':
        errors[name] = max(float(np.max(np.abs(a[j][np.isfinite(a[j])] - b[j][np.isfinite(a[j])])))
                           for a,b in zip(reference,current))
    else:
        errors[name] = max(float(np.abs(a[j].astype(float)-b[j].astype(float)).max())
                           for a,b in zip(reference,current))
print(json.dumps(dict(pci=device.pci_string, serial_seconds=serial_seconds,
                      group_seconds=group_seconds, max_absolute_differences=errors)), flush=True)

def compare(a, b):
    result = {}
    for j, name in enumerate(names):
        result[name] = dict(max_abs=max(float(np.max(np.abs(x[j].astype(float)-y[j].astype(float)))) for x,y in zip(a,b)),
                           unequal=sum(int(np.count_nonzero(x[j]!=y[j])) for x,y in zip(a,b)))
    result['depth_max_abs'] = max(float(np.max(np.abs(x[1][...,2]-y[1][...,2]))) for x,y in zip(a,b))
    return result
print('SERIAL_REPEAT', json.dumps(compare(reference, repeat)), flush=True)
print('GROUP_COMPARE', json.dumps(compare(reference, current)), flush=True)
actor.set_pose(sapien.Pose([1,.2,0]))
cameras[0].set_local_pose(sapien.Pose([0,.04,.02]))
scene.update_render()
moving_serial=serial()
moving_group=grouped()
print('MOVING_COMPARE',json.dumps(compare(moving_serial,moving_group)),flush=True)
