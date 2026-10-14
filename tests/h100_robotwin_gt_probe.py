"""Check the exact allocated device and native render buffers before collection."""
import json
import os
import numpy as np
import sapien

device = sapien.Device('cuda:0')
print('DEVICE', device, flush=True)
sapien.render.set_camera_shader_dir('rt')
sapien.render.set_ray_tracing_samples_per_pixel(32)
sapien.render.set_ray_tracing_path_depth(8)
sapien.render.set_ray_tracing_denoiser('oidn')
scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(device=device)])
scene.add_ground(-1)
scene.set_ambient_light([.5, .5, .5])
scene.add_directional_light([0, 1, -1], [1, 1, 1])
builder = scene.create_actor_builder()
builder.add_box_visual(half_size=[.1, .1, .1])
actor = builder.build_kinematic(name='gt-probe')
actor.set_pose(sapien.Pose([1, 0, 0]))
camera = scene.add_camera('probe', 320, 240, 1, .01, 10)
scene.step()
scene.update_render()
camera.take_picture()
report = dict(job=os.environ['SLURM_JOB_ID'], device=str(device), buffers={})
for key in ('Color', 'Position', 'Segmentation'):
    array = camera.get_picture(key)
    assert array.shape == (240, 320, 4), (key, array.shape)
    assert np.isfinite(array).all(), key
    report['buffers'][key] = dict(shape=list(array.shape), dtype=str(array.dtype),
                                minimum=float(array.min()), maximum=float(array.max()))
assert np.any(-camera.get_picture('Position')[..., 2] > 0)
print(json.dumps(report), flush=True)
