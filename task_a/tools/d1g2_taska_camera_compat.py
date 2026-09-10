"""Side-effect-free camera pose reads for Isaac Sim 4.5 / newer Isaac Lab.

Install after AppLauncher, before constructing the environment. Do not create
Fabric world transforms on attached cameras: those override the parent link.
"""
from __future__ import annotations


def prepare_camera_views():
    import numpy as np
    import torch
    import usdrt
    from isaaclab.sim.views import XformPrimView
    from isaacsim.core.utils.xforms import get_world_pose

    if hasattr(getattr(usdrt, 'hierarchy', None), 'IFabricHierarchy'):
        return lambda: None
    original = XformPrimView._get_world_poses_fabric

    def read_poses(view, indices=None):
        if indices is None:
            selected = range(len(view.prim_paths))
        elif isinstance(indices, slice):
            selected = range(len(view.prim_paths))[indices]
        elif isinstance(indices, torch.Tensor):
            selected = indices.detach().cpu().tolist()
        else:
            selected = list(indices)
        # 4.5 recursively combines live Fabric link poses and USD local camera
        # transforms. Unlike XFormPrim.get_world_poses(usd=False), it writes no
        # _worldPosition/_worldOrientation attributes on the camera prim.
        poses = [get_world_pose(view.prim_paths[i]) for i in selected]
        positions = np.asarray([p[0] for p in poses], dtype=np.float32).reshape(-1, 3)
        orientations = np.asarray([p[1] for p in poses], dtype=np.float32).reshape(-1, 4)
        return (torch.as_tensor(positions, device=view._device),
                torch.as_tensor(orientations, device=view._device))

    XformPrimView._get_world_poses_fabric = read_poses
    print('[D1G2] Camera compatibility: recursive pose reads without Fabric writes.', flush=True)

    def restore():
        if XformPrimView._get_world_poses_fabric is read_poses:
            XformPrimView._get_world_poses_fabric = original

    return restore
