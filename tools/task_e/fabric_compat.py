"""Use Isaac Sim 4.5's Fabric pose reader with newer IsaacLab camera views.

Import after AppLauncher and install before gym.make. Keep the patch installed
through reset/step, then call the returned restore function in a finally block.
This only replaces pose reads; it leaves native Fabric, GPU dynamics, camera
rendering and the observation interface enabled and unchanged.
"""
from __future__ import annotations


def prepare_legacy_fabric_camera_views():
    """Adapt missing Fabric hierarchy support on Isaac Sim 4.5 only."""
    from tools.task_e.check_environment import runtime_version

    version = runtime_version()
    if not (version or "").startswith("4.5"):
        print(f"[Task E] Camera Fabric compatibility: skipped (Isaac Sim {version or 'unknown'}).",
              flush=True)
        return lambda: None

    import usdrt

    if hasattr(getattr(usdrt, "hierarchy", None), "IFabricHierarchy"):
        print("[Task E] Camera Fabric compatibility: skipped (native hierarchy API available).",
              flush=True)
        return lambda: None

    import torch
    from isaaclab.sim.views import XformPrimView
    from isaacsim.core.prims.impl.xform_prim import XFormPrim

    original = XformPrimView._get_world_poses_fabric

    def get_world_poses_legacy(view, indices=None):
        legacy = getattr(view, "_task_e_legacy_fabric_reader", None)
        if legacy is None:
            # No positions/orientations passed, no USD transform reset. The
            # constructor's default-state snapshot reads USD; live reads below
            # explicitly use the old _worldPosition/_worldOrientation API.
            legacy = XFormPrim(
                prim_paths_expr=list(view.prim_paths),
                name=f"task_e_camera_pose_{id(view)}",
                reset_xform_properties=False,
                usd=True,
            )
            view._task_e_legacy_fabric_reader = legacy
            path_to_index = {path: index for index, path in enumerate(legacy.prim_paths)}
            view._task_e_legacy_fabric_order = [path_to_index[path] for path in view.prim_paths]

        order = view._task_e_legacy_fabric_order
        if indices is None:
            selected = range(len(order))
        elif isinstance(indices, slice):
            selected = range(len(order))[indices]
        elif isinstance(indices, torch.Tensor):
            selected = indices.detach().cpu().tolist()
        else:
            selected = list(indices)
        # The 4.5 bridge reinterprets tensor indices with Warp.view(uint32).
        # Passing Python ints uses its explicit uint32 conversion and avoids
        # interpreting int64 indices as twice as many 32-bit entries.
        legacy_indices = [order[index] for index in selected]
        positions, orientations = legacy.get_world_poses(indices=legacy_indices, usd=False)

        def as_torch(value):
            if isinstance(value, torch.Tensor):
                return value.to(device=view._device, dtype=torch.float32)
            # Support a Warp frontend without changing SimulationManager's
            # global backend. NumPy is handled by torch.as_tensor directly.
            if type(value).__module__.startswith("warp"):
                value = value.numpy()
            return torch.as_tensor(value, device=view._device, dtype=torch.float32)

        return as_torch(positions), as_torch(orientations)

    XformPrimView._get_world_poses_fabric = get_world_poses_legacy
    print(f"[Task E] Camera Fabric compatibility: enabled (Isaac Sim {version}, missing hierarchy API).",
          flush=True)

    def restore():
        if XformPrimView._get_world_poses_fabric is get_world_poses_legacy:
            XformPrimView._get_world_poses_fabric = original

    return restore
