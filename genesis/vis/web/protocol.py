"""
WebSocket message protocol for Genesis Web GUI.

The WebSocket carries two kinds of messages:
- Binary messages (server -> client): JPEG-encoded frames
- Text/JSON messages (bidirectional): control and state messages
"""

from enum import Enum

import numpy as np

import genesis as gs
from genesis.utils.misc import tensor_to_array
from genesis.vis.scene_ops import build_entity_joint_data


class MsgType(str, Enum):
    """Message types for the WebSocket protocol."""

    # Client -> Server
    SIM_CONTROL = "sim_control"
    CAMERA_UPDATE = "camera_update"
    ENTITY_UPDATE = "entity_update"
    VIS_TOGGLE = "vis_toggle"
    # Server -> Client
    STATE_UPDATE = "state_update"
    SCENE_INFO = "scene_info"


def _sanitize_floats(arr):
    """Replace inf/NaN with finite values for JSON serialization."""
    return np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)


def _joint_info(joint):
    """Extract serializable info from a RigidJoint."""
    dofs_limit = joint.dofs_limit
    if isinstance(dofs_limit, np.ndarray):
        limits = _sanitize_floats(dofs_limit).tolist()
    else:
        limits = _sanitize_floats(tensor_to_array(dofs_limit)).tolist()

    return {
        "name": joint.name,
        "type": joint.type.name,
        "n_dofs": joint.n_dofs,
        "n_qs": joint.n_qs,
        "dofs_limit": limits,
    }


def build_scene_info(scene):
    """Build a SCENE_INFO message from the current scene.

    Inspects scene entities and returns a dict ready for JSON serialization.
    """
    entities = []
    for entity in scene.entities:
        info = {
            "name": entity.name,
            "idx": entity.idx,
        }

        # Per-entity vis_mode and contact visualization
        try:
            info["vis_mode"] = getattr(entity.surface, "vis_mode", "visual") if hasattr(entity, "surface") else "visual"
        except Exception:
            info["vis_mode"] = "visual"
        info["visualize_contact"] = getattr(entity, "visualize_contact", False)

        # Only rigid entities have joints/dofs
        if hasattr(entity, "n_dofs") and entity.n_dofs > 0:
            info["n_dofs"] = entity.n_dofs
            info["n_qs"] = entity.n_qs
            try:
                info["joints"] = [_joint_info(j) for j in entity.joints]
            except Exception:
                info["joints"] = []

            # Rich joint data from shared module
            try:
                joint_data = build_entity_joint_data(entity)
                info.update(joint_data)
            except Exception:
                info["q_names"] = []
                info["q_limits_lower"] = []
                info["q_limits_upper"] = []
                info["q_is_quaternion"] = []
                info["quat_groups"] = []
                info["has_free_joint"] = False
                info["free_joint_q_start"] = -1

            # Current qpos
            try:
                qpos = entity.get_qpos()
                if hasattr(qpos, "cpu"):
                    qpos = qpos.cpu().numpy()
                info["qpos"] = _sanitize_floats(qpos).tolist()
            except Exception:
                pass

        entities.append(info)

    # Initial visualization state from the rasterizer context.
    # Falls back to empty dict when pyrender context is unavailable.
    vis_state = {}
    try:
        ctx = scene.visualizer._rasterizer._context
        vis_state = {
            "shadows": bool(ctx.shadow),
            "world_frame": bool(ctx.world_frame_shown),
            "link_frame": bool(ctx.link_frame_shown),
            "link_frame_size": float(getattr(ctx, "link_frame_size", 0.1)),
            "camera_frustum": bool(ctx.camera_frustum_shown),
            "face_normals": False,
            "vertex_normals": False,
            "wireframe": False,
            "orthographic": False,
        }
    except Exception:
        pass

    # Camera state from the first visualizer camera
    camera_state = {}
    try:
        camera = scene.visualizer.cameras[0]
        pos = camera.pos
        lookat = camera.lookat
        fov = camera.fov if hasattr(camera, "fov") else 30.0
        camera_state = {
            "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
            "lookat": [float(lookat[0]), float(lookat[1]), float(lookat[2])],
            "fov": float(fov),
        }
    except Exception:
        pass

    # Number of environments
    n_envs = getattr(scene, "n_envs", 1)

    return {
        "type": MsgType.SCENE_INFO.value,
        "entities": entities,
        "vis_state": vis_state,
        "camera_state": camera_state,
        "n_envs": n_envs,
    }


def build_state_update(sim_time=0.0, step=0, fps=0.0, paused=False, camera_state=None):
    """Build a STATE_UPDATE message with current simulation status."""
    msg = {
        "type": MsgType.STATE_UPDATE.value,
        "time": sim_time,
        "step": step,
        "fps": fps,
        "paused": paused,
    }
    if camera_state is not None:
        msg["camera_state"] = camera_state
    return msg
