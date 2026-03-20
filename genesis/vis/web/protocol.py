"""
WebSocket message protocol for Genesis Web GUI.

The WebSocket carries two kinds of messages:
- Binary messages (server -> client): JPEG-encoded frames
- Text/JSON messages (bidirectional): control and state messages

All message types are defined as Pydantic models. Run as a script to export JSON Schema:

    python -m genesis.vis.web.protocol --export-schema path/to/schema.json
"""

from __future__ import annotations

import json
import sys
from enum import Enum
from typing import Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, ValidationError

import genesis as gs
from genesis.utils.misc import tensor_to_array
from genesis.vis.scene_ops import build_entity_joint_data


# ---------------------------------------------------------------------------
# Message type enum
# ---------------------------------------------------------------------------


class MsgType(str, Enum):
    """Message types for the WebSocket protocol."""

    # Client -> Server
    SIM_CONTROL = "sim_control"
    CAMERA_UPDATE = "camera_update"
    ENTITY_UPDATE = "entity_update"
    VIS_TOGGLE = "vis_toggle"
    SET_RESOLUTION = "set_resolution"
    SET_TARGET_FPS = "set_target_fps"
    # Server -> Client
    STATE_UPDATE = "state_update"
    SCENE_INFO = "scene_info"


# ---------------------------------------------------------------------------
# Server -> Client models
# ---------------------------------------------------------------------------


class JointInfo(BaseModel):
    name: str
    type: str
    n_dofs: int
    n_qs: int
    dofs_limit: list[list[float]]


class EntityInfo(BaseModel):
    name: str
    idx: int
    vis_mode: str = "visual"
    visualize_contact: bool = False
    n_dofs: Optional[int] = None
    n_qs: Optional[int] = None
    joints: Optional[list[JointInfo]] = None
    q_names: Optional[list[str]] = None
    q_limits_lower: Optional[list[float]] = None
    q_limits_upper: Optional[list[float]] = None
    q_is_quaternion: Optional[list[bool]] = None
    quat_groups: Optional[list[list[int]]] = None
    has_free_joint: Optional[bool] = None
    free_joint_q_start: Optional[int] = None
    qpos: Optional[list[float]] = None


class VisState(BaseModel):
    shadows: bool = False
    world_frame: bool = False
    link_frame: bool = False
    link_frame_size: float = 0.1
    camera_frustum: bool = False
    face_normals: bool = False
    vertex_normals: bool = False
    wireframe: bool = False
    orthographic: bool = False


class CameraState(BaseModel):
    pos: Optional[list[float]] = None
    lookat: Optional[list[float]] = None
    fov: Optional[float] = None
    view_matrix: Optional[list[float]] = None
    proj_matrix: Optional[list[float]] = None


class EntityPosition(BaseModel):
    idx: int
    qpos: list[float]


class SceneInfoMsg(BaseModel):
    type: Literal["scene_info"] = "scene_info"
    entities: list[EntityInfo]
    vis_state: VisState = VisState()
    camera_state: CameraState = CameraState()
    n_envs: int = 1
    sim_dt: float = 0.01


class StateUpdateMsg(BaseModel):
    type: Literal["state_update"] = "state_update"
    time: float = 0.0
    step: int = 0
    fps: float = 0.0
    paused: bool = False
    camera_state: Optional[CameraState] = None
    entity_positions: Optional[list[EntityPosition]] = None


# ---------------------------------------------------------------------------
# Client -> Server models
# ---------------------------------------------------------------------------


class SimControlMsg(BaseModel):
    type: Literal["sim_control"] = "sim_control"
    action: Literal["play", "pause", "step", "reset"]


class CameraUpdateMsg(BaseModel):
    type: Literal["camera_update"] = "camera_update"
    action: Literal["orbit", "pan", "zoom", "set_pose", "set_fov", "reset"]
    # orbit
    d_azimuth: Optional[float] = None
    d_elevation: Optional[float] = None
    # pan
    dx: Optional[float] = None
    dy: Optional[float] = None
    # zoom
    factor: Optional[float] = None
    # set_pose
    pos: Optional[list[float]] = None
    lookat: Optional[list[float]] = None
    # set_fov
    fov: Optional[float] = None


class EntityUpdateMsg(BaseModel):
    type: Literal["entity_update"] = "entity_update"
    entity_idx: int
    # Full qpos update
    qpos: Optional[list[float]] = None
    normalize_quats: Optional[bool] = None
    quat_groups: Optional[list[list[int]]] = None
    # Single DOF update
    dof_idx: Optional[int] = None
    value: Optional[float] = None
    # Vis mode
    vis_mode: Optional[Literal["visual", "collision"]] = None
    # Wireframe
    wireframe: Optional[bool] = None
    # Contact visualization
    contact_viz: Optional[bool] = None


class VisToggleMsg(BaseModel):
    type: Literal["vis_toggle"] = "vis_toggle"
    property: str
    value: Union[bool, float, int, str]


class SetResolutionMsg(BaseModel):
    type: Literal["set_resolution"] = "set_resolution"
    width: int
    height: int


class SetTargetFpsMsg(BaseModel):
    type: Literal["set_target_fps"] = "set_target_fps"
    fps: float


# ---------------------------------------------------------------------------
# Incoming message parsing
# ---------------------------------------------------------------------------

_CLIENT_MODELS = {
    MsgType.SIM_CONTROL: SimControlMsg,
    MsgType.CAMERA_UPDATE: CameraUpdateMsg,
    MsgType.ENTITY_UPDATE: EntityUpdateMsg,
    MsgType.VIS_TOGGLE: VisToggleMsg,
    MsgType.SET_RESOLUTION: SetResolutionMsg,
    MsgType.SET_TARGET_FPS: SetTargetFpsMsg,
}


def parse_client_message(raw: dict) -> BaseModel | None:
    """Validate and parse a raw dict from a WebSocket client.

    Returns a typed Pydantic model instance, or None if the message type
    is unknown or validation fails.
    """
    msg_type = raw.get("type")
    model_cls = _CLIENT_MODELS.get(msg_type)
    if model_cls is None:
        return None
    try:
        return model_cls.model_validate(raw)
    except ValidationError as e:
        if gs.logger is not None:
            gs.logger.debug(f"Protocol validation failed for {msg_type}: {e}")
        return None


# ---------------------------------------------------------------------------
# Scene introspection helpers (build server -> client messages)
# ---------------------------------------------------------------------------


def sanitize_floats(arr):
    """Replace inf/NaN with finite values for JSON serialization."""
    return np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)


def _build_joint_info(joint) -> JointInfo:
    """Extract serializable info from a RigidJoint."""
    dofs_limit = joint.dofs_limit
    if not isinstance(dofs_limit, np.ndarray):
        dofs_limit = tensor_to_array(dofs_limit)
    limits = sanitize_floats(dofs_limit).tolist()
    return JointInfo(name=joint.name, type=joint.type.name, n_dofs=joint.n_dofs, n_qs=joint.n_qs, dofs_limit=limits)


def build_scene_info(scene) -> dict:
    """Build a SCENE_INFO message from the current scene.

    Returns a dict ready for JSON serialization (via model_dump).
    """
    entities = []
    for entity in scene.entities:
        try:
            vis_mode = getattr(entity.surface, "vis_mode", "visual") if hasattr(entity, "surface") else "visual"
        except Exception:
            vis_mode = "visual"

        info = EntityInfo(
            name=entity.name,
            idx=entity.idx,
            vis_mode=vis_mode,
            visualize_contact=getattr(entity, "visualize_contact", False),
        )

        if hasattr(entity, "n_dofs") and entity.n_dofs > 0:
            info.n_dofs = entity.n_dofs
            info.n_qs = entity.n_qs
            try:
                info.joints = [_build_joint_info(j) for j in entity.joints]
            except Exception:
                info.joints = []

            try:
                jdata = build_entity_joint_data(entity)
                info.q_names = jdata["q_names"]
                info.q_limits_lower = jdata["q_limits_lower"]
                info.q_limits_upper = jdata["q_limits_upper"]
                info.q_is_quaternion = jdata["q_is_quaternion"]
                info.quat_groups = jdata["quat_groups"]
                info.has_free_joint = jdata["has_free_joint"]
                info.free_joint_q_start = jdata["free_joint_q_start"]
            except Exception:
                pass

            try:
                qpos = entity.get_qpos()
                if hasattr(qpos, "cpu"):
                    qpos = qpos.cpu().numpy()
                info.qpos = sanitize_floats(qpos).tolist()
            except Exception:
                pass

        entities.append(info)

    vis_state = VisState()
    try:
        ctx = scene.visualizer._rasterizer._context
        vis_state = VisState(
            shadows=bool(ctx.shadow),
            world_frame=bool(ctx.world_frame_shown),
            link_frame=bool(ctx.link_frame_shown),
            link_frame_size=float(getattr(ctx, "link_frame_size", 0.1)),
            camera_frustum=bool(ctx.camera_frustum_shown),
        )
    except Exception:
        pass

    camera_state = CameraState()
    try:
        camera = scene.visualizer.cameras[0]
        pos = camera.pos
        lookat = camera.lookat
        fov = camera.fov if hasattr(camera, "fov") else 30.0
        camera_state = CameraState(
            pos=pos.tolist(),
            lookat=lookat.tolist(),
            fov=float(fov),
        )
    except Exception:
        pass

    msg = SceneInfoMsg(
        entities=entities,
        vis_state=vis_state,
        camera_state=camera_state,
        n_envs=getattr(scene, "n_envs", 1),
        sim_dt=float(getattr(scene, "dt", 0.01)),
    )
    return msg.model_dump(exclude_none=True)


def build_state_update(
    sim_time: float = 0.0,
    step: int = 0,
    fps: float = 0.0,
    paused: bool = False,
    camera_state: dict | None = None,
    entity_positions: list[dict] | None = None,
) -> dict:
    """Build a STATE_UPDATE message with current simulation status."""
    cam = CameraState(**camera_state) if camera_state else None
    positions = [EntityPosition(**ep) for ep in entity_positions] if entity_positions else None
    msg = StateUpdateMsg(time=sim_time, step=step, fps=fps, paused=paused, camera_state=cam, entity_positions=positions)
    return msg.model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# JSON Schema export
# ---------------------------------------------------------------------------

# All models that form the protocol
ALL_MODELS: list[type[BaseModel]] = [
    # Server -> Client
    JointInfo,
    EntityInfo,
    VisState,
    CameraState,
    EntityPosition,
    SceneInfoMsg,
    StateUpdateMsg,
    # Client -> Server
    SimControlMsg,
    CameraUpdateMsg,
    EntityUpdateMsg,
    VisToggleMsg,
    SetResolutionMsg,
    SetTargetFpsMsg,
]


def export_json_schema() -> dict:
    """Export a combined JSON Schema with all protocol models under $defs."""
    defs = {}
    for model in ALL_MODELS:
        schema = model.model_json_schema()
        # Hoist nested $defs to top level
        for key, value in schema.pop("$defs", {}).items():
            defs[key] = value
        defs[model.__name__] = schema
    return {"$defs": defs}


if __name__ == "__main__":
    if "--export-schema" in sys.argv:
        idx = sys.argv.index("--export-schema")
        out_path = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "protocol.schema.json"
        schema = export_json_schema()
        with open(out_path, "w") as f:
            json.dump(schema, f, indent=2)
        print(f"Exported JSON Schema to {out_path}")
    else:
        print("Usage: python -m genesis.vis.web.protocol --export-schema [output_path]")
