from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional

import genesis as gs
import numpy as np
import torch
from genesis.engine.sensors.sensor_manager import register_sensor
from genesis.options.sensors.options import SensorOptions
from genesis.engine.sensors.base_sensor import SharedSensorMetadata
from genesis.options.sensors.options import RigidSensorOptionsMixin
from genesis.utils.geom import T_to_trans_quat, pos_lookat_up_to_T
from genesis.utils.misc import ti_to_torch

# Light class moved from _ref.py reference implementation
import math


class Light:
    def __init__(self, pos, dir, color, intensity, directional, castshadow, cutoff, attenuation):
        self._pos = pos
        norm = math.sqrt(sum(x * x for x in dir))
        self._dir = tuple(x / norm for x in dir)
        self._color = color
        self._intensity = intensity
        self._directional = directional
        self._castshadow = castshadow
        self._cutoff = cutoff
        self._attenuation = attenuation

    @property
    def pos(self):
        return self._pos

    @property
    def dir(self):
        return self._dir

    @property
    def color(self):
        return self._color

    @property
    def intensity(self):
        return self._intensity

    @property
    def directional(self):
        return self._directional

    @property
    def castshadow(self):
        return self._castshadow

    @property
    def cutoffRad(self):
        return math.radians(self._cutoff)

    @property
    def cutoffDeg(self):
        return self._cutoff

    @property
    def attenuation(self):
        return self._attenuation


from .scene_exporter import (
    SceneDescriptionExporter,
    _build_mesh_transform_idx,
    _camera_quat_to_y_up,
    _pos_to_y_up,
    _quat_to_y_up,
)

try:
    from gs_apollo import ApolloRenderer as ApolloRendererImpl
except ImportError as e:
    ApolloRendererImpl = None
    _APOLLO_IMPORT_ERROR = e
else:
    _APOLLO_IMPORT_ERROR = None


# ----------------------------- Options / Data / Shared State -----------------------------


class ApolloCameraOptions(RigidSensorOptionsMixin, SensorOptions):
    """Options for the Apollo Camera Sensor plugin. These mirror the renderer options from the reference file."""

    # Per-camera parameters (mirroring Scene.add_camera subset)
    model: str = "pinhole"
    res: tuple[int, int] = (512, 512)
    pos: tuple[float, float, float] = (
        8.5,
        0.0,
        1.5,
    )  # Camera position offset. If attached to a link, this is relative to the link frame. If not attached, this is relative to the world origin.
    lookat: tuple[float, float, float] = (3.0, 0.0, 0.7)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)

    # Override to make entity_idx optional for static cameras
    entity_idx: Optional[int] = None
    lights: list[dict] = []
    fov: float = 60.0
    GUI: bool = True
    spp: int = 16
    denoise: bool | None = None
    near: float = 0.1
    far: float = 100.0

    app_mode: str = "batch_render"  # "batch_render" | "interactive" | "rendering_server"
    render_mode: str = "forward"  # "forward" | "pt_fast" | "pt_ref" | "debug"
    debug_view: str = "meshlet"  # used when render_mode == "debug"
    max_pt_depth: int = 2
    scene_description_export_path: str | None = None
    capture_animation: bool = False
    window_size: tuple[int, int] = (1024, 1024)

    # SensorManager integration knobs
    update_ground_truth_only: bool = True
    draw_debug: bool = False

    # SensorOptions API compatibility layer
    def validate(self, scene: gs.Scene):
        # Basic checks that match the reference renderer expectations
        if self.app_mode not in ("batch_render", "interactive", "rendering_server"):
            gs.raise_exception(f"Invalid app_mode: {self.app_mode}")
        if self.render_mode not in ("forward", "pt_fast", "pt_ref", "debug"):
            gs.raise_exception(f"Invalid render_mode: {self.render_mode}")
        if not isinstance(self.window_size, (tuple, list)) or len(self.window_size) != 2:
            gs.raise_exception("window_size must be a (width, height) tuple")
        if not (isinstance(self.res, (tuple, list)) and len(self.res) == 2):
            gs.raise_exception("res must be a (width, height) tuple")
        if not (isinstance(self.pos, (tuple, list)) and len(self.pos) == 3):
            gs.raise_exception("pos must be a (x, y, z) tuple")
        if not (isinstance(self.lookat, (tuple, list)) and len(self.lookat) == 3):
            gs.raise_exception("lookat must be a (x, y, z) tuple")
        if not (isinstance(self.up, (tuple, list)) and len(self.up) == 3):
            gs.raise_exception("up must be a (x, y, z) tuple")
        if not isinstance(self.lights, list):
            gs.raise_exception(f"lights must be a list, got: {type(self.lights)}")
        for i, light in enumerate(self.lights):
            if not isinstance(light, dict):
                gs.raise_exception(f"lights[{i}] must be a dict, got: {type(light)}")
        # Must have at least one light and one non-debug camera for meaningful rendering, but we allow building to proceed
        # so the user can add them before the first render call.
        return None


class ApolloCameraData(NamedTuple):
    rgb: np.ndarray | None
    depth: np.ndarray | None
    segmentation: np.ndarray | None
    normal: np.ndarray | None


@dataclass
class ApolloCameraSharedMetadata(SharedSensorMetadata):
    """Shared state for all Apollo camera sensors in a scene."""

    # External renderer and scene exporter
    renderer: Any | None = None
    scene_exporter: SceneDescriptionExporter | None = None

    # Cached scene structure for fast updates
    mesh_transform_idx: torch.Tensor | None = None
    is_mjcf_vgeom: torch.Tensor | None = None
    mjcf_link_indices: torch.Tensor | None = None
    convert_to_y_up_list: list[bool] | None = None

    # Ordered list of sensors (defines camera indices)
    sensors: list | None = None
    lights: Any | None = None
    camera_defs: list[dict[str, Any]] | None = None
    image_cache: dict = None  # {sensor_idx: torch.Tensor with shape (B, H, W, 3)}
    last_render_timestep: int = -1  # Track when Apollo cameras were last updated


# ------------------------------------ Sensor Impl ---------------------------------------


@register_sensor(ApolloCameraOptions, ApolloCameraSharedMetadata, gs.engine.sensors.camera.CameraData)
class ApolloCameraSensor(gs.engine.sensors.camera.BaseCameraSensor):
    """Apollo Camera Sensor plugin.

    This class integrates Apollo renderer into Genesis via SensorManager lifecycle,
    but returns images through a dedicated `render(camera_index)` method instead of
    SensorManager tensor caches (to avoid giant image caches).
    """

    def __init__(
        self,
        options: ApolloCameraOptions,
        idx: int,
        data_cls: type,
        manager: gs.SensorManager,
    ):
        super().__init__(options, idx, data_cls, manager)
        self._camera_idx: int | None = None
        self._current_camera_T: torch.Tensor | None = None  # Current world transform for attachment

    # -------------------------------------- Hooks for BaseCameraSensor --------------------------------------

    def _apply_camera_transform(self, camera_T: torch.Tensor):
        """Store the computed camera transform for attachment-based pose."""
        self._current_camera_T = camera_T.clone()
        self._stale = True

    def _render_current_state(self):
        """Perform the actual render for the current state."""
        scene = self._manager._sim.scene

        # Update visualization (positions, transforms, etc.)
        scene._visualizer.update_visual_states()

        # Update internal time
        self._t = scene.t

        # Prepare transforms
        geom_pos, geom_quat = self._get_geom_pos_quat_numpy(scene, self._shared_metadata.mesh_transform_idx)
        camera_pos, camera_quat = self._get_all_camera_pos_quat_numpy()

        # Render
        rgb = self._shared_metadata.renderer.render(
            self._camera_idx,
            geom_pos,
            geom_quat,
            camera_pos,
            camera_quat,
        )

        # Convert to torch tensor and store in cache
        if isinstance(rgb, torch.Tensor):
            rgb_tensor = rgb.to(dtype=torch.uint8, device=gs.device)
        else:
            rgb_tensor = torch.from_numpy(rgb.copy() if hasattr(rgb, "copy") else rgb).to(
                dtype=torch.uint8, device=gs.device
            )

        # Store in cache
        # rgb shape: (H, W, 3) for single env or needs batching
        n_envs = self._manager._sim._B
        if n_envs == 0:
            # Single environment case - add batch dimension
            self._shared_metadata.image_cache[self._idx][0] = rgb_tensor
        else:
            # TODO: Handle multi-env rendering properly
            # For now, just replicate the single render
            for i in range(n_envs):
                self._shared_metadata.image_cache[self._idx][i] = rgb_tensor

    # Apollo cameras don't support attachment - they use static pos/lookat/up from options

    # -------------------------------------- Lifecycle --------------------------------------

    def build(self):
        if _APOLLO_IMPORT_ERROR is not None:
            gs.raise_exception_from("Failed to import Apollo renderer.", _APOLLO_IMPORT_ERROR)

        if gs.backend != gs.cuda:
            gs.raise_exception("ApolloCameraSensor requires GPU backend.")

        scene = self._manager._sim.scene
        visualizer = scene._visualizer

        # Register this sensor and assign a camera index
        if self._shared_metadata.sensors is None:
            self._shared_metadata.sensors = []
        self._camera_idx = len(self._shared_metadata.sensors)
        self._shared_metadata.sensors.append(self)

        # Initialize shared lights container and add lights from options
        if self._shared_metadata.lights is None:
            self._shared_metadata.lights = gs.List()

        # Add lights from options to shared metadata
        for light_config in self._options.lights:
            self._add_light_to_apollo(light_config)

        # Initialize and register camera definition for exporter
        if self._shared_metadata.camera_defs is None:
            self._shared_metadata.camera_defs = []
        self._shared_metadata.camera_defs.append(
            {
                "pos": self._options.pos,
                "lookat": self._options.lookat,
                "up": self._options.up,
                "fov": self._options.fov,
                "aperture": self._options.aperture if hasattr(self._options, "aperture") else 2.8,
                "near": self._options.near,
                "far": self._options.far,
                "res": self._options.res,
                "spp": self._options.spp,
                "denoise": self._options.denoise if self._options.denoise is not None else False,
            }
        )
        # Camera definitions are passed directly to SceneDescriptionExporter during export

        # Ensure renderer + exporter are initialized once
        if self._shared_metadata.renderer is None:
            # Export scene description
            self._shared_metadata.scene_exporter = SceneDescriptionExporter(
                scene,
                cameras=self._shared_metadata.camera_defs,
                lights=self._shared_metadata.lights,
            )
            scene_description = self._shared_metadata.scene_exporter.export_to_json_str()
            if self._options.scene_description_export_path and self._options.capture_animation:
                self._shared_metadata.scene_exporter.export_to_file(self._options.scene_description_export_path)

            # Determine max resolution across all Apollo camera sensors (using their options)
            all_sensors: list[ApolloCameraSensor] = self._manager._sensors_by_type[type(self)]
            max_resolution = self._get_max_camera_resolution_from_options([s._options for s in all_sensors])

            # Instantiate Apollo renderer
            self._shared_metadata.renderer = ApolloRendererImpl(
                self._options.app_mode,
                self._options.render_mode,
                self._options.debug_view,
                self._options.max_pt_depth,
                max_resolution if self._options.app_mode == "batch_render" else self._options.window_size,
            )
            self._shared_metadata.renderer.load_scene_data(scene_description)
            self._shared_metadata.mesh_transform_idx = _build_mesh_transform_idx(scene)

        # Cache MJCF vgeoms indices
        self._shared_metadata.is_mjcf_vgeom = torch.tensor(
            [isinstance(vgeom.entity.morph, gs.morphs.MJCF) for vgeom in scene.rigid_solver.vgeoms],
            dtype=torch.bool,
            device=gs.device,
        )
        self._shared_metadata.mjcf_link_indices = torch.tensor(
            [vgeom.link.idx for vgeom in scene.rigid_solver.vgeoms],
            dtype=torch.long,
            device=gs.device,
        ).unsqueeze(1)

        # Cache Y-up conversion flags
        self._shared_metadata.convert_to_y_up_list = [
            not isinstance(entity.morph, gs.morphs.Primitive) for entity in scene.entities for vgeom in entity.vgeoms
        ]

        # Initialize image cache for this camera
        if self._shared_metadata.image_cache is None:
            self._shared_metadata.image_cache = {}
        n_envs = self._manager._sim.n_envs
        h, w = self._options.res[1], self._options.res[0]
        self._shared_metadata.image_cache[self._idx] = torch.zeros(
            (max(1, n_envs), h, w, 3), dtype=torch.uint8, device=gs.device
        )

        # If renderer already exists (i.e., additional cameras added), refresh scene description to include all cameras
        if self._shared_metadata.renderer is not None and self._shared_metadata.scene_exporter is not None:
            self._shared_metadata.scene_exporter = SceneDescriptionExporter(
                scene,
                cameras=self._shared_metadata.camera_defs,
                lights=self._shared_metadata.lights,
            )
            scene_description = self._shared_metadata.scene_exporter.export_to_json_str()
            try:
                self._shared_metadata.renderer.unload_scene()
            except Exception:
                pass
            self._shared_metadata.renderer.load_scene_data(scene_description)
            self._shared_metadata.mesh_transform_idx = _build_mesh_transform_idx(scene)

    def destroy(self):
        # Export animation if enabled
        if self._options.capture_animation and self._options.scene_description_export_path:
            if self._shared_metadata.scene_exporter is not None:
                self._shared_metadata.scene_exporter.export_to_file(self._options.scene_description_export_path)

        # Destroy renderer
        if self._shared_metadata.renderer is not None:
            self._shared_metadata.renderer.unload_scene()
            self._shared_metadata.renderer.destroy()
            self._shared_metadata.renderer = None

        # Clear cached state
        self._shared_metadata.scene_exporter = None
        self._shared_metadata.mesh_transform_idx = None
        self._shared_metadata.is_mjcf_vgeom = None
        self._shared_metadata.mjcf_link_indices = None
        self._shared_metadata.convert_to_y_up_list = None
        self._shared_metadata.sensors = None
        self._shared_metadata.image_cache = None

    # --------------------------------------- Lights ---------------------------------------

    # -------------------------------------- Helpers ---------------------------------------

    @staticmethod
    def _get_max_camera_resolution_from_options(
        options_list: Sequence[ApolloCameraOptions],
    ) -> tuple[int, int]:
        if not options_list:
            return 1024, 1024
        return max((opt.res for opt in options_list), key=lambda r: (r[0] * r[1], r[0], r[1]))

    def _overwrite_mjcf_vgeoms_transforms(
        self, scene, geom_pos: torch.Tensor, geom_quat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._shared_metadata.is_mjcf_vgeom is None:
            return geom_pos, geom_quat
        if self._shared_metadata.is_mjcf_vgeom.any():
            links_state_pos = scene.rigid_solver.get_links_pos()
            links_state_quat = scene.rigid_solver.get_links_quat()
            mjcf_link_pos = links_state_pos[self._shared_metadata.mjcf_link_indices]
            mjcf_link_quat = links_state_quat[self._shared_metadata.mjcf_link_indices]
            geom_pos[self._shared_metadata.is_mjcf_vgeom] = mjcf_link_pos[self._shared_metadata.is_mjcf_vgeom]
            geom_quat[self._shared_metadata.is_mjcf_vgeom] = mjcf_link_quat[self._shared_metadata.is_mjcf_vgeom]
        return geom_pos, geom_quat

    def _get_geom_pos_quat_tensor(self, scene, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        geom_pos = ti_to_torch(scene.rigid_solver.vgeoms_state.pos)
        geom_quat = ti_to_torch(scene.rigid_solver.vgeoms_state.quat)
        geom_pos, geom_quat = self._overwrite_mjcf_vgeoms_transforms(scene, geom_pos, geom_quat)

        geom_pos = _pos_to_y_up(geom_pos).transpose(0, 1)
        geom_quat = _quat_to_y_up(geom_quat, self._shared_metadata.convert_to_y_up_list).transpose(0, 1)

        geom_pos = torch.index_select(geom_pos, -2, idx).contiguous()
        geom_quat = torch.index_select(geom_quat, -2, idx).contiguous()
        return geom_pos, geom_quat

    def _get_geom_pos_quat_numpy(self, scene, idx: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        geom_pos, geom_quat = self._get_geom_pos_quat_tensor(scene, idx)
        return geom_pos.cpu().numpy(), geom_quat.cpu().numpy()

    def _get_all_camera_pos_quat_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute camera poses from attachment state or sensor options."""
        sensors = self._shared_metadata.sensors or []
        camera_transforms = []

        for sensor in sensors:
            if sensor._link is not None and sensor._current_camera_T is not None:
                # Use stored attachment-based transform
                camera_T = sensor._current_camera_T
            else:
                # Use static options-based transform (pos is already the offset)
                pos = torch.tensor(sensor._options.pos, dtype=gs.tc_float, device=gs.device)
                lookat = torch.tensor(sensor._options.lookat, dtype=gs.tc_float, device=gs.device)
                up = torch.tensor(sensor._options.up, dtype=gs.tc_float, device=gs.device)
                camera_T = pos_lookat_up_to_T(pos, lookat, up)

            camera_transforms.append(camera_T)

        # Stack all camera transforms
        T = torch.stack(camera_transforms, dim=0)
        camera_pos, camera_quat = T_to_trans_quat(T)
        camera_pos = _pos_to_y_up(camera_pos)
        camera_quat = _camera_quat_to_y_up(camera_quat)

        # Add batch dimension to satisfy Apollo expected (B, N, 3/4) shape
        camera_pos = camera_pos.unsqueeze(0)
        camera_quat = camera_quat.unsqueeze(0)
        return camera_pos.cpu().numpy(), camera_quat.cpu().numpy()

    def _add_light_to_apollo(self, light_config):
        """Add a light to Apollo shared metadata."""
        # Default values for Apollo lights
        pos = light_config.get("pos", (0.0, 0.0, 5.0))
        dir = light_config.get("dir", (0.0, 0.0, -1.0))
        color = light_config.get("color", (1.0, 1.0, 1.0))
        intensity = light_config.get("intensity", 1.0)
        directional = light_config.get("directional", True)
        castshadow = light_config.get("castshadow", True)
        cutoff = light_config.get("cutoff", 45.0)
        attenuation = light_config.get("attenuation", (1.0, 0.0, 0.0))

        self._shared_metadata.lights.append(
            Light(
                pos,
                dir,
                color,
                intensity,
                directional,
                castshadow,
                cutoff,
                attenuation,
            )
        )
