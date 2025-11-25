import base64
import json
import math
import os
import random
import xml.etree.ElementTree as ET

import genesis as gs
import genesis.utils.geom as gu
import numpy as np
import torch
from genesis.utils.misc import ti_to_torch
from trimesh.visual.color import ColorVisuals

CURRENT_SCENE_DESCRIPTION_VERSION = 1
DEFAULT_ASSET_ROOT_PATH = gs.utils.get_assets_dir()
get_rel_asset_path = lambda x: os.path.relpath(x, DEFAULT_ASSET_ROOT_PATH) if os.path.isabs(x) else x


# Helper functions
def _make_tensor(data, *, dtype: torch.dtype = torch.float32):
    return torch.tensor(data, dtype=dtype, device=gs.device)


def _pos_to_y_up(pos):
    # Swizzle to (X, Z, -Y)
    if isinstance(pos, tuple):
        return np.array([pos[0], pos[2], -pos[1]])
    else:
        pos = torch.as_tensor(pos)
        return torch.stack([pos[..., 0], pos[..., 2], -pos[..., 1]], dim=-1)


def _quat_to_y_up(quat, convert_to_y_up=True):
    # convert_to_y_up can be a single boolean or a list of booleans
    # quat shape: (..., n_vgeoms, 4) where n_vgeoms is the -2 dimension
    # convert_to_y_up shape: (n_vgeoms,)
    # Create mask for indices to convert to y-up (where convert_to_y_up = True)
    quat = torch.as_tensor(quat)
    if isinstance(quat, tuple):
        assert isinstance(convert_to_y_up, bool), "convert_to_y_up must be a single boolean if quat is a tuple"
        if convert_to_y_up:
            x, y, z, w = quat
            divisor = math.sqrt(2.0)
            quat = np.array([(x + w) / divisor, (x - w) / divisor, (z + y) / divisor, (z - y) / divisor])
        return _wxyz_to_xyzw(quat)
    elif quat.ndim == 1:
        assert isinstance(convert_to_y_up, bool), "convert_to_y_up must be a single boolean if quat is 1D"
        if convert_to_y_up:
            w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
            # This is the same as transforming the quat with [0.7071068, -0.7071068, 0, 0]
            quat = torch.stack([x + w, x - w, z + y, z - y], dim=-1) / math.sqrt(2.0)
        return _wxyz_to_xyzw(quat)
    else:
        convert_to_y_up_mask = torch.as_tensor(convert_to_y_up, dtype=torch.bool)

        # Assert that the batch dimension matches
        assert (
            len(convert_to_y_up_mask) == quat.shape[0]
        ), f"convert_to_y_up_mask length ({len(convert_to_y_up_mask)}) must match quat batch dimension ({quat.shape[0]})"

        # Apply transformation only where convert_to_y_up is True
        result = quat.clone()

        # Apply transformation to all positions to convert to y-up at once
        if convert_to_y_up_mask.any():
            # Get indices where transformation should be applied
            convert_to_y_up_indices = torch.where(convert_to_y_up_mask)[0]

            # Apply transformation to all positions to convert to y-up
            w = quat[convert_to_y_up_indices, ..., 0]
            x = quat[convert_to_y_up_indices, ..., 1]
            y = quat[convert_to_y_up_indices, ..., 2]
            z = quat[convert_to_y_up_indices, ..., 3]
            result[convert_to_y_up_indices, :] = torch.stack([x + w, x - w, z + y, z - y], dim=-1) / math.sqrt(2.0)

        return _wxyz_to_xyzw(result)


def _geom_pos_to_y_up(pos):
    return _pos_to_y_up(pos)


def _geom_quat_to_y_up(quat, convert_to_y_up):
    return _quat_to_y_up(quat, convert_to_y_up)


def _camera_pos_to_y_up(pos):
    return _pos_to_y_up(pos)


def _camera_quat_to_y_up(quat):
    if isinstance(quat, tuple):
        x, y, z, w = quat
        divisor = math.sqrt(2.0)
        quat = np.array([(x + w) / divisor, (w - x) / divisor, (z + y) / divisor, (z - y) / divisor])
        return _wxyz_to_xyzw(quat)
    elif isinstance(quat, torch.Tensor) or isinstance(quat, np.ndarray):
        quat = torch.as_tensor(quat)
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        # This is the same as transforming the quat with [0.7071068, -0.7071068, 0, 0]
        quat = torch.stack([x + w, w - x, z + y, z - y], dim=-1) / math.sqrt(2.0)
        return _wxyz_to_xyzw(quat)
    else:
        gs.raise_exception(f"Invalid quat type: {type(quat)}")


def _light_pos_to_y_up(pos):
    return _pos_to_y_up(pos).tolist()


def _camera_T_to_quat_y_up(T):
    R = T[:3, :3]
    quat = gu.R_to_quat(R)
    return _camera_quat_to_y_up(quat)


def _wxyz_to_xyzw(wxyz):
    if isinstance(wxyz, tuple):
        return np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    else:
        # Handle multi-dimensional tensors by indexing along the last dimension
        return torch.stack([wxyz[..., 1], wxyz[..., 2], wxyz[..., 3], wxyz[..., 0]], dim=-1)


def should_export_at_geom_level(entity):
    if isinstance(entity.morph, gs.morphs.Mesh):
        file_extension = os.path.splitext(entity.morph.file)[1]
        if file_extension in gs.morphs.GLTF_FORMATS or file_extension in gs.morphs.USD_FORMATS:
            return False

    if isinstance(entity.morph, gs.morphs.Primitive):
        return False

    # return True by default
    return True


def _build_convert_to_y_up_list(entities):
    convert_to_y_up_list = []
    for entity in entities:
        if isinstance(entity.morph, gs.morphs.Primitive):
            # For primitives, append 1 False
            convert_to_y_up_list.append(False)
        else:
            # Otherwise, append N Trues where N is number of vgeoms
            n_vgeoms = len(entity.vgeoms) if hasattr(entity, "vgeoms") else 1
            convert_to_y_up_list.extend([True] * n_vgeoms)
    return np.array(convert_to_y_up_list)


def _build_mesh_transform_idx(scene):
    entity_start_idx = 0
    idx = []
    for entity in scene.entities:
        if should_export_at_geom_level(entity):
            idx += list(range(entity_start_idx, entity_start_idx + entity.n_vgeoms))
        else:
            idx += [entity_start_idx]
        entity_start_idx += entity.n_vgeoms
    return _make_tensor(idx, dtype=gs.tc_int)


def _get_basename_no_extension(path):
    return os.path.splitext(os.path.basename(path))[0]


class SceneDescriptionFrame:
    def __init__(self):
        self._mesh_transforms = None
        self._camera_transforms = None
        self._light_transforms = None


class SceneDescriptionExporter:
    def __init__(self, scene, cameras=None, lights=None):
        self._scene = scene
        # If cameras are passed, use them; otherwise fall back to visualizer cameras
        if cameras is None:
            self._cameras = gs.List([camera for camera in scene.visualizer.cameras if not camera.debug])
            self._extra_camera_defs = []
        else:
            cam_objs = [cam for cam in cameras if isinstance(cam, gs.vis.camera.Camera)]
            self._cameras = gs.List(cam_objs)
            self._extra_camera_defs = [cam for cam in cameras if not isinstance(cam, gs.vis.camera.Camera)]
        # Optional lights provided by caller
        self._lights = lights
        self._json_content = dict()
        self._generate_initial_scene_description()
        self._mesh_transform_idx = _build_mesh_transform_idx(scene)

    def _generate_initial_scene_description(self, num_envs=1, asset_root_path=DEFAULT_ASSET_ROOT_PATH):
        # headers
        self._json_content = {
            "version": CURRENT_SCENE_DESCRIPTION_VERSION,
            "num_environments": num_envs,
            "asset_root_path": asset_root_path,
            "frame_time": self._scene.sim_options.dt,
            "mesh_entities": [],
            "camera_entities": [],
            "light_entities": [],
            "scene_animation": [],
        }

        # mesh entities
        for entity in self._scene.entities:
            self._add_entity_to_json(self._json_content["mesh_entities"], entity)
        self._convert_to_y_up_list = _build_convert_to_y_up_list(self._scene.entities)

        # camera entities
        for camera in self._cameras:
            self._add_camera_to_json(self._json_content["camera_entities"], camera)
        # extra camera definitions (dict-like)
        for cam_def in self._extra_camera_defs:
            self._add_apollo_camera_to_json(self._json_content["camera_entities"], cam_def)

        # light entities passed by caller
        if self._lights is not None:
            for light in self._lights:
                self._add_apollo_renderer_light_to_json(self._json_content["light_entities"], light)

        # environment map
        # TODO: Support environment map for Apollo renderer
        if self._scene.visualizer.raytracer is not None:
            self._add_environment_map_to_json(self._json_content["light_entities"])

    def capture_frame(self):
        frame = dict()

        frame["mesh_transforms"] = self._get_mesh_transforms()
        frame["camera_transforms"] = self._get_camera_transforms()

        self._json_content["scene_animation"].append(frame)

    def serialize_transforms(self, transforms):
        # Flatten and then encode in base64
        return base64.b64encode(transforms.cpu().numpy().flatten().tobytes()).decode("utf-8")

    def _get_vgeoms_pos_quat(self):
        vgeoms_state_pos = ti_to_torch(self._scene.rigid_solver.vgeoms_state.pos)
        vgeoms_state_quat = ti_to_torch(self._scene.rigid_solver.vgeoms_state.quat)

        pos = vgeoms_state_pos.clone()
        quat = vgeoms_state_quat.clone()

        # Evaluate index mask
        is_mjcf_vgeom = _make_tensor(
            [isinstance(vgeom.entity.morph, gs.morphs.MJCF) for vgeom in self._scene.rigid_solver.vgeoms],
            dtype=torch.bool,
        )
        mjcf_indices = torch.where(is_mjcf_vgeom)[0]

        # Convert to tensors for efficient indexing
        if mjcf_indices.any():
            # Collect indices for vgeoms that are MJCF
            link_indices = _make_tensor([vgeom.link.idx for vgeom in self._scene.rigid_solver.vgeoms], dtype=torch.int)
            link_indices = link_indices[mjcf_indices]

            # Update MJCF transforms
            links_state_pos = self._scene.rigid_solver.get_links_pos()
            links_state_quat = self._scene.rigid_solver.get_links_quat()
            pos[mjcf_indices] = links_state_pos[link_indices].unsqueeze(1)
            quat[mjcf_indices] = links_state_quat[link_indices].unsqueeze(1)

        return pos, quat

    def _get_mesh_transforms(self):
        pos, quat = self._get_vgeoms_pos_quat()

        # Convert to y-up
        pos = _geom_pos_to_y_up(pos)
        quat = _geom_quat_to_y_up(quat, self._convert_to_y_up_list)

        # Select transforms, by merging transforms of entities that don't expand in scene description
        pos = torch.index_select(pos, -3, self._mesh_transform_idx).contiguous()
        quat = torch.index_select(quat, -3, self._mesh_transform_idx).contiguous()

        transforms = dict()
        transforms["pos"] = self.serialize_transforms(pos)
        transforms["quat"] = self.serialize_transforms(quat)
        return transforms

    def _get_camera_transforms(self):
        transforms = dict()
        pos = torch.stack([camera.get_pos() for camera in self._cameras])
        pos = _camera_pos_to_y_up(pos)
        transforms["pos"] = self.serialize_transforms(pos)

        quat = torch.stack([camera.get_quat() for camera in self._cameras])
        # No need to convert to y-up space, since the quat returned by get_quat is already in y-up space
        # TODO: Consider storing the z-up quat in the camera objects, and only convert on demand
        quat = _camera_quat_to_y_up(quat)
        transforms["quat"] = self.serialize_transforms(quat)
        return transforms

    def export_to_file(self, export_path):
        if export_path is not None:
            dir_path = os.path.dirname(export_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(export_path, "w") as f:
                json.dump(self._json_content, f, indent=4)

    def export_to_json_str(self):
        return json.dumps(self._json_content, indent=4)

    # Meshes
    def _set_mesh_extra_properties(self, entity, properties):
        morph = entity.morph
        if isinstance(morph, gs.morphs.Plane):
            properties["tiling"] = tuple(p / t for p, t in zip(morph.plane_size, morph.tile_size, strict=False))

    def _add_entity_to_json(self, entities_array, entity):
        if should_export_at_geom_level(entity):
            self._add_entity_geoms_to_json(entities_array, entity)
        else:
            self._add_raw_entity_to_json(entities_array, entity)

    def _get_entity_name(self, entity):
        if isinstance(entity.morph, gs.morphs.FileMorph):
            return _get_basename_no_extension(entity.morph.file)
        elif isinstance(entity.morph, gs.morphs.Primitive):
            return type(entity.morph).__name__
        else:
            return None

    def _get_entity_type(self, entity):
        if isinstance(entity.morph, gs.morphs.FileMorph):
            return "mesh"
        elif isinstance(entity.morph, gs.morphs.Box):
            return "box"
        elif isinstance(entity.morph, gs.morphs.Cylinder):
            return "cylinder"
        elif isinstance(entity.morph, gs.morphs.Sphere):
            return "sphere"
        elif isinstance(entity.morph, gs.morphs.Plane):
            return "plane"
        else:
            return "unknown"

    def _get_primitive_scale(self, morph):
        if isinstance(morph, gs.morphs.Box):
            return (morph.size[0], morph.size[1], morph.size[2])
        elif isinstance(morph, gs.morphs.Sphere):
            return (morph.radius, morph.radius, morph.radius)
        elif isinstance(morph, gs.morphs.Cylinder):
            return (morph.radius, morph.radius, morph.height)
        elif isinstance(morph, gs.morphs.Plane):
            return (morph.plane_size[0], morph.plane_size[1], 1.0)
        else:
            return (1.0, 1.0, 1.0)

    def _get_file_morph_scale(self, morph):
        if isinstance(morph.scale, float):
            return (morph.scale, morph.scale, morph.scale)
        elif isinstance(morph.scale, tuple) and len(morph.scale) == 3:
            return morph.scale

    def _get_entity_scale(self, entity):
        if isinstance(entity.morph, gs.morphs.Primitive):
            scale = self._get_primitive_scale(entity.morph)
        elif isinstance(entity.morph, gs.morphs.FileMorph):
            scale = self._get_file_morph_scale(entity.morph)
        else:
            scale = (1.0, 1.0, 1.0)

        # Swizzle y and z for now, until Apollo is z-up
        return (scale[0], scale[2], scale[1])

    def _get_entity_uri(self, entity):
        if isinstance(entity.morph, gs.morphs.FileMorph):
            return get_rel_asset_path(entity.morph.file)
        else:
            return None

    def _get_vgeom_name(self, vgeom):
        if "mesh_path" in vgeom.metadata:
            return _get_basename_no_extension(vgeom.metadata["mesh_path"])
        elif vgeom.type == gs.GEOM_TYPE.BOX:
            return "box"
        elif vgeom.type == gs.GEOM_TYPE.PLANE:
            return "plane"
        elif vgeom.type == gs.GEOM_TYPE.CYLINDER:
            return "cylinder"
        elif vgeom.type == gs.GEOM_TYPE.CAPSULE:
            return "capsule"
        elif vgeom.type == gs.GEOM_TYPE.SPHERE:
            return "sphere"
        else:
            return None

    def _get_vgeom_uri_mjcf(self, vgeom):
        # get meshdir and assetdir from mujoco file
        entity_file_path = vgeom.entity.morph.file
        xml_root = ET.parse(entity_file_path).getroot()
        compiler = xml_root.find("compiler")
        meshdir = None
        assetdir = None
        if compiler is not None:
            meshdir = compiler.get("meshdir")
            assetdir = compiler.get("assetdir")

        entity_file_abs_dir = os.path.dirname(entity_file_path)
        mesh_path = vgeom.metadata["mesh_path"]

        # Search for the mesh in the meshdir or assetdir
        if meshdir is not None:
            abs_mesh_path = os.path.join(entity_file_abs_dir, meshdir, mesh_path)
        elif assetdir is not None:
            abs_mesh_path = os.path.join(entity_file_abs_dir, assetdir, mesh_path)
        else:
            abs_mesh_path = os.path.join(entity_file_abs_dir, mesh_path)
        if os.path.exists(abs_mesh_path):
            return get_rel_asset_path(abs_mesh_path)

        return None

    def _get_vgeom_uri(self, vgeom):
        if "mesh_path" in vgeom.metadata:
            mesh_path = vgeom.metadata["mesh_path"]
            if isinstance(vgeom.entity.morph, gs.morphs.MJCF):
                return self._get_vgeom_uri_mjcf(vgeom)
            else:
                return get_rel_asset_path(mesh_path)
        else:
            return None

    def _get_material_property_override(self, material_override, surface, property_name):
        if hasattr(surface, property_name) and getattr(surface, property_name) is not None:
            material_override[property_name] = getattr(surface, property_name)
            return

    def _get_material_texture_override(self, material_override, surface, texture_name):
        if hasattr(surface, texture_name) and getattr(surface, texture_name) is not None:
            texture = getattr(surface, texture_name)
            if (
                isinstance(texture, gs.textures.ImageTexture)
                and hasattr(texture, "image_path")
                and texture.image_path is not None
            ):
                material_override[texture_name] = get_rel_asset_path(texture.image_path)
            return

    def _get_entity_material_override(self, entity):
        surface = entity.surface
        material_override = {}

        # Define attribute pairs as (attribute_name, property_name) tuples
        MATERIAL_PROPERTIES = [
            # Material properties
            "color",
            "opacity",
            "roughness",
            "metallic",
            "emissive",
            "ior",
            "doublesided",
            "subsurface",
            "thickness",
            "metal_type",
        ]

        MATERIAL_TEXTURES = [
            # Texture properties
            "diffuse_texture",
            "opacity_texture",
            "roughness_texture",
            "metallic_texture",
            "normal_texture",
            "emissive_texture",
            "specular_texture",
            "transmission_texture",
            "thickness_texture",
        ]

        # Update all attributes
        for property_name in MATERIAL_PROPERTIES:
            self._get_material_property_override(material_override, surface, property_name)

        for texture_name in MATERIAL_TEXTURES:
            self._get_material_texture_override(material_override, surface, texture_name)

        # Special case for plane
        if isinstance(entity.morph, gs.morphs.Plane):
            material_override["diffuse_texture"] = "textures/checker.png"

        return material_override

    def _get_vgeom_material_override(self, entity_type, vgeom, entity_material_override):
        # If entity-level material override is not specified,
        # use the geometry-level material override for certain properties
        material_override = entity_material_override

        # Fall back to color from ColorVisuals if color override not specified on entity level
        if "color" not in entity_material_override or "opacity" not in entity_material_override:
            visual = vgeom.get_trimesh().visual
            if isinstance(visual, ColorVisuals):
                geom_color = visual.main_color / 255.0
                if "color" not in entity_material_override:
                    material_override["color"] = geom_color[:3].tolist()
                if "opacity" not in entity_material_override:
                    material_override["opacity"] = geom_color[3]

        return material_override

    def _get_entity_fixed(self, entity):
        if isinstance(entity.morph, gs.morphs.Primitive):
            return entity.morph.fixed
        else:
            return False

    def _get_vgeom_init_pos(self, vgeom, links_pos, vgeoms_pos):
        if isinstance(vgeom.entity.morph, gs.morphs.MJCF):
            return _geom_pos_to_y_up(links_pos[vgeom.link.idx]).tolist()
        else:
            return _geom_pos_to_y_up(vgeoms_pos[vgeom._idx, 0]).tolist()

    def _get_vgeom_init_quat(self, vgeom, links_quat, vgeoms_quat):
        convert_to_y_up = not isinstance(vgeom.entity.morph, gs.morphs.Primitive)
        if isinstance(vgeom.entity.morph, gs.morphs.MJCF):
            return _geom_quat_to_y_up(links_quat[vgeom.link.idx], convert_to_y_up).tolist()
        else:
            return _geom_quat_to_y_up(vgeoms_quat[vgeom._idx, 0], convert_to_y_up).tolist()

    def _get_vgeom_type(self, vgeom, entity_type):
        if not hasattr(vgeom, "type"):
            return entity_type
        vgeom_type = vgeom.type
        if vgeom_type == gs.GEOM_TYPE.BOX:
            return "box"
        elif vgeom_type == gs.GEOM_TYPE.PLANE:
            return "plane"
        elif vgeom_type == gs.GEOM_TYPE.CYLINDER:
            return "cylinder"
        elif vgeom_type == gs.GEOM_TYPE.CAPSULE:
            return "capsule"
        elif vgeom_type == gs.GEOM_TYPE.SPHERE:
            return "sphere"
        else:
            return entity_type

    def _get_vgeom_scale(self, vgeom, entity_scale):
        if not hasattr(vgeom, "type"):
            return entity_scale
        vgeom_type = vgeom.type
        if vgeom_type == gs.GEOM_TYPE.BOX:
            extents = vgeom.data
            return (
                entity_scale[0] * extents[0],
                entity_scale[1] * extents[1],
                entity_scale[2] * extents[2],
            )
        elif vgeom_type == gs.GEOM_TYPE.PLANE:
            return (entity_scale[3], entity_scale[4], 1.0)
        elif vgeom_type == gs.GEOM_TYPE.CYLINDER:
            radius = vgeom.data[0]
            height = vgeom.data[1]
            return (entity_scale[0] * radius, entity_scale[1] * radius, entity_scale[2] * height)
        elif vgeom_type == gs.GEOM_TYPE.CAPSULE:
            radius = vgeom.data[0]
            height = vgeom.data[1]
            return (entity_scale[0] * radius, entity_scale[1] * radius, entity_scale[2] * height)
        elif vgeom_type == gs.GEOM_TYPE.SPHERE:
            radius = vgeom.data[0]
            return (entity_scale[0] * radius, entity_scale[1] * radius, entity_scale[2] * radius)
        else:
            return entity_scale

    def _add_entity_geoms_to_json(self, entities_array, entity):
        # Skip if entity is not a RigidEntity
        if not isinstance(entity, gs.engine.entities.RigidEntity):
            return

        entity_type = self._get_entity_type(entity)
        entity_scale = self._get_entity_scale(entity)
        entity_material_override = self._get_entity_material_override(entity)
        entity_fixed = self._get_entity_fixed(entity)
        links_state_pos = self._scene.rigid_solver.get_links_pos().cpu().numpy()
        links_state_quat = self._scene.rigid_solver.get_links_quat().cpu().numpy()
        vgeoms_state_pos = ti_to_torch(self._scene.rigid_solver.vgeoms_state.pos).cpu().numpy()
        vgeoms_state_quat = ti_to_torch(self._scene.rigid_solver.vgeoms_state.quat).cpu().numpy()
        for vgeom in entity.vgeoms:
            vgeom_dict = {}
            vgeom_name = self._get_vgeom_name(vgeom)
            if vgeom_name is not None:
                vgeom_dict["name"] = vgeom_name
            vgeom_dict["entity_type"] = self._get_vgeom_type(vgeom, entity_type)
            # TODO: Batch vgeoms
            vgeom_dict["position"] = self._get_vgeom_init_pos(vgeom, links_state_pos, vgeoms_state_pos)
            vgeom_dict["rotation"] = self._get_vgeom_init_quat(vgeom, links_state_quat, vgeoms_state_quat)
            vgeom_dict["scale"] = self._get_vgeom_scale(vgeom, entity_scale)
            uri = self._get_vgeom_uri(vgeom)
            if uri is not None:
                vgeom_dict["uri"] = uri
            self._set_mesh_extra_properties(entity, vgeom_dict)
            vgeom_dict["material_override"] = self._get_vgeom_material_override(
                entity_type, vgeom, entity_material_override
            )
            vgeom_dict["fixed"] = entity_fixed
            entities_array.append(vgeom_dict)

    def _get_entity_position(self, entity):
        return _geom_pos_to_y_up(entity.morph.pos).tolist()

    def _get_entity_rotation(self, entity):
        convert_to_y_up = not isinstance(entity.morph, gs.morphs.Primitive)
        return _quat_to_y_up(entity.morph.quat, convert_to_y_up).tolist()

    def _add_raw_entity_to_json(self, entities_array, entity):
        # Skip if entity is not a RigidEntity
        if not isinstance(entity, gs.engine.entities.RigidEntity):
            return

        entity_name = self._get_entity_name(entity)
        entity_type = self._get_entity_type(entity)
        entity_scale = self._get_entity_scale(entity)
        entity_material_override = self._get_entity_material_override(entity)
        entity_fixed = self._get_entity_fixed(entity)
        entity_dict = {}
        if entity_name is not None:
            entity_dict["name"] = entity_name
        entity_dict["entity_type"] = entity_type
        entity_dict["position"] = self._get_entity_position(entity)
        entity_dict["rotation"] = self._get_entity_rotation(entity)
        entity_dict["scale"] = entity_scale
        entity_dict["fixed"] = entity_fixed
        uri = self._get_entity_uri(entity)
        if uri is not None:
            entity_dict["uri"] = uri
        self._set_mesh_extra_properties(entity, entity_dict)
        entity_dict["material_override"] = entity_material_override
        entities_array.append(entity_dict)

    # Cameras
    def _add_camera_to_json(self, cameras_array, camera):
        # Skip if camera is not a CameraEntity
        if not isinstance(camera, gs.vis.camera.Camera):
            return

        if camera.debug:
            return

        camera_dict = {}
        camera_dict["position"] = self._get_camera_position(camera)
        camera_dict["rotation"] = self._get_camera_rotation(camera)
        camera_dict["fov_y"] = camera.fov
        camera_dict["aperture"] = camera.aperture
        camera_dict["near_plane"] = camera.near
        camera_dict["far_plane"] = camera.far
        camera_dict["resolution"] = camera.res
        camera_dict["focal_length"] = camera.focal_len
        camera_dict["samples_per_pixel"] = camera.spp
        camera_dict["denoise"] = camera.denoise
        cameras_array.append(camera_dict)

    def _add_apollo_camera_to_json(self, cameras_array, cam_def):
        # cam_def is a dict-like: {pos, lookat, up, fov, aperture, near, far, res, spp, denoise}
        pos = torch.as_tensor(cam_def["pos"], dtype=gs.tc_float, device=gs.device)
        lookat = torch.as_tensor(cam_def["lookat"], dtype=gs.tc_float, device=gs.device)
        up = torch.as_tensor(cam_def["up"], dtype=gs.tc_float, device=gs.device)
        T = gu.pos_lookat_up_to_T(pos, lookat, up)
        camera_dict = {}
        camera_dict["position"] = _camera_pos_to_y_up(pos).cpu().tolist()
        camera_dict["rotation"] = _camera_T_to_quat_y_up(T).tolist()
        camera_dict["fov_y"] = cam_def["fov"]
        camera_dict["aperture"] = cam_def["aperture"]
        camera_dict["near_plane"] = cam_def["near"]
        camera_dict["far_plane"] = cam_def["far"]
        camera_dict["resolution"] = cam_def["res"]
        camera_dict["focal_length"] = cam_def.get("focal_length", None)
        camera_dict["samples_per_pixel"] = cam_def["spp"]
        camera_dict["denoise"] = cam_def["denoise"]
        cameras_array.append(camera_dict)

    def _get_camera_position(self, camera):
        return _camera_pos_to_y_up(camera._initial_pos).cpu().tolist()

    def _get_camera_rotation(self, camera):
        if camera._initial_transform is not None:
            transform = camera._initial_transform
        else:
            transform = gu.pos_lookat_up_to_T(camera._initial_pos, camera._initial_lookat, camera._initial_up)

        return _camera_T_to_quat_y_up(transform).tolist()

    # Lights
    def _add_apollo_renderer_light_to_json(self, lights_array, light):
        # Accept any Light-like object with required attributes (duck typing)
        required_attrs = ["directional", "dir", "color", "intensity", "castshadow"]
        if not all(hasattr(light, a) for a in required_attrs):
            return

        light_dict = {}
        if light.directional:
            light_dict["type"] = "directional"
            light_dict["direction"] = _light_pos_to_y_up(light.dir)
        else:
            light_dict["type"] = "spot"
            if hasattr(light, "pos"):
                light_dict["position"] = _light_pos_to_y_up(light.pos)
            light_dict["direction"] = _light_pos_to_y_up(light.dir)
            light_dict["radius"] = random.randint(10, 50)  # Temporary random radius
            light_dict["attenuation"] = getattr(light, "attenuation", (1.0, 0.0, 0.0))
            cutoff_deg = getattr(light, "cutoffDeg", 45.0)
            light_dict["inner_cone_angle"] = cutoff_deg
            light_dict["outer_cone_angle"] = cutoff_deg
            light_dict["falloff"] = 1.0
        light_dict["color"] = light.color
        light_dict["intensity"] = light.intensity
        light_dict["cast_shadow"] = light.castshadow
        lights_array.append(light_dict)

    def _add_environment_map_to_json(self, lights_array):
        # TODO: Fix this to align with the new schema
        if self._scene.visualizer.raytracer is None:
            return

        light_dict = {}
        light_dict["type"] = "environment"
        surface = self._scene.visualizer.raytracer.env_sphere.surface
        if surface is not None:
            full_texture_path = surface.emissive_texture.image_path
            light_dict["texture"] = get_rel_asset_path(full_texture_path)
        light_dict["rotation"] = self._scene.visualizer.raytracer.env_sphere.quat.tolist()
        light_dict["intensity"] = 1.0
        lights_array.append(light_dict)
