import os

import numpy as np
import trimesh
from PIL import Image

import genesis as gs

from .. import mesh as mu
from .usd_stage_utils import (
    decompress_usdz,
    detect_baked_cache,
    get_stage_scale_and_upaxis,
    run_material_baking,
)

try:
    from pxr import Usd, UsdGeom, UsdShade
except ImportError as e:
    raise ImportError(
        "Failed to import USD dependencies. Try installing Genesis with 'usd' optional dependencies."
    ) from e


cs_encode = {
    "raw": "linear",
    "sRGB": "srgb",
    "auto": None,
    "": None,
}


# utils
def get_input_attribute_value(shader: UsdShade.Shader, input_name, input_type=None):
    shader_input = shader.GetInput(input_name)

    if input_type != "value":
        if shader_input.GetPrim().IsValid() and shader_input.HasConnectedSource():
            shader_input_connect, shader_input_name = shader_input.GetConnectedSource()[:2]
            return shader_input_connect.GetPrim(), shader_input_name

    if input_type != "attribute":
        return shader_input.Get(), None
    return None, None


def get_shader(prim: Usd.Prim, output_name):
    if prim.IsA(UsdShade.Shader):
        return UsdShade.Shader(prim)
    elif prim.IsA(UsdShade.NodeGraph):
        return UsdShade.NodeGraph(prim).ComputeOutputSource(output_name)[0]
    else:
        gs.raise_exception(f"Invalid shader type: {prim.GetTypeName()} at {prim.GetPath()}.")


def parse_preview_surface(prim: Usd.Prim, output_name):
    shader = get_shader(prim, output_name)
    shader_id = shader.GetShaderId()

    if shader_id == "UsdPreviewSurface":
        uvname = None

        def parse_component(component_name, component_encode):
            component, component_output = get_input_attribute_value(shader, component_name)
            if component_output is None:  # constant value
                component_factor = component
                component_image = None
                component_uvname = None
            else:  # texture shader
                component_image, component_overencode, component_uvname = parse_preview_surface(
                    component, component_output
                )
                if component_overencode is not None:
                    component_encode = component_overencode
                component_factor = None

            component_texture = mu.create_texture(component_image, component_factor, component_encode)
            return component_texture, component_uvname

        # parse color
        color_texture, color_uvname = parse_component("diffuseColor", "srgb")
        if color_uvname is not None:
            uvname = color_uvname

        # parse opacity
        opacity_texture, opacity_uvname = parse_component("opacity", "linear")
        if opacity_uvname is not None and uvname is None:
            uvname = opacity_uvname
        if opacity_texture is not None:
            alpha_cutoff = get_input_attribute_value(shader, "opacityThreshold", "value")[0]
            opacity_texture.apply_cutoff(alpha_cutoff)

        # parse emissive
        emissive_texture, emissive_uvname = parse_component("emissiveColor", "srgb")
        if emissive_texture is not None and emissive_texture.is_black():
            emissive_texture = None
        if emissive_uvname is not None and uvname is None:
            uvname = emissive_uvname

        # parse mertalic
        use_specular = get_input_attribute_value(shader, "useSpecularWorkflow", "value")[0]
        if not use_specular:
            metallic_texture, metallic_uvname = parse_component("metallic", "linear")
            if metallic_uvname is not None and uvname is None:
                uvname = metallic_uvname
        else:
            metallic_texture = None

        # parse roughness
        roughness_texture, roughness_uvname = parse_component("roughness", "linear")
        if roughness_uvname is not None and uvname is None:
            uvname = roughness_uvname

        # parse normal
        normal_texture, normal_uvname = parse_component("normal", "linear")
        if normal_uvname is not None and uvname is None:
            uvname = normal_uvname

        # parse ior
        ior = get_input_attribute_value(shader, "ior", "value")[0]

        if uvname is None:
            uvname = "st"

        return {
            "color_texture": color_texture,
            "opacity_texture": opacity_texture,
            "roughness_texture": roughness_texture,
            "metallic_texture": metallic_texture,
            "emissive_texture": emissive_texture,
            "normal_texture": normal_texture,
            "ior": ior,
        }, uvname

    elif shader_id == "UsdUVTexture":
        texture = get_input_attribute_value(shader, "file", "value")[0]
        if texture is not None:
            texture_image = np.asarray(Image.open(texture.resolvedPath))
            if texture_image.ndim == 3:
                if output_name == "r":
                    texture_image = texture_image[:, :, 0]
                elif output_name == "g":
                    texture_image = texture_image[:, :, 1]
                elif output_name == "b":
                    texture_image = texture_image[:, :, 2]
                elif output_name == "a":
                    texture_image = texture_image[:, :, 3]
                elif output_name == "rgb":
                    texture_image = texture_image[:, :, :3]
                else:
                    gs.raise_exception(f"Invalid output channel for UsdUVTexture: {output_name}.")
        else:
            texture_image = None

        texture_encode = get_input_attribute_value(shader, "sourceColorSpace", "value")[0] or "sRGB"
        texture_encode = cs_encode[texture_encode]
        texture_uvs, texture_uvs_output = get_input_attribute_value(shader, "st", "attribute")
        texture_uvs_name = parse_preview_surface(texture_uvs, texture_uvs_output)

        return texture_image, texture_encode, texture_uvs_name

    elif shader_id.startswith("UsdPrimvarReader"):
        primvar_name = get_input_attribute_value(shader, "varname", "value")[0]
        return primvar_name


def parse_usd_material(
    material: UsdShade.Material, surface: gs.surfaces.Surface
) -> tuple[gs.surfaces.Surface, str, bool]:
    surface_outputs = material.GetSurfaceOutputs()
    material_dict, uv_name = None, None
    material_surface = surface.copy()

    require_bake = False
    material_candidates = []
    for surface_output in surface_outputs:
        if not surface_output.HasConnectedSource():
            continue
        surface_output_connectable, surface_output_name, _ = surface_output.GetConnectedSource()
        surface_output_connect = surface_output_connectable.GetPrim()
        surface_shader = get_shader(surface_output_connect, "surface")
        surface_shader_implement = surface_shader.GetImplementationSource()
        surface_shader_id = surface_shader.GetShaderId()

        if surface_shader_implement == "id" and surface_shader_id == "UsdPreviewSurface":
            material_dict, uv_name = parse_preview_surface(surface_output_connect, surface_output_name)
            require_bake = False
            break

        material_candidates.append((surface_shader.GetPath(), surface_shader_id, surface_shader_implement))
        require_bake = True

    if material_dict is not None:
        material_surface.update_texture(
            color_texture=material_dict.get("color_texture"),
            opacity_texture=material_dict.get("opacity_texture"),
            roughness_texture=material_dict.get("roughness_texture"),
            metallic_texture=material_dict.get("metallic_texture"),
            normal_texture=material_dict.get("normal_texture"),
            emissive_texture=material_dict.get("emissive_texture"),
            ior=material_dict.get("ior"),
        )

    if require_bake:
        candidates_str = "\n".join(
            f"\tShader at {shader_path} with implement {shader_impl} and ID {shader_id}."
            for shader_path, shader_id, shader_impl in material_candidates
        )
        gs.logger.debug(f"Material require baking:\n{candidates_str}")
    return material_surface, uv_name, require_bake


# entrance
def parse_mesh_usd(path: str, group_by_material: bool, scale, surface: gs.surfaces.Surface, bake_cache=True):
    if path.lower().endswith(gs.options.morphs.USD_FORMATS[-1]):
        path = decompress_usdz(path)

    # Detect bake file caches using shared utility
    baked_path = detect_baked_cache(path) if bake_cache else None
    is_bake_cache_found = baked_path is not None
    if is_bake_cache_found:
        path = baked_path
        gs.logger.info(f"Baked assets detected and used: {path}")

    stage = Usd.Stage.Open(path)

    # Get stage scale and up-axis using shared utility
    meters_per_unit, yup = get_stage_scale_and_upaxis(stage)
    scale *= meters_per_unit
    xform_cache = UsdGeom.XformCache()

    mesh_infos = mu.MeshInfoGroup()
    materials = {}
    baked_materials = {}

    # parse materials
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            material_usd = UsdShade.Material(prim)
            material_spec = prim.GetPrimStack()[-1]
            material_id = material_spec.layer.identifier + material_spec.path.pathString
            material_pack = materials.get(material_id, None)

            if material_pack is None:
                material, uv_name, require_bake = parse_usd_material(material_usd, surface)
                materials[material_id] = (material, uv_name)
                if not is_bake_cache_found and require_bake:
                    baked_materials[material_id] = material_usd.GetPath()

    if baked_materials:
        # Use shared utility for material baking subprocess
        baked_stage_path = run_material_baking(
            stage=stage,
            materials_to_bake={k: str(v) for k, v in baked_materials.items()},
            original_path=path,
        )

        # Re-parse baked materials if baking succeeded
        if baked_stage_path:
            baked_stage = Usd.Stage.Open(baked_stage_path)
            for baked_material_id, baked_material_path in baked_materials.items():
                baked_material_usd = UsdShade.Material(baked_stage.GetPrimAtPath(baked_material_path))
                baked_material, uv_name, _ = parse_usd_material(baked_material_usd, surface)
                materials[baked_material_id] = (baked_material, uv_name)

    # parse geometries
    for prim in stage.Traverse():
        if prim.HasRelationship("material:binding"):
            if not prim.HasAPI(UsdShade.MaterialBindingAPI):
                UsdShade.MaterialBindingAPI.Apply(prim)
    for i, prim in enumerate(stage.Traverse()):
        if prim.IsA(UsdGeom.Mesh):
            matrix = np.asarray(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float32)
            if yup:
                matrix @= mu.Y_UP_TRANSFORM
            mesh_usd = UsdGeom.Mesh(prim)
            mesh_spec = prim.GetPrimStack()[-1]
            mesh_id = mesh_spec.layer.identifier + mesh_spec.path.pathString

            if not mesh_usd.GetPointsAttr().HasValue():
                continue
            points = np.array(mesh_usd.GetPointsAttr().Get(), dtype=np.float32)
            faces = np.array(mesh_usd.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            faces_vertex_counts = np.array(mesh_usd.GetFaceVertexCountsAttr().Get())
            points_faces_varying = False

            # parse normals
            normals = None
            normal_attr = mesh_usd.GetNormalsAttr()
            if normal_attr.HasValue():
                normals = np.array(normal_attr.Get(), dtype=np.float32)
                if normals.shape[0] != points.shape[0]:
                    if normals.shape[0] == faces.shape[0]:  # face varying meshes, adjacent faces do not share vertices
                        points_faces_varying = True
                    else:
                        gs.raise_exception(f"Size of normals mismatch for mesh {mesh_id} in usd file {path}.")

            # parse materials
            prim_bindings = UsdShade.MaterialBindingAPI(prim)
            material_usd = prim_bindings.ComputeBoundMaterial()[0]
            if material_usd.GetPrim().IsValid():
                material_spec = material_usd.GetPrim().GetPrimStack()[-1]
                material_file = material_spec.layer.identifier
                material_file = path if material_file == baked_path else material_file
                material_id = material_file + material_spec.path.pathString
                material, uv_name = materials.get(material_id, (None, "st"))
            else:
                material, uv_name, material_id = surface.copy(), "st", None

            # parse uvs
            uvs = None
            if uv_name is not None:
                uv_var = UsdGeom.PrimvarsAPI(prim).GetPrimvar(uv_name)
                if uv_var.IsDefined() and uv_var.HasValue():
                    uvs = np.array(uv_var.ComputeFlattened(), dtype=np.float32)
                    uvs[:, 1] = 1.0 - uvs[:, 1]
                    if uvs.shape[0] != points.shape[0]:
                        if uvs.shape[0] == faces.shape[0]:
                            points_faces_varying = True
                        elif uvs.shape[0] == 1:
                            uvs = None
                        else:
                            gs.raise_exception(f"Size of uvs mismatch for mesh {mesh_id} in usd file {path}.")

            # rearrange points and faces
            if points_faces_varying:
                points = points[faces]
                faces = np.arange(faces.shape[0])

            # triangulate faces
            if np.max(faces_vertex_counts) > 3:
                triangles = []
                bi = 0
                for face_vertex_count in faces_vertex_counts:
                    if face_vertex_count == 3:
                        triangles.append([faces[bi + 0], faces[bi + 1], faces[bi + 2]])
                    elif face_vertex_count > 3:
                        for i in range(1, face_vertex_count - 1):
                            triangles.append([faces[bi + 0], faces[bi + i], faces[bi + i + 1]])
                    bi += face_vertex_count
                triangles = np.array(triangles, dtype=np.int32)
                gs.logger.warning(f"Mesh {mesh_usd} has non-triangle faces.")
            else:
                triangles = faces.reshape(-1, 3)

            # process mesh
            processed_mesh = trimesh.Trimesh(
                vertices=points,
                faces=triangles,
                vertex_normals=normals,
                visual=trimesh.visual.TextureVisuals(uv=uvs) if uvs is not None else None,
                process=True,
            )
            points = processed_mesh.vertices
            triangles = processed_mesh.faces
            normals = processed_mesh.vertex_normals
            if uvs is not None:
                uvs = processed_mesh.visual.uv

            # apply tranform
            points, normals = mu.apply_transform(matrix, points, normals)

            group_idx = material_id if group_by_material else i
            mesh_info, first_created = mesh_infos.get(group_idx)
            if first_created:
                mesh_info.set_property(
                    surface=material,
                    metadata={
                        "path": path,  # unbaked file or cache
                        "name": material_id if group_by_material else mesh_id,
                        "require_bake": material_id in baked_materials,
                        "bake_success": material_id in baked_materials and material is not None,
                    },
                )
            mesh_info.append(points, triangles, normals, uvs)

    return mesh_infos.export_meshes(scale=scale)
