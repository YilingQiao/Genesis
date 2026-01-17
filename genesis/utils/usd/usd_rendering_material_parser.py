"""
USD Rendering Material Parser

Parser for extracting and parsing rendering materials from USD stages.
"""

from pxr import UsdShade

import genesis as gs

from . import usda
from .usd_parser_context import UsdParserContext


def parse_all_materials(context: UsdParserContext) -> tuple[dict, dict]:
    """
    Find all materials in the USD stage and parse them.

    Parameters
    ----------
    context : UsdParserContext
        The parser context to store materials in.

    Returns
    -------
    tuple[dict, dict]
        A tuple of (materials_dict, materials_requiring_bake_dict):
        - materials_dict: material_id -> (surface, uv_name)
        - materials_requiring_bake_dict: material_id -> prim_path (for baking)
    """
    stage = context.stage
    materials = context.materials
    materials_requiring_bake = {}  # Track locally, return to caller
    default_surface = gs.surfaces.Default()

    # Parse materials from the stage
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            material_usd = UsdShade.Material(prim)
            material_spec = prim.GetPrimStack()[-1]
            material_id = material_spec.layer.identifier + material_spec.path.pathString

            if material_id not in materials:
                material, uv_name, require_bake = usda.parse_usd_material(material_usd, default_surface)
                materials[material_id] = (material, uv_name)

                # Track materials requiring baking (return to caller)
                if require_bake:
                    materials_requiring_bake[material_id] = str(material_usd.GetPath())
                    gs.logger.debug(f"Material {material_id} requires baking")

    return materials, materials_requiring_bake
