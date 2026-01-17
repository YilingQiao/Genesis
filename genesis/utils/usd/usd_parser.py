"""
USD Parser

Main parser entrance for importing USD stages into Genesis scenes.
Provides the parse pipeline: materials -> articulations -> rigid bodies.
"""

from typing import Dict, Literal

from pxr import Usd, UsdShade

import genesis as gs
from genesis.options.morphs import USD, USD_FORMATS

from . import usda
from .usd_parser_context import UsdParserContext
from .usd_rendering_material_parser import parse_all_materials
from .usd_rigid_entity_parser import parse_all_rigid_entities
from .usd_stage_utils import (
    decompress_usdz,
    detect_baked_cache,
    get_stage_scale_and_upaxis,
    run_material_baking,
)


def import_from_stage(
    scene: gs.Scene,
    stage: Usd.Stage | str,
    vis_mode: Literal["visual", "collision"],
    usd_morph: USD,
    visualize_contact: bool = False,
):
    """
    Import all entities from a USD stage or file into the scene.

    Parse Pipeline:
    1. Preprocess file (USDZ decompression, baked cache detection)
    2. Parse all rendering materials and record them in UsdParserContext
    3. Run material baking if needed
    4. Parse all rigid entities (articulations and rigid bodies) and return created gs Entities

    Parameters
    ----------
    scene : gs.Scene
        The scene to add entities to.
    stage : Usd.Stage | str
        The USD stage to import from, or a file path string to open.
    vis_mode : Literal["visual", "collision"]
        Visualization mode.
    usd_morph : USD
        USD morph configuration.
    visualize_contact : bool, optional
        Whether to visualize contact, by default False.

    Returns
    -------
    Dict[str, Entity]
        Dictionary of created entities (both articulations and rigid bodies) keyed by prim path.
    """
    from genesis.engine.entities.base_entity import Entity as GSEntity

    original_file_path = None
    baked_path = None

    # Handle file path input with preprocessing
    if isinstance(stage, str):
        original_file_path = stage

        # USDZ decompression (MUST happen before Stage.Open)
        # Use USD_FORMATS constant - last element is .usdz
        if stage.lower().endswith(USD_FORMATS[-1]):
            stage = decompress_usdz(stage)

        # Check for existing baked cache
        baked_path = detect_baked_cache(stage)
        if baked_path:
            gs.logger.info(f"Baked assets detected and used: {baked_path}")
            stage = baked_path

        stage = Usd.Stage.Open(stage)

    # Create parser context
    context = UsdParserContext(stage)

    # Get stage scale and up-axis
    meters_per_unit, up_axis_is_y = get_stage_scale_and_upaxis(stage)
    context.set_stage_metadata(meters_per_unit, up_axis_is_y)

    context._vis_mode = vis_mode
    usd_morph.parser_ctx = context

    # Return Values
    entities: Dict[str, GSEntity] = {}

    # Step 1: Parse all rendering materials (returns tuple with baking needs)
    materials, materials_requiring_bake = parse_all_materials(context)
    gs.logger.debug(f"Parsed {len(materials)} materials from USD stage.")

    # Step 2: Material baking if needed (targeted re-parse)
    if materials_requiring_bake and not baked_path:
        file_to_bake = original_file_path or stage.GetRootLayer().realPath
        baked_stage_path = run_material_baking(
            stage=stage,
            materials_to_bake=materials_requiring_bake,
            original_path=file_to_bake,
        )
        if baked_stage_path:
            # Only re-parse the baked materials (not all materials)
            baked_stage = Usd.Stage.Open(baked_stage_path)
            default_surface = gs.surfaces.Default()
            for baked_material_id, baked_material_path in materials_requiring_bake.items():
                baked_material_usd = UsdShade.Material(baked_stage.GetPrimAtPath(baked_material_path))
                baked_material, uv_name, _ = usda.parse_usd_material(baked_material_usd, default_surface)
                context.materials[baked_material_id] = (baked_material, uv_name)
            gs.logger.debug(f"Re-parsed {len(materials_requiring_bake)} baked materials.")

    # Step 3: Parse all rigid entities (articulations and rigid bodies)
    entities = parse_all_rigid_entities(scene, stage, context, usd_morph, vis_mode, visualize_contact)
    gs.logger.debug(f"Parsed {len(entities)} rigid entities from USD stage.")

    if not entities:
        gs.logger.warning(f"No articulations or rigid bodies found in USD: {usd_morph.file}")
    return entities
