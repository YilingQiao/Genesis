"""
USD Stage Utilities

Shared utilities for USD stage preprocessing including:
- USDZ decompression
- Baked cache detection
- Stage metadata extraction
- Material baking orchestration
"""

import io
import logging
import os
import shutil
import subprocess
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom

import genesis as gs

from .. import mesh as mu


def decompress_usdz(usdz_path: str) -> str:
    """
    Decompress a USDZ file to a cache directory.

    Parameters
    ----------
    usdz_path : str
        Path to the USDZ file to decompress.

    Returns
    -------
    str
        Path to the root USD file within the decompressed directory.
    """
    usdz_folder = mu.get_usd_zip_path(usdz_path)

    # The first file in the package must be a native usd file.
    # See https://openusd.org/docs/Usdz-File-Format-Specification.html
    zip_files = Usd.ZipFile.Open(usdz_path)
    zip_filelist = zip_files.GetFileNames()
    root_file = zip_filelist[0]
    if not root_file.lower().endswith(gs.options.morphs.USD_FORMATS[:-1]):
        gs.raise_exception(f"Invalid usdz root file: {root_file}")
    root_path = os.path.join(usdz_folder, root_file)

    if not os.path.exists(root_path):
        for file_name in zip_filelist:
            file_data = io.BytesIO(zip_files.GetFile(file_name))
            file_path = os.path.join(usdz_folder, file_name)
            file_folder = os.path.dirname(file_path)
            os.makedirs(file_folder, exist_ok=True)
            with open(file_path, "wb") as out:
                out.write(file_data.read())
        gs.logger.warning(f"USDZ file {usdz_path} decompressed to {root_path}.")
    else:
        gs.logger.info(f"Decompressed assets detected and used: {root_path}.")
    return root_path


def replace_asset_symlinks(stage: Usd.Stage):
    """
    Replace asset symlinks with real files for baking compatibility.

    Some baking tools don't handle symlinks correctly, so this function
    replaces symlinks with copies of the actual files.

    Parameters
    ----------
    stage : Usd.Stage
        The USD stage to process for symlinks.
    """
    asset_paths = set()

    for prim in stage.TraverseAll():
        for attr in prim.GetAttributes():
            value = attr.Get()
            if isinstance(value, Sdf.AssetPath):
                asset_paths.add(value.resolvedPath)
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, Sdf.AssetPath):
                        asset_paths.add(v.resolvedPath)

    for asset_path in map(Path, asset_paths):
        if not asset_path.is_symlink():
            continue

        real_path = asset_path.resolve()
        if asset_path.suffix.lower() == real_path.suffix.lower():
            continue

        asset_path.unlink()
        if real_path.is_file():
            gs.logger.warning(f"Replacing symlink {asset_path} with real file {real_path}.")
            shutil.copy2(real_path, asset_path)


def get_stage_scale_and_upaxis(stage: Usd.Stage) -> tuple:
    """
    Get stage metersPerUnit and up-axis.

    Parameters
    ----------
    stage : Usd.Stage
        The USD stage to extract metadata from.

    Returns
    -------
    tuple
        (meters_per_unit: float, up_axis_is_y: bool)
    """
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    up_axis_is_y = UsdGeom.GetStageUpAxis(stage) == "Y"
    return meters_per_unit, up_axis_is_y


def detect_baked_cache(file_path: str) -> str | None:
    """
    Check if a baked cache exists for the given USD file.

    Parameters
    ----------
    file_path : str
        Path to the USD file.

    Returns
    -------
    str or None
        The baked path if found, None otherwise.
    """
    baked_folder = mu.get_usd_bake_path(file_path)
    baked_path = os.path.join(baked_folder, os.path.basename(file_path))
    if os.path.exists(baked_path):
        return baked_path
    return None


def run_material_baking(
    stage: Usd.Stage,
    materials_to_bake: dict,
    original_path: str,
) -> str | None:
    """
    Run material baking subprocess for non-UsdPreviewSurface materials.

    This function runs the usda_bake.py script in a subprocess to convert
    non-UsdPreviewSurface materials to UsdPreviewSurface format.

    Parameters
    ----------
    stage : Usd.Stage
        The USD stage containing materials to bake.
    materials_to_bake : dict
        Dictionary mapping material_id -> prim_path for materials needing baking.
    original_path : str
        Path to the original USD file.

    Returns
    -------
    str or None
        Path to the baked stage if successful, None otherwise.
    """
    if not materials_to_bake:
        return None

    device = gs.device
    if device.type == "cpu":
        try:
            device, *_ = gs.utils.get_device(gs.cuda)
        except gs.GenesisException as e:
            gs.logger.warning(f"USD baking requires CUDA GPU: {e}")
            return None

    replace_asset_symlinks(stage)

    baked_folder = mu.get_usd_bake_path(original_path)
    os.makedirs(baked_folder, exist_ok=True)

    # Note that it is necessary to call 'bake_usd_material' via a subprocess to ensure proper isolation of
    # omniverse kit, otherwise the global conversion registry of some Python bindings will be conflicting between
    # each, ultimately leading to segfault...
    commands = [
        "python",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "usda_bake.py"),
        "--input_file",
        original_path,
        "--output_dir",
        baked_folder,
        "--usd_material_paths",
        *map(str, materials_to_bake.values()),
        "--device",
        str(device.index if device.index is not None else 0),
        "--log_level",
        logging.getLevelName(gs.logger.level).lower(),
    ]
    gs.logger.debug(f"Execute: {' '.join(commands)}")

    try:
        result = subprocess.run(
            commands,
            capture_output=True,
            check=True,
            text=True,
        )
        if result.stdout:
            gs.logger.debug(result.stdout)
        if result.stderr:
            gs.logger.warning(result.stderr)
    except (subprocess.CalledProcessError, OSError) as e:
        gs.logger.warning(f"Baking process failed: {e} (Note that USD baking may only support Python 3.10 now.)")
        return None

    baked_path = os.path.join(baked_folder, os.path.basename(original_path))
    if os.path.exists(baked_path):
        gs.logger.warning(f"USD materials baked to file {baked_path}")

        # Cleanup baked texture folders
        for baked_texture_obj in Path(baked_folder).glob("baked_textures*"):
            shutil.rmtree(baked_texture_obj)

        return baked_path

    return None
