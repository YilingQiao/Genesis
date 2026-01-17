"""
Test USD rendering property support.

This module tests the rendering property parsing features added to usd_parser.py:
- Stage metersPerUnit scaling
- Y-up to Z-up axis conversion
- Shared utility functions
"""

import os
import tempfile

import numpy as np
import pytest

# Check for USD support
try:
    from pxr import Usd, UsdGeom

    HAS_USD_SUPPORT = True
except ImportError:
    HAS_USD_SUPPORT = False


@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_get_stage_scale_and_upaxis_default():
    """Test stage scale and up-axis detection with default values."""
    from genesis.utils.usd.usd_stage_utils import get_stage_scale_and_upaxis

    with tempfile.NamedTemporaryFile(suffix=".usda", delete=False) as f:
        f.write(b"""#usda 1.0
(
    defaultPrim = "Root"
)
def Xform "Root" {
}
""")
        f.flush()

        try:
            stage = Usd.Stage.Open(f.name)
            meters_per_unit, up_axis_is_y = get_stage_scale_and_upaxis(stage)

            # Default values: metersPerUnit=0.01 (cm), upAxis=Y
            assert meters_per_unit == pytest.approx(0.01, rel=1e-6)
            assert up_axis_is_y is True
        finally:
            os.unlink(f.name)


@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_get_stage_scale_and_upaxis_custom():
    """Test stage scale and up-axis detection with custom values."""
    from genesis.utils.usd.usd_stage_utils import get_stage_scale_and_upaxis

    with tempfile.NamedTemporaryFile(suffix=".usda", delete=False) as f:
        f.write(b"""#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 1.0
    upAxis = "Z"
)
def Xform "Root" {
}
""")
        f.flush()

        try:
            stage = Usd.Stage.Open(f.name)
            meters_per_unit, up_axis_is_y = get_stage_scale_and_upaxis(stage)

            assert meters_per_unit == pytest.approx(1.0, rel=1e-6)
            assert up_axis_is_y is False
        finally:
            os.unlink(f.name)


@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_detect_baked_cache_not_found():
    """Test baked cache detection when no cache exists."""
    from genesis.utils.usd.usd_stage_utils import detect_baked_cache

    with tempfile.NamedTemporaryFile(suffix=".usda", delete=False) as f:
        f.write(b"""#usda 1.0
(
    defaultPrim = "Root"
)
def Xform "Root" {
}
""")
        f.flush()

        try:
            result = detect_baked_cache(f.name)
            assert result is None
        finally:
            os.unlink(f.name)


@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_parser_context_stage_metadata():
    """Test UsdParserContext stage metadata handling."""
    from genesis.utils.usd.usd_parser_context import UsdParserContext

    with tempfile.NamedTemporaryFile(suffix=".usda", delete=False) as f:
        f.write(b"""#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 0.5
    upAxis = "Y"
)
def Xform "Root" {
}
""")
        f.flush()

        try:
            stage = Usd.Stage.Open(f.name)
            context = UsdParserContext(stage)

            # Default values before setting metadata
            assert context.meters_per_unit == 1.0
            assert context.up_axis_is_y is False
            assert context.stage_scale == 1.0

            # Set metadata with morph scale
            context.set_stage_metadata(meters_per_unit=0.5, up_axis_is_y=True, morph_scale=2.0)

            assert context.meters_per_unit == 0.5
            assert context.up_axis_is_y is True
            assert context.stage_scale == pytest.approx(1.0, rel=1e-6)  # 2.0 * 0.5 = 1.0
        finally:
            os.unlink(f.name)


@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_y_up_transform_consistency():
    """Test that Y_UP_TRANSFORM is consistent between imports."""
    from genesis.utils import mesh as mu

    # Get Y_UP_TRANSFORM
    y_up_transform = mu.Y_UP_TRANSFORM

    # Verify it's a valid rotation matrix (det = 1, R @ R.T = I)
    rot_part = y_up_transform[:3, :3]
    assert np.abs(np.linalg.det(rot_part) - 1.0) < 1e-6
    assert np.allclose(rot_part @ rot_part.T, np.eye(3), atol=1e-6)

    # Verify it converts Y-up to Z-up coordinate system
    # The transform maps: Y -> -Z, Z -> Y (90-degree rotation around X axis)
    y_axis = np.array([0, 1, 0])
    z_up_result = rot_part @ y_axis
    # Y axis becomes negative Z axis in the new coordinate system
    assert np.allclose(z_up_result, [0, 0, -1], atol=1e-6)


@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_add_stage_y_up_and_meters_per_unit(initialize_genesis):
    """
    Integration test for add_stage with Y-up axis and custom metersPerUnit.

    This test validates AC3 (metersPerUnit scaling) and AC4 (Y-up conversion)
    by creating a temporary USD stage with:
    - upAxis = "Y"
    - metersPerUnit = 0.01 (centimeters)
    - A simple rigid body with physics APIs

    The test verifies that:
    1. The stage loads successfully via add_stage
    2. The entity is created with correct scaling (0.01 * 1.0 = 0.01)
    3. Y-up conversion is applied to link transforms
    """
    from genesis.utils import mesh as mu

    import genesis as gs

    # Create a temporary USD file with Y-up and metersPerUnit=0.01
    with tempfile.NamedTemporaryFile(suffix=".usda", delete=False, mode="w") as f:
        f.write("""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 0.01
    upAxis = "Y"
)

def Xform "World" (
    kind = "assembly"
)
{
    def Xform "RigidBody" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        # Position at (100, 200, 0) in centimeters
        double3 xformOp:translate = (100, 200, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        float physics:mass = 1.0

        def Cube "Collision" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            # 10cm cube = 0.1m after scaling
            double size = 10
        }
    }
}
""")
        f.flush()
        usd_path = f.name

    try:
        scene = gs.Scene(show_viewer=False)

        # Add the stage
        entities = scene.add_stage(
            morph=gs.morphs.USD(file=usd_path),
            vis_mode="collision",
        )

        # Verify entity was created
        assert len(entities) > 0, "No entities created from USD stage"

        # Get the first entity
        entity = list(entities.values())[0]
        assert entity is not None

        # Verify entity has links
        assert len(entity.links) > 0, "Entity has no links"

        # Get the root link (first link with the rigid body)
        link = entity.links[0]

        # Calculate expected position:
        # 1. Original USD position in Y-up: (100, 200, 0)
        # 2. Apply Y_UP_TRANSFORM: (x, y, z) -> (x, z, -y) = (100, 0, -200)
        # 3. Apply metersPerUnit scaling (0.01): (1.0, 0.0, -2.0)
        y_up_transform = mu.Y_UP_TRANSFORM
        original_pos = np.array([100.0, 200.0, 0.0, 1.0])
        transformed_pos = y_up_transform @ original_pos
        meters_per_unit = 0.01
        expected_pos = transformed_pos[:3] * meters_per_unit

        # Assert link position matches expected values (with tolerance)
        link_pos = np.array(link.pos)
        assert np.allclose(link_pos, expected_pos, atol=1e-5), (
            f"Link position {link_pos} does not match expected {expected_pos} (Y-up conversion + metersPerUnit scaling)"
        )

        # Verify collision geometry is scaled correctly
        # The cube size is 10 (cm) which should become ~0.1m in extents after scaling
        if len(link.geoms) > 0:
            geom = link.geoms[0]
            # Geom mesh should have vertices scaled by metersPerUnit
            # A 10cm cube centered at origin has vertices at +/-5cm = +/-0.05m
            if hasattr(geom, "mesh") and geom.mesh is not None:
                verts = geom.mesh.verts
                # Check that vertex extents are scaled correctly
                max_extent = np.max(np.abs(verts))
                # USD Cube with size=10 has half-extent of 5, scaled by 0.01 = 0.05
                expected_half_extent = 5.0 * meters_per_unit
                assert np.isclose(max_extent, expected_half_extent, rtol=0.1), (
                    f"Geom extent {max_extent} does not match expected {expected_half_extent}"
                )

        # Build and verify the scene builds without error
        scene.build()

    finally:
        os.unlink(usd_path)


@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_add_stage_z_up_no_conversion(initialize_genesis):
    """
    Integration test for add_stage with Z-up axis (no conversion needed).

    This test validates that Z-up stages work correctly without Y-up conversion.
    The position should remain unchanged since:
    - upAxis = "Z" (no Y-up conversion)
    - metersPerUnit = 1.0 (no scaling)
    """
    import genesis as gs

    # Create a temporary USD file with Z-up
    with tempfile.NamedTemporaryFile(suffix=".usda", delete=False, mode="w") as f:
        f.write("""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1.0
    upAxis = "Z"
)

def Xform "World" (
    kind = "assembly"
)
{
    def Xform "RigidBody" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        double3 xformOp:translate = (1, 2, 3)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        float physics:mass = 1.0

        def Cube "Collision" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double size = 1
        }
    }
}
""")
        f.flush()
        usd_path = f.name

    try:
        scene = gs.Scene(show_viewer=False)

        # Add the stage
        entities = scene.add_stage(
            morph=gs.morphs.USD(file=usd_path),
            vis_mode="collision",
        )

        # Verify entity was created
        assert len(entities) > 0, "No entities created from USD stage"

        # Get the first entity
        entity = list(entities.values())[0]
        assert entity is not None

        # Verify entity has links
        assert len(entity.links) > 0, "Entity has no links"

        # Get the root link
        link = entity.links[0]

        # For Z-up with metersPerUnit=1.0, position should be unchanged
        # Original position: (1, 2, 3)
        # No Y-up conversion (already Z-up)
        # No scaling (metersPerUnit=1.0)
        expected_pos = np.array([1.0, 2.0, 3.0])

        # Assert link position is unchanged
        link_pos = np.array(link.pos)
        assert np.allclose(link_pos, expected_pos, atol=1e-5), (
            f"Link position {link_pos} does not match expected {expected_pos} (Z-up should have no conversion applied)"
        )

        # Verify collision geometry size is unchanged
        # Cube size=1 with metersPerUnit=1.0 should have half-extent of 0.5
        if len(link.geoms) > 0:
            geom = link.geoms[0]
            if hasattr(geom, "mesh") and geom.mesh is not None:
                verts = geom.mesh.verts
                max_extent = np.max(np.abs(verts))
                expected_half_extent = 0.5  # size=1, half is 0.5
                assert np.isclose(max_extent, expected_half_extent, rtol=0.1), (
                    f"Geom extent {max_extent} does not match expected {expected_half_extent}"
                )

        # Build and verify the scene builds without error
        scene.build()

    finally:
        os.unlink(usd_path)


@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_usdz_baking_path_normalization(initialize_genesis, monkeypatch):
    """
    Test that USDZ baking uses the decompressed path consistently.

    This test validates AC2/AC5 by verifying that:
    1. decompress_usdz is called for .usdz files
    2. The decompressed path is used for detect_baked_cache
    3. The decompressed path is used for run_material_baking

    Uses monkeypatching to intercept the path handling.
    Calls import_from_stage directly to bypass morph file validation.
    """
    import genesis as gs
    from genesis.utils.usd import usd_parser
    from genesis.utils.usd.usd_parser import import_from_stage

    # Create a temporary USD file (will be used as "decompressed" content)
    with tempfile.NamedTemporaryFile(suffix=".usda", delete=False, mode="w") as f:
        f.write("""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1.0
    upAxis = "Z"
)

def Xform "World" (
    kind = "assembly"
)
{
    def Xform "RigidBody" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        float physics:mass = 1.0

        def Cube "Collision" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double size = 1
        }
    }
}
""")
        f.flush()
        usda_path = f.name

    # Create a fake .usdz path (doesn't need to exist since we monkeypatch decompress)
    fake_usdz_path = usda_path.replace(".usda", ".usdz")

    # Track what paths are passed to baking functions
    captured_paths = {"decompress_called": False, "decompress_input": None, "baking_path": None}

    def mock_decompress_usdz(usdz_path):
        """Mock decompress_usdz to return the usda path."""
        captured_paths["decompress_called"] = True
        captured_paths["decompress_input"] = usdz_path
        return usda_path  # Return the real usda file path

    def mock_run_material_baking(stage, materials_to_bake, original_path):
        """Mock run_material_baking to capture the path argument."""
        captured_paths["baking_path"] = original_path
        return None  # No baking actually happens

    def mock_parse_all_materials(context):
        """Mock parse_all_materials to force the baking path to execute."""
        # Return non-empty materials_requiring_bake to trigger baking branch
        return {}, {"fake_material_id": "/root/Looks/FakeMaterial"}

    # Apply monkeypatches
    monkeypatch.setattr(usd_parser, "decompress_usdz", mock_decompress_usdz)
    monkeypatch.setattr(usd_parser, "run_material_baking", mock_run_material_baking)
    monkeypatch.setattr(usd_parser, "parse_all_materials", mock_parse_all_materials)

    try:
        scene = gs.Scene(show_viewer=False)

        # Create a morph using the real usda path (for validation), but call import_from_stage
        # with the fake .usdz path to test the path normalization logic
        usd_morph = gs.morphs.USD(file=usda_path)

        # Call import_from_stage directly with fake .usdz path to trigger decompression
        entities = import_from_stage(
            scene=scene,
            stage=fake_usdz_path,  # This triggers the .usdz path
            vis_mode="collision",
            usd_morph=usd_morph,
        )

        # Verify decompress_usdz was called with the .usdz path
        assert captured_paths["decompress_called"], "decompress_usdz was not called for .usdz file"
        assert captured_paths["decompress_input"] == fake_usdz_path, (
            f"decompress_usdz received wrong path: {captured_paths['decompress_input']}"
        )

        # Verify run_material_baking received the decompressed path (not the .usdz path)
        assert captured_paths["baking_path"] is not None, "run_material_baking was not called"
        assert captured_paths["baking_path"] == usda_path, (
            f"run_material_baking received {captured_paths['baking_path']} but expected decompressed path {usda_path}"
        )
        # Specifically verify it did NOT receive the original .usdz path
        assert captured_paths["baking_path"] != fake_usdz_path, (
            "run_material_baking should use decompressed path, not original .usdz path"
        )

    finally:
        os.unlink(usda_path)
