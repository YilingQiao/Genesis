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
    - A simple rigid body with physics APIs and explicit rotation

    The test verifies that:
    1. The stage loads successfully via add_stage
    2. The entity is created with correct scaling (0.01 * 1.0 = 0.01)
    3. Y-up conversion is applied to link position AND rotation (quaternion)
    4. Both collision and visual modes work correctly
    """
    from genesis.utils import geom as gu
    from genesis.utils import mesh as mu

    import genesis as gs

    # Create a temporary USD file with Y-up, metersPerUnit=0.01, and explicit rotation
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
        # Position at (100, 200, 0) in centimeters with 45-degree rotation around Y
        double3 xformOp:translate = (100, 200, 0)
        float3 xformOp:rotateXYZ = (0, 45, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]

        float physics:mass = 1.0

        def Cube "Collision" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            # 10cm cube = 0.1m after scaling
            double size = 10
        }

        def Mesh "Visual"
        {
            # Simple visual mesh (cube-like)
            float3[] points = [(-5, -5, -5), (5, -5, -5), (5, 5, -5), (-5, 5, -5),
                               (-5, -5, 5), (5, -5, 5), (5, 5, 5), (-5, 5, 5)]
            int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
            int[] faceVertexIndices = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 5, 4, 2, 3, 7, 6, 0, 3, 7, 4, 1, 2, 6, 5]
        }
    }
}
""")
        f.flush()
        usd_path = f.name

    try:
        # Test with collision mode
        scene_collision = gs.Scene(show_viewer=False)

        entities = scene_collision.add_stage(
            morph=gs.morphs.USD(file=usd_path),
            vis_mode="collision",
        )

        assert len(entities) > 0, "No entities created from USD stage"

        entity = list(entities.values())[0]
        assert entity is not None
        assert len(entity.links) > 0, "Entity has no links"

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

        link_pos = np.array(link.pos)
        assert np.allclose(link_pos, expected_pos, atol=1e-5), (
            f"Link position {link_pos} does not match expected {expected_pos} (Y-up + metersPerUnit)"
        )

        # Calculate expected quaternion:
        # Original rotation: 45 degrees around Y axis in Y-up space
        # After Y_UP_TRANSFORM (left multiply), the rotation basis changes
        # Expected: Y_UP_R @ original_R
        angle_rad = np.radians(45)
        cos_half = np.cos(angle_rad / 2)
        sin_half = np.sin(angle_rad / 2)
        # Rotation around Y axis: quat = [cos(a/2), 0, sin(a/2), 0] (w, x, y, z)
        original_quat = np.array([cos_half, 0, sin_half, 0])
        original_rot = gu.quat_to_R(original_quat)

        # Apply Y_UP_TRANSFORM to rotation: Y_UP_R @ original_R
        y_up_rot = y_up_transform[:3, :3]
        expected_rot = y_up_rot @ original_rot
        expected_quat = gu.R_to_quat(expected_rot)

        link_quat = np.array(link.quat)
        # Quaternions can have opposite signs but represent the same rotation
        if np.dot(link_quat, expected_quat) < 0:
            expected_quat = -expected_quat

        assert np.allclose(link_quat, expected_quat, atol=1e-4), (
            f"Link quaternion {link_quat} does not match expected {expected_quat} (Y-up rotation conversion)"
        )

        # Verify collision geometry is scaled correctly
        if len(link.geoms) > 0:
            geom = link.geoms[0]
            if hasattr(geom, "mesh") and geom.mesh is not None:
                verts = geom.mesh.verts
                max_extent = np.max(np.abs(verts))
                expected_half_extent = 5.0 * meters_per_unit
                assert np.isclose(max_extent, expected_half_extent, rtol=0.1), (
                    f"Geom extent {max_extent} does not match expected {expected_half_extent}"
                )

        scene_collision.build()

        # Test with visual mode to verify Y-up conversion for visual meshes (AC4)
        scene_visual = gs.Scene(show_viewer=False)

        entities_visual = scene_visual.add_stage(
            morph=gs.morphs.USD(file=usd_path),
            vis_mode="visual",
        )

        assert len(entities_visual) > 0, "No entities created in visual mode"

        entity_visual = list(entities_visual.values())[0]
        link_visual = entity_visual.links[0]

        # Position and quaternion should be the same in visual mode
        link_pos_visual = np.array(link_visual.pos)
        assert np.allclose(link_pos_visual, expected_pos, atol=1e-5), (
            f"Visual mode position {link_pos_visual} does not match expected {expected_pos}"
        )

        link_quat_visual = np.array(link_visual.quat)
        if np.dot(link_quat_visual, expected_quat) < 0:
            expected_quat = -expected_quat
        assert np.allclose(link_quat_visual, expected_quat, atol=1e-4), (
            f"Visual mode quaternion {link_quat_visual} does not match expected {expected_quat}"
        )

        scene_visual.build()

    finally:
        os.unlink(usd_path)


@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_add_stage_z_up_no_conversion(initialize_genesis):
    """
    Integration test for add_stage with Z-up axis (no conversion needed).

    This test validates that Z-up stages work correctly without Y-up conversion.
    The position and rotation should remain unchanged since:
    - upAxis = "Z" (no Y-up conversion)
    - metersPerUnit = 1.0 (no scaling)
    """
    from genesis.utils import geom as gu

    import genesis as gs

    # Create a temporary USD file with Z-up and explicit rotation
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
        float3 xformOp:rotateXYZ = (30, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]

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

        entities = scene.add_stage(
            morph=gs.morphs.USD(file=usd_path),
            vis_mode="collision",
        )

        assert len(entities) > 0, "No entities created from USD stage"

        entity = list(entities.values())[0]
        assert entity is not None
        assert len(entity.links) > 0, "Entity has no links"

        link = entity.links[0]

        # For Z-up with metersPerUnit=1.0, position should be unchanged
        expected_pos = np.array([1.0, 2.0, 3.0])

        link_pos = np.array(link.pos)
        assert np.allclose(link_pos, expected_pos, atol=1e-5), (
            f"Link position {link_pos} does not match expected {expected_pos} (Z-up should have no conversion)"
        )

        # For Z-up, quaternion should match the original rotation (30 degrees around X)
        # No Y_UP_TRANSFORM should be applied
        angle_rad = np.radians(30)
        cos_half = np.cos(angle_rad / 2)
        sin_half = np.sin(angle_rad / 2)
        # Rotation around X axis: quat = [cos(a/2), sin(a/2), 0, 0] (w, x, y, z)
        expected_quat = np.array([cos_half, sin_half, 0, 0])

        link_quat = np.array(link.quat)
        # Quaternions can have opposite signs but represent the same rotation
        if np.dot(link_quat, expected_quat) < 0:
            expected_quat = -expected_quat

        assert np.allclose(link_quat, expected_quat, atol=1e-4), (
            f"Link quaternion {link_quat} does not match expected {expected_quat} (Z-up should have no rotation conversion)"
        )

        # Verify collision geometry size is unchanged
        if len(link.geoms) > 0:
            geom = link.geoms[0]
            if hasattr(geom, "mesh") and geom.mesh is not None:
                verts = geom.mesh.verts
                max_extent = np.max(np.abs(verts))
                expected_half_extent = 0.5  # size=1, half is 0.5
                assert np.isclose(max_extent, expected_half_extent, rtol=0.1), (
                    f"Geom extent {max_extent} does not match expected {expected_half_extent}"
                )

        scene.build()

    finally:
        os.unlink(usd_path)


@pytest.mark.skipif(not HAS_USD_SUPPORT, reason="USD support not available")
def test_usdz_baking_path_normalization(initialize_genesis, monkeypatch):
    """
    Test that USDZ baking uses the decompressed path consistently.

    This test validates AC2/AC5 by verifying that:
    1. decompress_usdz is called for .usdz files
    2. detect_baked_cache receives the decompressed path (not the .usdz path)
    3. run_material_baking receives the decompressed path (not the .usdz path)

    Uses monkeypatching to intercept path handling while using the actual add_stage path.
    Creates a real .usdz file to pass morph file validation.
    """
    import zipfile

    import genesis as gs
    from genesis.utils.usd import usd_parser

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

    # Create a real .usdz file (zip file containing the .usda)
    # This is required because gs.morphs.USD validates file existence
    usdz_path = usda_path.replace(".usda", ".usdz")
    usda_basename = os.path.basename(usda_path)
    with zipfile.ZipFile(usdz_path, "w") as zf:
        zf.write(usda_path, usda_basename)

    # Track what paths are passed to functions
    captured_paths = {
        "decompress_called": False,
        "decompress_input": None,
        "detect_cache_path": None,
        "baking_path": None,
    }

    def mock_decompress_usdz(usdz_path_arg):
        """Mock decompress_usdz to return the usda path."""
        captured_paths["decompress_called"] = True
        captured_paths["decompress_input"] = usdz_path_arg
        return usda_path  # Return the real usda file path

    def mock_detect_baked_cache(file_path):
        """Mock detect_baked_cache to capture its input path."""
        captured_paths["detect_cache_path"] = file_path
        return None  # No cache found

    def mock_run_material_baking(stage, materials_to_bake, original_path):
        """Mock run_material_baking to capture the path argument."""
        captured_paths["baking_path"] = original_path
        return None  # No baking actually happens

    def mock_parse_all_materials(context):
        """Mock parse_all_materials to force the baking path to execute."""
        # Return non-empty materials_requiring_bake to trigger baking branch
        return {}, {"fake_material_id": "/root/Looks/FakeMaterial"}

    # Apply monkeypatches to the usd_parser module (where the imports are used)
    monkeypatch.setattr(usd_parser, "decompress_usdz", mock_decompress_usdz)
    monkeypatch.setattr(usd_parser, "detect_baked_cache", mock_detect_baked_cache)
    monkeypatch.setattr(usd_parser, "run_material_baking", mock_run_material_baking)
    monkeypatch.setattr(usd_parser, "parse_all_materials", mock_parse_all_materials)

    try:
        scene = gs.Scene(show_viewer=False)

        # Use add_stage with the real .usdz file to test the full path
        entities = scene.add_stage(
            morph=gs.morphs.USD(file=usdz_path),
            vis_mode="collision",
        )

        # Verify decompress_usdz was called with the .usdz path
        assert captured_paths["decompress_called"], "decompress_usdz was not called for .usdz file"
        assert captured_paths["decompress_input"] == usdz_path, (
            f"decompress_usdz received wrong path: {captured_paths['decompress_input']}, expected {usdz_path}"
        )

        # Verify detect_baked_cache received the decompressed path (not the .usdz path)
        assert captured_paths["detect_cache_path"] is not None, "detect_baked_cache was not called"
        assert captured_paths["detect_cache_path"] == usda_path, (
            f"detect_baked_cache received {captured_paths['detect_cache_path']} "
            f"but expected decompressed path {usda_path}"
        )
        assert captured_paths["detect_cache_path"] != usdz_path, (
            "detect_baked_cache should use decompressed path, not original .usdz path"
        )

        # Verify run_material_baking received the decompressed path (not the .usdz path)
        assert captured_paths["baking_path"] is not None, "run_material_baking was not called"
        assert captured_paths["baking_path"] == usda_path, (
            f"run_material_baking received {captured_paths['baking_path']} but expected decompressed path {usda_path}"
        )
        assert captured_paths["baking_path"] != usdz_path, (
            "run_material_baking should use decompressed path, not original .usdz path"
        )

    finally:
        os.unlink(usda_path)
        if os.path.exists(usdz_path):
            os.unlink(usdz_path)
