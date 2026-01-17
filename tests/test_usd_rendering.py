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
