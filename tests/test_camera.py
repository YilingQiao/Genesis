"""
Tests for camera behavior, specifically the set_pose() method.

These tests verify the fix for commit 8f44cea which introduced a regression
where the camera's up vector gets unconditionally overwritten with the computed
Y-axis from the transform matrix, causing yaw/roll drift during animation.
"""

import numpy as np
import pytest

import genesis as gs

from .utils import assert_allclose


@pytest.fixture
def camera_scene(show_viewer):
    """Create a minimal scene with a camera for testing."""
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, 0.0, 2.0),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    # Add a simple plane so we have something to render
    scene.add_entity(gs.morphs.Plane())

    # Add camera for testing
    cam = scene.add_camera(
        res=(320, 240),
        pos=(2.0, 0.0, 1.0),
        lookat=(0.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        fov=30,
    )

    scene.build()

    yield scene, cam


def test_camera_no_yaw_drift_without_explicit_up(camera_scene):
    """
    When set_pose() is called with only pos/lookat (no up),
    the stored up vector should NOT change.

    This test fails on current code (after commit 8f44cea) because
    the up vector is unconditionally overwritten with transform[..., :3, 1].

    The expected behavior is that when up is not provided, the stored
    up vector should remain unchanged.
    """
    scene, cam = camera_scene

    # Set a known starting position with explicit up vector
    # Use a position where the camera is ABOVE the lookat point (looking down)
    # This creates a non-trivial up vector that will drift if the bug exists
    cam.set_pose(pos=(2.0, 0.0, 1.5), lookat=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0))

    # Get the up vector after the first set_pose with explicit up
    # Note: This will be orthogonalized, not exactly (0,0,1)
    initial_up = cam.up.copy()

    # Orbit camera around target at multiple positions without specifying up
    # Camera is above target, so it has a vertical view component
    orbit_positions = [
        (2.0, 0.0, 1.5),  # Start position
        (1.5, 1.5, 1.5),  # 45 degrees
        (0.0, 2.0, 1.5),  # 90 degrees
        (-1.5, 1.5, 1.5),  # 135 degrees
        (-2.0, 0.0, 1.5),  # 180 degrees
        (-1.5, -1.5, 1.5),  # 225 degrees
        (0.0, -2.0, 1.5),  # 270 degrees
        (1.5, -1.5, 1.5),  # 315 degrees
        (2.0, 0.0, 1.5),  # Back to start (360 degrees)
    ]

    for pos in orbit_positions:
        cam.set_pose(pos=pos, lookat=(0.0, 0.0, 0.0))
        current_up = cam.up

        # The up vector should remain unchanged when not explicitly provided
        assert_allclose(
            current_up,
            initial_up,
            tol=1e-6,
            err_msg=f"Up vector drifted at position {pos}: expected {initial_up}, got {current_up}",
        )


def test_camera_up_orthogonalized_when_explicit(camera_scene):
    """
    When set_pose() is called with explicit up vector,
    the stored up should be orthogonalized (perpendicular to view direction).

    This ensures smooth animations when the user explicitly provides up.
    """
    scene, cam = camera_scene

    # Set a position where the up vector won't be perfectly aligned
    pos = np.array([2.0, 2.0, 2.0])
    lookat = np.array([0.0, 0.0, 0.0])

    # Provide an up vector that's not perfectly orthogonal to view direction
    # The view direction is lookat - pos = (-2, -2, -2) normalized
    provided_up = np.array([0.0, 0.0, 1.0])

    cam.set_pose(pos=pos, lookat=lookat, up=provided_up)

    stored_up = cam.up
    # Compute view direction (from camera to lookat, then negate for OpenGL convention)
    view_dir = pos - lookat
    view_dir = view_dir / np.linalg.norm(view_dir)

    # The stored up should be perpendicular to the view direction
    # (dot product should be approximately 0)
    dot_product = np.dot(stored_up, view_dir)
    assert_allclose(
        dot_product,
        0.0,
        tol=1e-5,
        err_msg=f"Stored up {stored_up} is not perpendicular to view direction {view_dir}. Dot product: {dot_product}",
    )

    # The stored up should also be a unit vector
    up_norm = np.linalg.norm(stored_up)
    assert_allclose(
        up_norm,
        1.0,
        tol=1e-6,
        err_msg=f"Stored up vector is not normalized: norm = {up_norm}",
    )


def test_camera_smooth_orbit_no_discontinuity(camera_scene):
    """
    Simulating smooth camera orbit should not cause
    sudden jumps in camera orientation.

    "Smooth" is defined as: when orbiting in small increments,
    the angular change between consecutive transforms should be
    proportional to the orbit step size.

    This test verifies that there are no sudden jumps or discontinuities
    in the camera's rotation when orbiting smoothly.
    """
    scene, cam = camera_scene

    # Set initial position with explicit up
    # Camera is above the lookat point (looking down) to have a non-trivial up vector
    radius = 2.0
    height = 1.5
    lookat = (0.0, 0.0, 0.0)

    initial_pos = np.array([radius, 0.0, height])
    cam.set_pose(pos=initial_pos, lookat=lookat, up=(0.0, 0.0, 1.0))

    prev_transform = cam.transform.copy()

    # Orbit in small 5-degree increments
    num_steps = 72  # Full 360 degree orbit
    angle_step = 2 * np.pi / num_steps

    max_angular_change = 0.0
    expected_max_change = angle_step * 2  # Allow some tolerance

    for i in range(1, num_steps + 1):
        angle = i * angle_step
        pos = (radius * np.cos(angle), radius * np.sin(angle), height)

        cam.set_pose(pos=pos, lookat=lookat)

        current_transform = cam.transform

        # Extract rotation matrices
        prev_rot = prev_transform[:3, :3]
        curr_rot = current_transform[:3, :3]

        # Compute relative rotation
        relative_rot = curr_rot @ prev_rot.T

        # Compute angle of rotation (using trace of rotation matrix)
        # trace(R) = 1 + 2*cos(theta)
        trace = np.trace(relative_rot)
        # Clamp to valid range for arccos
        cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
        angular_change = np.arccos(cos_angle)

        max_angular_change = max(max_angular_change, angular_change)

        # Check that angular change is not too large (no sudden jumps)
        # For a 5-degree orbit step, we expect roughly 5-degree rotation change
        # Allow up to 3x the expected change for numerical tolerance
        assert angular_change < expected_max_change * 3, (
            f"Sudden jump detected at step {i}: angular change = {np.degrees(angular_change):.2f} degrees, "
            f"expected max = {np.degrees(expected_max_change * 3):.2f} degrees"
        )

        prev_transform = current_transform.copy()

    # Verify we actually measured some rotation (sanity check)
    assert max_angular_change > 0.01, "No rotation detected during orbit - test may be broken"


def test_camera_up_preserved_after_transform_set_pose(camera_scene):
    """
    When set_pose() is called with a transform matrix (no explicit up),
    the stored up vector should be extracted from the transform's Y-axis
    but should not cause drift in subsequent pos/lookat calls.
    """
    scene, cam = camera_scene

    # First set pose using pos/lookat/up with camera above lookat (looking down)
    cam.set_pose(pos=(2.0, 0.0, 1.5), lookat=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0))
    up_after_plu = cam.up.copy()

    # Now set pose using a transform matrix
    transform = cam.transform.copy()
    cam.set_pose(transform=transform)

    # Get up after transform-based set_pose
    up_after_transform = cam.up

    # Now do another pos/lookat call without up (different position, same height above lookat)
    cam.set_pose(pos=(0.0, 2.0, 1.5), lookat=(0.0, 0.0, 0.0))
    up_after_second_plu = cam.up

    # The up vector should be consistent throughout
    # (or at least not drift unexpectedly)
    assert_allclose(
        up_after_second_plu,
        up_after_plu,
        tol=1e-5,
        err_msg=f"Up vector changed unexpectedly: was {up_after_plu}, now {up_after_second_plu}",
    )
