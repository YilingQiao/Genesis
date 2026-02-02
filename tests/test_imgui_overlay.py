"""Unit tests for ImGuiOverlayPlugin (no imgui-bundle required)."""

import pytest
import genesis as gs


@pytest.mark.required
def test_imgui_overlay_plugin():
    """Test ImGuiOverlayPlugin core logic without requiring imgui-bundle or display."""
    from genesis.ext.pyrender.imgui_overlay import ImGuiOverlayPlugin

    # Note: gs.init() is called automatically by conftest.py's initialize_genesis fixture
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    plugin = ImGuiOverlayPlugin()

    # Test initial state
    assert plugin.paused is False
    assert plugin.speed == 1.0
    assert plugin._available is False

    # Test should_step() logic
    assert plugin.should_step() is True
    plugin.paused = True
    assert plugin.should_step() is False
    plugin._step_requested = True
    assert plugin.should_step() is True
    assert plugin.should_step() is False

    # Test get_speed()
    plugin.speed = 2.5
    assert plugin.get_speed() == 2.5

    # Test _cache_entity_data()
    plugin.scene = scene
    plugin._cache_entity_data()
    assert len(plugin._entity_cache) == 1
    entity_data = list(plugin._entity_cache.values())[0]
    assert "q_names" in entity_data
    assert entity_data["n_qs"] == 9  # Panda: 7 arm + 2 gripper


@pytest.mark.required
def test_imgui_overlay_spherical_joint():
    """Test ImGuiOverlayPlugin handles spherical (ball) joints without IndexError."""
    from genesis.ext.pyrender.imgui_overlay import ImGuiOverlayPlugin

    # Load a model with a ball joint
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.MJCF(file="xml/one_ball_joint.xml"))
    scene.build()

    plugin = ImGuiOverlayPlugin()
    plugin.scene = scene

    # This should NOT raise IndexError for spherical joints
    plugin._cache_entity_data()

    assert len(plugin._entity_cache) == 1
    entity_data = list(plugin._entity_cache.values())[0]

    # Ball joint has 4 quaternion components (qw, qx, qy, qz)
    # Check that all are marked as quaternion (read-only)
    assert entity_data["n_qs"] == 4
    assert all(entity_data["q_is_quaternion"])  # All should be True


@pytest.mark.required
def test_imgui_overlay_multi_env():
    """Test ImGuiOverlayPlugin handles multi-env scenes correctly."""
    from genesis.ext.pyrender.imgui_overlay import ImGuiOverlayPlugin
    import numpy as np

    # Create a multi-env scene
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build(n_envs=4)

    plugin = ImGuiOverlayPlugin()
    plugin.scene = scene

    plugin._cache_entity_data()
    assert len(plugin._entity_cache) == 1

    entity_data = list(plugin._entity_cache.values())[0]
    entity = entity_data["entity"]

    # Get qpos - should be 2D with shape [n_envs, n_qs]
    qpos_tensor = entity.get_qpos()
    qpos_np = qpos_tensor.cpu().numpy()

    # Verify qpos is 2D for multi-env
    assert qpos_np.ndim == 2
    assert qpos_np.shape[0] == 4  # n_envs
    assert qpos_np.shape[1] == 9  # n_qs for Panda

    # Verify plugin correctly extracts env 0 only
    # (simulating what _render_joint_panel does)
    qpos_env0 = qpos_np[0]
    assert qpos_env0.shape == (9,)

    # Verify set_qpos with envs_idx=0 doesn't crash
    new_qpos = qpos_env0.copy()
    new_qpos[0] = 0.1  # Change first joint slightly
    entity.set_qpos(new_qpos, envs_idx=0)

    # Verify only env 0 was changed
    updated_qpos = entity.get_qpos().cpu().numpy()
    assert np.isclose(updated_qpos[0, 0], 0.1)
    # Other envs should still have original value
    for env_idx in range(1, 4):
        assert not np.isclose(updated_qpos[env_idx, 0], 0.1)


@pytest.mark.required
def test_imgui_overlay_single_env_update():
    """Test ImGuiOverlayPlugin handles single-env joint updates without envs_idx.

    Regression test: In single-env scenes (n_envs=0), set_qpos must be called
    WITHOUT envs_idx parameter, otherwise Scene._sanitize_envs_idx raises.
    """
    from genesis.ext.pyrender.imgui_overlay import ImGuiOverlayPlugin
    import numpy as np

    # Create a single-env scene (default, n_envs not specified)
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()  # No n_envs = single-env

    plugin = ImGuiOverlayPlugin()
    plugin.scene = scene

    plugin._cache_entity_data()
    assert len(plugin._entity_cache) == 1

    entity_data = list(plugin._entity_cache.values())[0]
    entity = entity_data["entity"]

    # Get qpos - should be 1D for single-env
    qpos_tensor = entity.get_qpos()
    qpos_np = qpos_tensor.cpu().numpy()

    # Verify qpos is 1D for single-env
    assert qpos_np.ndim == 1
    assert qpos_np.shape[0] == 9  # n_qs for Panda

    # Verify set_qpos without envs_idx works (this would crash before the fix)
    new_qpos = qpos_np.copy()
    new_qpos[0] = 0.2  # Change first joint slightly
    entity.set_qpos(new_qpos)  # No envs_idx - must work for single-env

    # Verify the change was applied
    updated_qpos = entity.get_qpos().cpu().numpy()
    assert np.isclose(updated_qpos[0], 0.2)
