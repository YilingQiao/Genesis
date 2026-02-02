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
