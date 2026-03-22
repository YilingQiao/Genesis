"""Unit tests for SceneController with a real Genesis scene."""

import pytest

import genesis as gs
from genesis.vis.controller import SceneController


@pytest.mark.required
def test_scene_controller_exists(show_viewer):
    """Controller is accessible via scene.controller after build."""
    scene = gs.Scene(show_viewer=show_viewer, show_FPS=False)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    assert scene.controller is not None
    assert isinstance(scene.controller, SceneController)
    assert scene.controller._ctx is scene.visualizer._rasterizer._context


@pytest.mark.required
def test_scene_controller_vis_toggles(show_viewer):
    """Visualization toggles: set/get round-trip."""
    scene = gs.Scene(show_viewer=show_viewer, show_FPS=False)
    scene.add_entity(gs.morphs.Plane())
    scene.build()
    c = scene.controller

    # Shadows
    c.set_shadows(True)
    assert c.get_shadows() is True
    c.set_shadows(False)
    assert c.get_shadows() is False

    # World frame
    c.set_world_frame(True)
    assert c.get_world_frame() is True
    c.set_world_frame(False)
    assert c.get_world_frame() is False

    # Link frame
    c.set_link_frame(True)
    assert c.get_link_frame() is True
    c.set_link_frame(False)
    assert c.get_link_frame() is False

    # Camera frustum
    c.set_camera_frustum(True)
    assert c.get_camera_frustum() is True
    c.set_camera_frustum(False)
    assert c.get_camera_frustum() is False

    # Wireframe
    c.set_wireframe(True)
    assert c.get_wireframe() is True
    c.set_wireframe(False)
    assert c.get_wireframe() is False

    # Face normals
    c.set_face_normals(True)
    assert c.get_face_normals() is True
    c.set_face_normals(False)
    assert c.get_face_normals() is False

    # Vertex normals
    c.set_vertex_normals(True)
    assert c.get_vertex_normals() is True
    c.set_vertex_normals(False)
    assert c.get_vertex_normals() is False


@pytest.mark.required
def test_scene_controller_entity_rendering(show_viewer):
    """Entity rendering: vis mode switch, wireframe, contact viz."""
    scene = gs.Scene(show_viewer=show_viewer, show_FPS=False)
    scene.add_entity(gs.morphs.Plane())
    panda = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()
    c = scene.controller

    # Vis mode switch
    c.switch_entity_vis_mode(panda, "collision")
    assert panda.surface.vis_mode == "collision"
    c.switch_entity_vis_mode(panda, "visual")
    assert panda.surface.vis_mode == "visual"

    # Entity wireframe
    c.set_entity_wireframe(panda, True)
    assert c._entity_wireframe.get(panda.idx) is True

    # Contact viz
    c.set_entity_contact_viz(panda, True)
    assert panda._visualize_contact is True
    c.set_entity_contact_viz(panda, False)
    assert panda._visualize_contact is False

    # Refresh visual transforms (should not crash)
    c.refresh_visual_transforms()


@pytest.mark.required
def test_scene_controller_wireframe_interaction(show_viewer):
    """Global wireframe clears per-entity wireframe state."""
    scene = gs.Scene(show_viewer=show_viewer, show_FPS=False)
    scene.add_entity(gs.morphs.Plane())
    panda = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()
    c = scene.controller

    c.set_entity_wireframe(panda, True)
    assert c._entity_wireframe.get(panda.idx) is True

    # Global wireframe clears per-entity state
    c.set_wireframe(True)
    assert len(c._entity_wireframe) == 0


@pytest.mark.required
def test_scene_controller_noop_safety():
    """All methods work when ctx is None (headless)."""
    c = SceneController(None, ctx=None)

    # Setters should not raise
    c.set_shadows(True)
    c.set_world_frame(True)
    c.set_link_frame(True)
    c.set_link_frame_size(0.5)
    c.set_camera_frustum(True)
    c.set_wireframe(True)
    c.set_face_normals(True)
    c.set_vertex_normals(True)
    c.refresh_visual_transforms()

    # Getters return defaults
    assert c.get_shadows() is False
    assert c.get_world_frame() is False
    assert c.get_link_frame() is False
    assert c.get_link_frame_size() == 0.0
    assert c.get_camera_frustum() is False
    assert c.get_wireframe() is False
    assert c.get_face_normals() is False
    assert c.get_vertex_normals() is False


@pytest.mark.required
def test_scene_controller_state_snapshots(show_viewer):
    """State snapshot methods return expected structure."""
    scene = gs.Scene(show_viewer=show_viewer, show_FPS=False)
    scene.add_entity(gs.morphs.Plane())
    scene.add_camera(pos=(2.0, 2.0, 1.5), lookat=(0.0, 0.0, 0.5), res=(640, 480), fov=30)
    scene.build()
    c = scene.controller

    vis = c.get_vis_state()
    assert "shadows" in vis
    assert "world_frame" in vis
    assert "link_frame" in vis
    assert "link_frame_size" in vis
    assert "camera_frustum" in vis
    assert "face_normals" in vis
    assert "vertex_normals" in vis
    assert "wireframe" in vis

    cam = c.get_scene_camera_state()
    assert "pos" in cam
    assert "lookat" in cam
    assert "fov" in cam
    assert len(cam["pos"]) == 3
    assert len(cam["lookat"]) == 3


# ---------------------------------------------------------------------------
# Migration regression assertions (AC-4, AC-5, AC-6, AC-8)
# ---------------------------------------------------------------------------


@pytest.mark.required
def test_migration_server_no_direct_ctx_access():
    """AC-4: server.py has no references to _rasterizer._context (except orthographic)."""
    import inspect
    from genesis.vis.web import server

    source = inspect.getsource(server)
    # Orthographic toggle is the only allowed direct rasterizer access
    lines_with_ctx = [
        line.strip()
        for line in source.splitlines()
        if "_rasterizer._context" in line and "orthographic" not in line.lower() and "toggle_orthographic" not in line
    ]
    assert len(lines_with_ctx) == 0, f"Direct _rasterizer._context access in server.py: {lines_with_ctx}"

    # Dead methods should be gone
    assert not hasattr(server.GenesisWebServer, "_get_ctx")
    assert not hasattr(server.GenesisWebServer, "_toggle_wireframe")
    assert not hasattr(server.GenesisWebServer, "_toggle_render_flag")
    assert not hasattr(server.GenesisWebServer, "_update_visual_transforms")


@pytest.mark.required
def test_migration_scene_ops_no_rendering_functions():
    """AC-8: scene_ops.py has no rendering functions, keeps build_entity_joint_data."""
    from genesis.vis import scene_ops

    # These should NOT be importable
    assert not hasattr(scene_ops, "refresh_visual_transforms")
    assert not hasattr(scene_ops, "switch_entity_vis_mode")
    assert not hasattr(scene_ops, "set_entity_wireframe")
    assert not hasattr(scene_ops, "set_entity_contact_viz")

    # This should still be importable
    assert hasattr(scene_ops, "build_entity_joint_data")
    assert hasattr(scene_ops, "FREE_JOINT_POS_LIMIT")


@pytest.mark.required
def test_migration_viewer_no_shadow_normal_render_flags():
    """AC-6: viewer _default_render_flags has no shadows, face_normals, vertex_normals."""
    from genesis.ext.pyrender.viewer import Viewer

    # Read the source to verify the keys are not in _default_render_flags
    import inspect

    source = inspect.getsource(Viewer)
    # Find the _default_render_flags dict definition
    in_dict = False
    flag_lines = []
    for line in source.splitlines():
        if "_default_render_flags" in line and "{" in line:
            in_dict = True
        if in_dict:
            flag_lines.append(line)
            if "}" in line:
                break
    dict_text = "\n".join(flag_lines)
    assert '"shadows"' not in dict_text, "shadows should not be in _default_render_flags"
    assert '"face_normals"' not in dict_text, "face_normals should not be in _default_render_flags"
    assert '"vertex_normals"' not in dict_text, "vertex_normals should not be in _default_render_flags"
