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
