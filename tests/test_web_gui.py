"""Unit tests for the Genesis Web GUI module.

All tests use pure stubs/mocks — no gs.Scene.build(), no EGL, no GPU.
"""

import threading

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import genesis as gs


# ---------------------------------------------------------------------------
# Import / enum tests (no scene needed)
# ---------------------------------------------------------------------------


@pytest.mark.required
def test_web_gui_imports():
    """Verify all web GUI modules can be imported."""
    from genesis.vis.web.protocol import MsgType, build_scene_info, build_state_update
    from genesis.vis.web.frame_producer import FrameProducer
    from genesis.vis.web.server import GenesisWebServer

    assert MsgType.SIM_CONTROL.value == "sim_control"
    assert MsgType.CAMERA_UPDATE.value == "camera_update"
    assert MsgType.STATE_UPDATE.value == "state_update"
    assert MsgType.SCENE_INFO.value == "scene_info"


# ---------------------------------------------------------------------------
# Protocol tests (use stubbed scene objects)
# ---------------------------------------------------------------------------


def _make_stub_scene_entity(name="robot", idx=0, n_dofs=9, n_qs=9, joints=None):
    """Create a stub entity for build_scene_info."""
    entity = MagicMock()
    entity.name = name
    entity.idx = idx
    entity.n_dofs = n_dofs
    entity.n_qs = n_qs
    entity.visualize_contact = False

    surface = MagicMock()
    surface.vis_mode = "visual"
    entity.surface = surface

    if joints is None:
        joints = []
        for i in range(n_dofs):
            j = MagicMock()
            j.name = f"joint{i}"
            j.type = gs.JOINT_TYPE.REVOLUTE
            j.n_dofs = 1
            j.n_qs = 1
            j.dofs_limit = np.array([[-3.14, 3.14]])
            joints.append(j)
    entity.joints = joints

    qpos_tensor = MagicMock()
    qpos_tensor.cpu.return_value.numpy.return_value = np.zeros(n_qs)
    entity.get_qpos.return_value = qpos_tensor

    return entity


def _make_stub_protocol_scene(entities=None, n_envs=1):
    """Create a stub scene for build_scene_info.

    Uses a mock controller for vis/camera state (post-migration architecture).
    """
    scene = MagicMock()
    scene.entities = entities if entities is not None else []
    scene.n_envs = n_envs
    # Controller provides vis/camera state
    scene.controller.get_vis_state.return_value = {
        "shadows": False,
        "world_frame": False,
        "link_frame": False,
        "link_frame_size": 0.1,
        "camera_frustum": False,
        "face_normals": False,
        "vertex_normals": False,
        "wireframe": False,
        "orthographic": False,
    }
    scene.controller.get_scene_camera_state.return_value = {}
    return scene


@pytest.mark.required
def test_protocol_build_scene_info():
    """Test build_scene_info produces correct structure with a stubbed entity."""
    from genesis.vis.web.protocol import build_scene_info

    entity = _make_stub_scene_entity(name="panda", idx=0, n_dofs=9, n_qs=9)
    scene = _make_stub_protocol_scene(entities=[entity])

    info = build_scene_info(scene)
    assert info["type"] == "scene_info"
    assert len(info["entities"]) >= 1

    panda_info = info["entities"][0]
    assert "n_dofs" in panda_info
    assert panda_info["n_dofs"] == 9
    assert "joints" in panda_info
    assert len(panda_info["joints"]) > 0


@pytest.mark.required
def test_protocol_build_state_update():
    """Test build_state_update produces correct structure."""
    from genesis.vis.web.protocol import build_state_update

    msg = build_state_update(sim_time=1.5, step=100, fps=30.0, paused=True)
    assert msg["type"] == "state_update"
    assert msg["time"] == 1.5
    assert msg["step"] == 100
    assert msg["fps"] == 30.0
    assert msg["paused"] is True


# ---------------------------------------------------------------------------
# FrameProducer tests (uses stub scene — FrameProducer.__init__ is lightweight)
# ---------------------------------------------------------------------------


@pytest.mark.required
def test_frame_producer_consume_empty():
    """Test that consume_frame returns None when no frame has been produced."""
    from genesis.vis.web.frame_producer import FrameProducer

    mock_scene = MagicMock()
    producer = FrameProducer(mock_scene, resolution=(320, 240))
    assert producer.consume_frame() is None


@pytest.mark.required
def test_frame_producer_thread_safety():
    """Test that the latest-frame buffer is thread-safe."""
    from genesis.vis.web.frame_producer import FrameProducer

    mock_scene = MagicMock()
    producer = FrameProducer(mock_scene, resolution=(320, 240))

    # Simulate producing a frame by writing directly to the buffer
    test_data = b"fake-jpeg-data"
    with producer._lock:
        producer._latest_frame = test_data

    # Consume from another thread
    result = [None]

    def consumer():
        result[0] = producer.consume_frame()

    t = threading.Thread(target=consumer)
    t.start()
    t.join(timeout=2.0)

    assert result[0] == test_data
    # After consumption, buffer should be empty
    assert producer.consume_frame() is None


# ---------------------------------------------------------------------------
# Server tests — stubbed (no real scene)
# ---------------------------------------------------------------------------


def _make_stub_server(entities=None, n_envs=1, cameras=None):
    """Create a GenesisWebServer with stubbed scene — no renderer required.

    Post-migration: uses scene.controller for vis/entity operations.
    """
    from genesis.vis.web.server import GenesisWebServer

    server = object.__new__(GenesisWebServer)
    server.scene = MagicMock()
    server.scene.entities = entities if entities is not None else []
    server.scene.n_envs = n_envs
    server.scene.visualizer.cameras = cameras if cameras is not None else []
    # Mock controller for vis/entity operations
    server.scene.controller = MagicMock()
    server._paused = False
    server._step_requested = False
    server._reset_requested = False
    server._pending_commands = []
    server._commands_lock = threading.Lock()
    server._initial_camera_pos = None
    server._initial_camera_lookat = None
    server._initial_camera_fov = None
    server._perspective_camera = None
    return server


def _make_stub_entity(n_qs=9, n_envs=1):
    """Create a stub entity that records get_qpos/set_qpos calls."""
    entity = MagicMock()
    entity.n_qs = n_qs

    if n_envs > 1:
        qpos_data = np.zeros((n_envs, n_qs))
    else:
        qpos_data = np.zeros(n_qs)

    def fake_get_qpos(**kwargs):
        return qpos_data.copy()

    entity.get_qpos = MagicMock(side_effect=fake_get_qpos)
    entity.set_qpos = MagicMock()
    entity.has_set_qpos = True
    return entity


def _make_numeric_camera(pos=None, lookat=None, up=None, fov=30.0):
    """Create a camera stub with numeric numpy attributes for orbit/pan/zoom."""
    camera = MagicMock()
    camera.pos = np.array(pos if pos is not None else [3.0, 0.0, 2.0], dtype=np.float32)
    camera.lookat = np.array(lookat if lookat is not None else [0.0, 0.0, 0.0], dtype=np.float32)
    camera.up = np.array(up if up is not None else [0.0, 0.0, 1.0], dtype=np.float32)
    camera.fov = fov
    return camera


@pytest.mark.required
def test_server_cooperative_control():
    """Test should_step() and sim control commands using Pydantic models."""
    from genesis.vis.web.protocol import SimControlMsg

    server = _make_stub_server()

    # Initially not paused
    assert server.should_step() is True

    # Simulate pause command
    server._enqueue_command(SimControlMsg(action="pause"))
    server.process_commands()
    assert server.should_step() is False

    # Simulate step command (single step while paused)
    server._enqueue_command(SimControlMsg(action="step"))
    server.process_commands()
    assert server.should_step() is True  # consumes the step
    assert server.should_step() is False  # back to paused

    # Simulate play command
    server._enqueue_command(SimControlMsg(action="play"))
    server.process_commands()
    assert server.should_step() is True


@pytest.mark.required
def test_lazy_import_wrapper():
    """Test that the __init__.py lazy import wrapper works."""
    from genesis.vis.web import GenesisWebServer

    # The wrapper is a factory function — calling it should produce a server instance
    mock_scene = MagicMock()
    server = GenesisWebServer(mock_scene, port=0)
    assert hasattr(server, "should_step")
    assert hasattr(server, "process_commands")
    assert hasattr(server, "start")


# ---------------------------------------------------------------------------
# Camera handler tests
# ---------------------------------------------------------------------------


@pytest.mark.required
def test_camera_update_no_cameras():
    """Test _handle_camera_update returns safely when scene has no cameras."""
    from genesis.vis.web.protocol import CameraUpdateMsg

    server = _make_stub_server(cameras=[])
    cmd = CameraUpdateMsg(action="orbit", d_azimuth=1.0, d_elevation=0.0)

    # Should return without error (early return guard)
    server._handle_camera_update(cmd)


@pytest.mark.required
def test_camera_update_with_cameras():
    """Test _handle_camera_update: orbit action calls set_pose on the camera."""
    from genesis.vis.web.protocol import CameraUpdateMsg

    camera = _make_numeric_camera(pos=[3.0, 0.0, 2.0], lookat=[0.0, 0.0, 0.0])
    server = _make_stub_server(cameras=[camera])

    cmd = CameraUpdateMsg(action="orbit", d_azimuth=0.1, d_elevation=0.05)
    server._handle_camera_update(cmd)

    # Verify set_pose was called with new pos and lookat
    assert camera.set_pose.call_count == 1
    call_kwargs = camera.set_pose.call_args[1]
    assert "pos" in call_kwargs
    assert "lookat" in call_kwargs


# ---------------------------------------------------------------------------
# Entity update handler tests
# ---------------------------------------------------------------------------


@pytest.mark.required
def test_entity_update_valid_single_env():
    """Test _handle_entity_update accepts valid single-DOF and full qpos for single-env."""
    from genesis.vis.web.protocol import EntityUpdateMsg

    entity = _make_stub_entity(n_qs=9, n_envs=1)
    server = _make_stub_server(entities=[entity], n_envs=1)

    # Valid single-DOF update
    cmd = EntityUpdateMsg(entity_idx=0, dof_idx=0, value=0.5)
    server._handle_entity_update(cmd)
    assert entity.set_qpos.call_count == 1
    _, kwargs = entity.set_qpos.call_args
    assert "envs_idx" not in kwargs, "envs_idx should NOT be passed for single-env"

    # Valid full qpos update
    entity.set_qpos.reset_mock()
    cmd = EntityUpdateMsg(entity_idx=0, qpos=[0.0] * 9)
    server._handle_entity_update(cmd)
    assert entity.set_qpos.call_count == 1
    _, kwargs = entity.set_qpos.call_args
    assert "envs_idx" not in kwargs, "envs_idx should NOT be passed for single-env"


@pytest.mark.required
def test_entity_update_multi_env_full_qpos():
    """Test multi-env full-qpos: set_qpos called with envs_idx=0."""
    from genesis.vis.web.protocol import EntityUpdateMsg

    entity = _make_stub_entity(n_qs=9, n_envs=4)
    server = _make_stub_server(entities=[entity], n_envs=4)

    cmd = EntityUpdateMsg(entity_idx=0, qpos=[0.1] * 9)
    server._handle_entity_update(cmd)
    assert entity.set_qpos.call_count == 1
    _, kwargs = entity.set_qpos.call_args
    assert kwargs.get("envs_idx") == 0, "multi-env full-qpos must pass envs_idx=0"


@pytest.mark.required
def test_entity_update_multi_env_single_dof():
    """Test multi-env single-DOF: reads batched qpos, extracts env-0, writes with envs_idx=0."""
    from genesis.vis.web.protocol import EntityUpdateMsg

    entity = _make_stub_entity(n_qs=9, n_envs=4)
    server = _make_stub_server(entities=[entity], n_envs=4)

    cmd = EntityUpdateMsg(entity_idx=0, dof_idx=2, value=0.77)
    server._handle_entity_update(cmd)

    # get_qpos must be called
    assert entity.get_qpos.call_count == 1

    # set_qpos must be called WITH envs_idx=0
    assert entity.set_qpos.call_count == 1
    set_args, set_kwargs = entity.set_qpos.call_args
    assert set_kwargs.get("envs_idx") == 0, "multi-env single-DOF must write with envs_idx=0"

    # The written qpos should be 1D (env-0 extracted) with dof 2 set to 0.77
    written_qpos = set_args[0]
    assert len(written_qpos) == 9, "written qpos should be 1D with n_qs elements"
    assert written_qpos[2] == 0.77


@pytest.mark.required
def test_entity_update_vis_mode_uses_controller():
    """Test vis_mode update routes through scene.controller."""
    from genesis.vis.web.protocol import EntityUpdateMsg

    entity = _make_stub_entity()
    server = _make_stub_server(entities=[entity])

    cmd = EntityUpdateMsg(entity_idx=0, vis_mode="collision")
    server._handle_entity_update(cmd)

    server.scene.controller.switch_entity_vis_mode.assert_called_once_with(entity, "collision")


@pytest.mark.required
def test_entity_update_wireframe_uses_controller():
    """Test wireframe update routes through scene.controller."""
    from genesis.vis.web.protocol import EntityUpdateMsg

    entity = _make_stub_entity()
    server = _make_stub_server(entities=[entity])

    cmd = EntityUpdateMsg(entity_idx=0, wireframe=True)
    server._handle_entity_update(cmd)

    server.scene.controller.set_entity_wireframe.assert_called_once_with(entity, True)


@pytest.mark.required
def test_vis_toggle_uses_controller():
    """Test vis toggles route through scene.controller."""
    from genesis.vis.web.protocol import VisToggleMsg

    server = _make_stub_server()

    toggles = [
        ("shadows", True, "set_shadows"),
        ("wireframe", True, "set_wireframe"),
        ("world_frame", True, "set_world_frame"),
        ("link_frame", False, "set_link_frame"),
        ("camera_frustum", True, "set_camera_frustum"),
        ("face_normals", True, "set_face_normals"),
        ("vertex_normals", False, "set_vertex_normals"),
    ]

    for prop, value, method_name in toggles:
        server.scene.controller.reset_mock()
        cmd = VisToggleMsg(property=prop, value=value)
        server._handle_vis_toggle(cmd)
        getattr(server.scene.controller, method_name).assert_called_once_with(bool(value))
