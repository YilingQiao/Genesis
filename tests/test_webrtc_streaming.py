import asyncio
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def initialize_genesis():
    # Override tests/conftest.py autouse fixture: this unit test only validates the streaming app wiring.
    yield


def _load_webrtc_module():
    pytest.importorskip("aiortc")
    pytest.importorskip("aiohttp")
    pytest.importorskip("av")

    module_path = Path(__file__).resolve().parents[1] / "genesis" / "vis" / "streaming" / "webrtc_aiortc.py"
    spec = importlib.util.spec_from_file_location("genesis.vis.streaming.webrtc_aiortc", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_streaming_common_module():
    module_path = Path(__file__).resolve().parents[1] / "genesis" / "vis" / "streaming" / "common.py"
    spec = importlib.util.spec_from_file_location("genesis.vis.streaming.common", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyCamera:
    res = (64, 48)

    def render(self, rgb=True, depth=False, segmentation=False, normal=False, force_render=False):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        return frame, None, None, None


def _collect_routes(app) -> set[tuple[str, str]]:
    routes = set()
    for route in app.router.routes():
        info = route.get_info()
        if "path" in info:
            routes.add((route.method, info["path"]))
        elif "formatter" in info:
            routes.add((route.method, info["formatter"]))
    return routes


# ---------------------------------------------------------------------------
# FrameBuffer tests
# ---------------------------------------------------------------------------


def test_frame_buffer_empty_on_init():
    module = _load_webrtc_module()
    buf = module.FrameBuffer()
    assert buf.consume() is None


def test_frame_buffer_produce_consume():
    module = _load_webrtc_module()
    buf = module.FrameBuffer()
    frame = np.ones((48, 64, 3), dtype=np.uint8) * 42
    buf.produce(frame)
    result = buf.consume()
    assert result is not None
    np.testing.assert_array_equal(result, frame)


def test_frame_buffer_consume_is_non_destructive():
    module = _load_webrtc_module()
    buf = module.FrameBuffer()
    frame = np.ones((48, 64, 3), dtype=np.uint8)
    buf.produce(frame)
    first = buf.consume()
    second = buf.consume()
    assert first is not None and second is not None
    np.testing.assert_array_equal(first, second)


def test_frame_buffer_clear():
    module = _load_webrtc_module()
    buf = module.FrameBuffer()
    buf.produce(np.zeros((48, 64, 3), dtype=np.uint8))
    buf.clear()
    assert buf.consume() is None


def test_frame_buffer_produce_overwrites():
    module = _load_webrtc_module()
    buf = module.FrameBuffer()
    frame1 = np.ones((48, 64, 3), dtype=np.uint8) * 1
    frame2 = np.ones((48, 64, 3), dtype=np.uint8) * 2
    buf.produce(frame1)
    buf.produce(frame2)
    result = buf.consume()
    assert result is not None
    np.testing.assert_array_equal(result, frame2)


# ---------------------------------------------------------------------------
# GenesisCameraVideoTrack tests
# ---------------------------------------------------------------------------


def test_track_recv_returns_black_frame_when_buffer_empty():
    module = _load_webrtc_module()
    buf = module.FrameBuffer()
    track = module.GenesisCameraVideoTrack(buf, width=64, height=48, fps=30)

    frame = asyncio.get_event_loop().run_until_complete(track.recv())
    arr = frame.to_ndarray(format="rgb24")
    assert arr.shape == (48, 64, 3)
    assert arr.sum() == 0


def test_track_recv_returns_buffer_frame():
    module = _load_webrtc_module()
    buf = module.FrameBuffer()
    expected = np.ones((48, 64, 3), dtype=np.uint8) * 128
    buf.produce(expected)
    track = module.GenesisCameraVideoTrack(buf, width=64, height=48, fps=30)

    frame = asyncio.get_event_loop().run_until_complete(track.recv())
    arr = frame.to_ndarray(format="rgb24")
    np.testing.assert_array_equal(arr, expected)


def test_track_pts_increments():
    module = _load_webrtc_module()
    buf = module.FrameBuffer()
    buf.produce(np.zeros((48, 64, 3), dtype=np.uint8))
    track = module.GenesisCameraVideoTrack(buf, width=64, height=48, fps=30)

    loop = asyncio.get_event_loop()
    f1 = loop.run_until_complete(track.recv())
    f2 = loop.run_until_complete(track.recv())
    assert f1.pts == 0
    assert f2.pts == 1


def test_track_rejects_invalid_fps():
    module = _load_webrtc_module()
    buf = module.FrameBuffer()
    with pytest.raises(ValueError, match="fps must be > 0"):
        module.GenesisCameraVideoTrack(buf, width=64, height=48, fps=0)


def test_track_rejects_fractional_fps_that_truncates_to_zero():
    module = _load_webrtc_module()
    buf = module.FrameBuffer()
    with pytest.raises(ValueError, match="fps must be > 0"):
        module.GenesisCameraVideoTrack(buf, width=64, height=48, fps=0.5)


# ---------------------------------------------------------------------------
# Multiple tracks share one buffer
# ---------------------------------------------------------------------------


def test_multiple_tracks_share_buffer():
    module = _load_webrtc_module()
    buf = module.FrameBuffer()
    expected = np.ones((48, 64, 3), dtype=np.uint8) * 200
    buf.produce(expected)

    track_a = module.GenesisCameraVideoTrack(buf, width=64, height=48, fps=30)
    track_b = module.GenesisCameraVideoTrack(buf, width=64, height=48, fps=30)

    loop = asyncio.get_event_loop()
    frame_a = loop.run_until_complete(track_a.recv())
    frame_b = loop.run_until_complete(track_b.recv())

    np.testing.assert_array_equal(
        frame_a.to_ndarray(format="rgb24"),
        frame_b.to_ndarray(format="rgb24"),
    )


# ---------------------------------------------------------------------------
# WebRTCStreamer route tests (constructor unchanged — still takes camera)
# ---------------------------------------------------------------------------


def test_webrtc_streamer_routes_created():
    module = _load_webrtc_module()
    WebRTCStreamer = module.WebRTCStreamer

    streamer = WebRTCStreamer(camera=DummyCamera(), host="127.0.0.1", port=0, fps=30)
    app = streamer.create_app()
    routes = _collect_routes(app)

    assert ("GET", "/") in routes
    assert ("POST", "/offer") in routes
    assert ("POST", "/shutdown") not in routes


def test_webrtc_streamer_shutdown_route_enabled():
    module = _load_webrtc_module()
    WebRTCStreamer = module.WebRTCStreamer

    streamer = WebRTCStreamer(
        camera=DummyCamera(),
        host="127.0.0.1",
        port=0,
        fps=30,
        allow_browser_shutdown=True,
    )
    app = streamer.create_app()
    routes = _collect_routes(app)

    assert ("GET", "/") in routes
    assert ("POST", "/offer") in routes
    assert ("POST", "/shutdown") in routes


def test_webrtc_streamer_has_frame_buffer():
    module = _load_webrtc_module()
    streamer = module.WebRTCStreamer(camera=DummyCamera(), host="127.0.0.1", port=0, fps=30)
    assert isinstance(streamer._frame_buffer, module.FrameBuffer)
    assert streamer._stopped is False
    assert streamer._render_task is None


def test_webrtc_streamer_rejects_invalid_fps():
    module = _load_webrtc_module()
    with pytest.raises(ValueError, match="fps must be > 0"):
        module.WebRTCStreamer(camera=DummyCamera(), host="127.0.0.1", port=0, fps=0)


def test_webrtc_streamer_rejects_fractional_fps_that_truncates_to_zero():
    module = _load_webrtc_module()
    with pytest.raises(ValueError, match="fps must be > 0"):
        module.WebRTCStreamer(camera=DummyCamera(), host="127.0.0.1", port=0, fps=0.5)


def test_webrtc_index_html_has_chrome_playback_fallback():
    module = _load_webrtc_module()
    html = module.WebRTCStreamer._index_html(ice_servers=[], allow_shutdown=False)

    assert "video.muted = true;" in html
    assert "fallbackStream" in html
    assert "await video.play();" in html


# ---------------------------------------------------------------------------
# Streaming common tests
# ---------------------------------------------------------------------------


def test_build_stream_url_wildcard_host_uses_detected_ip():
    module = _load_streaming_common_module()
    module._detect_non_loopback_ipv4 = lambda: "10.20.30.40"

    url = module.build_stream_url(host="0.0.0.0", port=8000)

    assert url == "http://10.20.30.40:8000"


def test_build_stream_url_wildcard_host_falls_back_to_loopback():
    module = _load_streaming_common_module()
    module._detect_non_loopback_ipv4 = lambda: None

    url = module.build_stream_url(host="0.0.0.0", port=8000)

    assert url == "http://127.0.0.1:8000"
