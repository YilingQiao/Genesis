#!/usr/bin/env python
"""Standalone test runner for webrtc_aiortc — no pytest, no plugins, no sockets required.

Usage:
    python tests/run_webrtc_tests_standalone.py

This script can verify AC6 in environments where pytest plugins (xdist, rerunfailures)
fail due to socket permission errors.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_webrtc_module():
    module_path = Path(__file__).resolve().parents[1] / "genesis" / "vis" / "streaming" / "webrtc_aiortc.py"
    spec = importlib.util.spec_from_file_location("genesis.vis.streaming.webrtc_aiortc", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyCamera:
    res = (64, 48)

    def render(self, **kw):
        return np.zeros((48, 64, 3), dtype=np.uint8), None, None, None


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_buffer_empty_on_init(m):
    assert m.FrameBuffer().consume() is None


def test_buffer_produce_consume(m):
    b = m.FrameBuffer()
    f = np.ones((48, 64, 3), dtype=np.uint8) * 42
    b.produce(f)
    np.testing.assert_array_equal(b.consume(), f)


def test_buffer_non_destructive(m):
    b = m.FrameBuffer()
    f = np.ones((48, 64, 3), dtype=np.uint8)
    b.produce(f)
    np.testing.assert_array_equal(b.consume(), b.consume())


def test_buffer_clear(m):
    b = m.FrameBuffer()
    b.produce(np.zeros((48, 64, 3), dtype=np.uint8))
    b.clear()
    assert b.consume() is None


def test_buffer_overwrite(m):
    b = m.FrameBuffer()
    b.produce(np.ones((48, 64, 3), dtype=np.uint8) * 1)
    b.produce(np.ones((48, 64, 3), dtype=np.uint8) * 2)
    np.testing.assert_array_equal(b.consume(), np.ones((48, 64, 3), dtype=np.uint8) * 2)


def test_track_black_frame_when_empty(m):
    b = m.FrameBuffer()
    t = m.GenesisCameraVideoTrack(b, width=64, height=48, fps=30)
    f = asyncio.new_event_loop().run_until_complete(t.recv())
    assert f.to_ndarray(format="rgb24").sum() == 0


def test_track_buffer_frame(m):
    b = m.FrameBuffer()
    exp = np.ones((48, 64, 3), dtype=np.uint8) * 128
    b.produce(exp)
    t = m.GenesisCameraVideoTrack(b, width=64, height=48, fps=30)
    f = asyncio.new_event_loop().run_until_complete(t.recv())
    np.testing.assert_array_equal(f.to_ndarray(format="rgb24"), exp)


def test_track_pts_increments(m):
    b = m.FrameBuffer()
    b.produce(np.zeros((48, 64, 3), dtype=np.uint8))
    t = m.GenesisCameraVideoTrack(b, width=64, height=48, fps=30)
    loop = asyncio.new_event_loop()
    f1 = loop.run_until_complete(t.recv())
    f2 = loop.run_until_complete(t.recv())
    assert f1.pts == 0 and f2.pts == 1


def test_track_rejects_fps_zero(m):
    try:
        m.GenesisCameraVideoTrack(m.FrameBuffer(), width=64, height=48, fps=0)
        raise AssertionError("should raise ValueError")
    except ValueError:
        pass


def test_track_rejects_fractional_fps(m):
    try:
        m.GenesisCameraVideoTrack(m.FrameBuffer(), width=64, height=48, fps=0.5)
        raise AssertionError("should raise ValueError")
    except ValueError:
        pass


def test_multi_track_shared_buffer(m):
    b = m.FrameBuffer()
    exp = np.ones((48, 64, 3), dtype=np.uint8) * 200
    b.produce(exp)
    ta = m.GenesisCameraVideoTrack(b, width=64, height=48, fps=30)
    tb = m.GenesisCameraVideoTrack(b, width=64, height=48, fps=30)
    loop = asyncio.new_event_loop()
    np.testing.assert_array_equal(
        loop.run_until_complete(ta.recv()).to_ndarray(format="rgb24"),
        loop.run_until_complete(tb.recv()).to_ndarray(format="rgb24"),
    )


def test_streamer_has_frame_buffer(m):
    s = m.WebRTCStreamer(camera=DummyCamera(), host="127.0.0.1", port=0, fps=30)
    assert isinstance(s._frame_buffer, m.FrameBuffer)
    assert s._stopped is False
    assert s._render_task is None


def test_streamer_rejects_fps_zero(m):
    try:
        m.WebRTCStreamer(camera=DummyCamera(), host="127.0.0.1", port=0, fps=0)
        raise AssertionError("should raise ValueError")
    except ValueError:
        pass


def test_streamer_rejects_fractional_fps(m):
    try:
        m.WebRTCStreamer(camera=DummyCamera(), host="127.0.0.1", port=0, fps=0.5)
        raise AssertionError("should raise ValueError")
    except ValueError:
        pass


def test_streamer_routes(m):
    s = m.WebRTCStreamer(camera=DummyCamera(), host="127.0.0.1", port=0, fps=30)
    app = s.create_app()
    routes = set()
    for r in app.router.routes():
        info = r.get_info()
        if "path" in info:
            routes.add((r.method, info["path"]))
    assert ("GET", "/") in routes
    assert ("POST", "/offer") in routes


def test_streamer_shutdown_route(m):
    s = m.WebRTCStreamer(camera=DummyCamera(), host="127.0.0.1", port=0, fps=30, allow_browser_shutdown=True)
    app = s.create_app()
    routes = set()
    for r in app.router.routes():
        info = r.get_info()
        if "path" in info:
            routes.add((r.method, info["path"]))
    assert ("POST", "/shutdown") in routes


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_buffer_empty_on_init,
    test_buffer_produce_consume,
    test_buffer_non_destructive,
    test_buffer_clear,
    test_buffer_overwrite,
    test_track_black_frame_when_empty,
    test_track_buffer_frame,
    test_track_pts_increments,
    test_track_rejects_fps_zero,
    test_track_rejects_fractional_fps,
    test_multi_track_shared_buffer,
    test_streamer_has_frame_buffer,
    test_streamer_rejects_fps_zero,
    test_streamer_rejects_fractional_fps,
    test_streamer_routes,
    test_streamer_shutdown_route,
]

if __name__ == "__main__":
    try:
        import aiortc  # noqa: F401
        import aiohttp  # noqa: F401
        import av  # noqa: F401
    except ImportError:
        print("ERROR: missing webrtc deps (aiortc, aiohttp, av) — cannot verify")
        sys.exit(2)

    m = _load_webrtc_module()
    passed = 0
    failed = 0
    for fn in ALL_TESTS:
        try:
            fn(m)
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  {passed} passed, {failed} failed out of {len(ALL_TESTS)} tests")
    print(f"{'=' * 60}")
    sys.exit(1 if failed else 0)
