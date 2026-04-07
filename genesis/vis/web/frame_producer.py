import threading

import genesis as gs


# Try fast JPEG encoder first, fall back to OpenCV
try:
    from turbojpeg import TurboJPEG

    _turbojpeg = TurboJPEG()
    _USE_TURBOJPEG = True
except (ImportError, RuntimeError, OSError):
    _turbojpeg = None
    _USE_TURBOJPEG = False


def _encode_jpeg(rgb_arr, quality):
    """Encode a uint8 RGB array to JPEG bytes."""
    if _USE_TURBOJPEG:
        return _turbojpeg.encode(rgb_arr, quality=quality)
    else:
        import cv2

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        # cv2.imencode expects BGR input
        bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, encode_param)
        if not ok:
            return None
        return buf.tobytes()


class FrameProducer:
    """Captures and encodes frames on the main thread.

    Uses a latest-frame-only buffer (not a queue) for backpressure.
    If the web server hasn't consumed the previous frame when a new
    one is produced, the old frame is overwritten.
    """

    def __init__(self, scene, resolution=(1280, 720), jpeg_quality=80):
        self._scene = scene
        self._resolution = resolution
        self._jpeg_quality = jpeg_quality

        self._lock = threading.Lock()
        self._latest_frame: bytes | None = None

    def produce_frame(self):
        """Called from main thread after scene.step().

        Captures a frame from the rasterizer and JPEG-encodes it.
        Must be called on the same thread that created the OpenGL context.
        """
        visualizer = self._scene.visualizer
        rasterizer = visualizer._rasterizer
        camera = visualizer.cameras[0]

        # Ensure the pyrender scene context has the latest visual state
        if hasattr(rasterizer, "update_scene"):
            rasterizer.update_scene(force_render=False)

        # render_camera handles make_current/make_uncurrent internally
        # when the rasterizer is in offscreen mode
        rgb_arr, _depth, _seg, _normal = rasterizer.render_camera(camera, rgb=True)

        if rgb_arr is None:
            return

        jpeg_bytes = _encode_jpeg(rgb_arr, self._jpeg_quality)
        if jpeg_bytes is None:
            return

        with self._lock:
            self._latest_frame = jpeg_bytes

    def consume_frame(self) -> bytes | None:
        """Called from web server thread. Returns latest JPEG bytes or None.

        After consumption the buffer is cleared so the same frame is not
        sent twice.
        """
        with self._lock:
            frame = self._latest_frame
            self._latest_frame = None
        return frame
