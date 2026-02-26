import os
import sys
import time

import cv2


def _import_picamera2():
    try:
        from picamera2 import Picamera2

        return Picamera2
    except ImportError:
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        fallback_paths = [
            "/usr/lib/python3/dist-packages",
            f"/usr/lib/python{py_ver}/dist-packages",
        ]
        for p in fallback_paths:
            if os.path.isdir(p) and p not in sys.path:
                sys.path.append(p)
        try:
            from picamera2 import Picamera2

            return Picamera2
        except ImportError:
            return None


class AutoCamera:
    def __init__(self, kind, handle):
        self.kind = kind
        self.handle = handle

    def read(self):
        if self.kind == "rpi":
            frame_rgb = self.handle.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return True, frame_bgr
        return self.handle.read()

    def release(self):
        if self.kind == "rpi":
            self.handle.stop()
            self.handle.close()
        else:
            self.handle.release()


def open_best_camera(camera_index=0, rpi_size=(1280, 720), rpi_warmup_s=0.2):
    picam2_cls = _import_picamera2()
    if picam2_cls is not None:
        try:
            cam = picam2_cls()
            config = cam.create_video_configuration(
                main={"size": rpi_size, "format": "RGB888"}
            )
            cam.configure(config)
            cam.start()
            time.sleep(rpi_warmup_s)
            return AutoCamera("rpi", cam)
        except Exception:
            pass

    cam = cv2.VideoCapture(camera_index)
    if cam.isOpened():
        return AutoCamera("webcam", cam)
    cam.release()
    raise RuntimeError(
        "No camera detected. Ensure Raspberry Pi camera is enabled or webcam is connected."
    )
