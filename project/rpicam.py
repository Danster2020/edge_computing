import cv2
import time
import os
import sys


def import_picamera2():
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
        from picamera2 import Picamera2

        return Picamera2


Picamera2 = import_picamera2()
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(raw={"size":(1640, 1232)}, main={"format":"RGB888", "size": (640, 480)}))
picam2.start()
time.sleep(2)

while True:
    img = picam2.capture_array()
    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
picam2.close()
