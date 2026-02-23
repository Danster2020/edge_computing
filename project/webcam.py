import cv2
import argparse
import sys
import os
import subprocess
import numpy as np
from benchmark import Benchmark
from models.yolo_decoder import YoloModel
from models.rf_detr_decoder import RfDetrModel


def load_model(path):
    if "rf-detr" in path.lower() or "rfdetr" in path.lower():
        return RfDetrModel(path)
    return YoloModel(path)


class RpiCamVidCapture:
    def __init__(self, width=1280, height=720, framerate=30):
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 3 // 2  # YUV420 (I420)
        cmd = [
            "rpicam-vid",
            "--nopreview",
            "-t",
            "0",
            "--codec",
            "yuv420",
            "--width",
            str(width),
            "--height",
            str(height),
            "--framerate",
            str(framerate),
            "-o",
            "-",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def read(self):
        if self.proc.stdout is None:
            return False, None

        buf = self.proc.stdout.read(self.frame_bytes)
        if not buf or len(buf) < self.frame_bytes:
            return False, None

        yuv = np.frombuffer(buf, dtype=np.uint8).reshape((self.height * 3 // 2, self.width))
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        return True, frame

    def release(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def open_camera(camera_type, camera_index):
    if camera_type == "picamera":
        try:
            from picamera import PiCamera
            from picamera.array import PiRGBArray
        except ImportError:
            print("picamera module is not available.")
            print("Install (legacy): sudo apt install -y python3-picamera")
            print("On Raspberry Pi OS Bookworm/Pi 5, prefer --camera rpi (picamera2).")
            sys.exit(1)

        class PiCameraCapture:
            def __init__(self, width=1280, height=720, framerate=30):
                self.camera = PiCamera()
                self.camera.resolution = (width, height)
                self.camera.framerate = framerate
                self.raw_capture = PiRGBArray(self.camera, size=(width, height))
                self.stream = self.camera.capture_continuous(
                    self.raw_capture,
                    format="bgr",
                    use_video_port=True,
                )

            def read(self):
                frame_data = next(self.stream, None)
                if frame_data is None:
                    return False, None
                frame = frame_data.array
                self.raw_capture.truncate(0)
                return True, frame

            def release(self):
                self.stream.close()
                self.raw_capture.close()
                self.camera.close()

        return PiCameraCapture()

    if camera_type == "rpicam":
        try:
            return RpiCamVidCapture()
        except FileNotFoundError:
            print("rpicam-vid not found. Install Raspberry Pi camera apps.")
            sys.exit(1)

    if camera_type == "rpi":
        try:
            from picamera2 import Picamera2
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
            except ImportError:
                print("picamera2 is not available in this Python environment.")
                print("If using uv, try: python3 webcam.py --camera rpi")
                print("Or install in uv env: uv add picamera2")
                print("System install command: sudo apt install -y python3-picamera2")
                sys.exit(1)

        cam = Picamera2()
        config = cam.create_video_configuration(
            main={"size": (1280, 720), "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        return cam

    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        print(f"Failed to open webcam index {camera_index}")
        sys.exit(1)
    return cam


def read_frame(camera_type, cam):
    if camera_type == "picamera":
        return cam.read()
    if camera_type == "rpi":
        frame_rgb = cam.capture_array()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return True, frame
    if camera_type == "rpicam":
        return cam.read()
    return cam.read()


def close_camera(camera_type, cam):
    if camera_type == "picamera":
        cam.release()
    elif camera_type == "rpi":
        cam.stop()
    elif camera_type == "rpicam":
        cam.release()
    else:
        cam.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=["webcam", "rpi", "rpicam", "picamera"], default="webcam")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    model_name = "rf-detr-base-coco"  # yolo11n rf-detr-base-coco
    model_path = f"onnx_models/{model_name}.onnx"

    model = load_model(model_path)
    bench = Benchmark()
    cam = open_camera(args.camera, args.camera_index)

    for _ in range(20):
        ret, frame = read_frame(args.camera, cam)
        if ret:
            model(frame)

    while True:
        ret, frame = read_frame(args.camera, cam)
        if not ret:
            break

        bench.measure(model, frame)
        img = model.draw(frame)

        fps = bench.fps()
        cv2.putText(img, f"FPS: {fps:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

        cv2.imshow("Benchmark", img)

        if cv2.waitKey(1) == ord('q'):
            break

    close_camera(args.camera, cam)
    cv2.destroyAllWindows()

    avg_latency = bench.average_latency_ms()
    p95_latency = bench.percentile_latency_ms()
    avg_fps = (1000 / avg_latency) if avg_latency > 0 else 0.0

    print("\n===== BENCHMARK RESULTS =====")
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"95th percentile latency: {p95_latency:.2f} ms")
    print(f"Average FPS: {avg_fps:.2f}")
    if not bench.times:
        print("No frames were processed. Check camera selection/permissions.")
    bench.save_csv(f"{model_name}_{args.camera}_results.csv")


if __name__ == "__main__":
    main()
