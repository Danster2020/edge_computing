import cv2
import time
import csv
import os
import sys
import argparse
import subprocess
import numpy as np
from collections import Counter
from benchmark import Benchmark
from models.yolo_decoder import YoloModel
from models.rf_detr_decoder import RfDetrModel
import psutil

MODEL_NAME = "rf-detr-base-coco" # yolo11n rf-detr-base-coco
MODEL_PATH = f"onnx_models/{MODEL_NAME}.onnx"
BENCHMARK_SECONDS = 10


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
                print("If using uv, try: python3 run_webcam_benchmark.py <experiment_name> --camera rpi")
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
    parser.add_argument("experiment_name")
    parser.add_argument("--camera", choices=["webcam", "rpi", "rpicam", "picamera"], default="webcam")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    experiment_name = args.experiment_name

    # Extract model name from path
    model_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]

    # Create simulations folder if not exists
    os.makedirs("simulations", exist_ok=True)

    # Output filenames
    csv_path = f"simulations/{model_name}_{args.camera}_{experiment_name}.csv"
    summary_path = f"simulations/{model_name}_{args.camera}_{experiment_name}_summary.txt"

    # ==========================
    # Setup
    # ==========================
    model = load_model(MODEL_PATH)
    bench = Benchmark()

    cam = open_camera(args.camera, args.camera_index)

    # Warmup
    for _ in range(20):
        ret, frame = read_frame(args.camera, cam)
        if ret:
            model(frame)

    print(f"\nRunning benchmark for {BENCHMARK_SECONDS} seconds...")
    start_time = time.perf_counter()

    frame_id = 0
    detection_rows = []
    process = psutil.Process()


    # ==========================
    # Benchmark loop
    # ==========================
    while True:
        ret, frame = read_frame(args.camera, cam)
        if not ret:
            break

        latency = bench.measure(model, frame)
        boxes, scores, class_ids = model.get_detections()

        labels = [model.class_names[c] for c in class_ids]
        count = len(labels)
        
        # CPU/memory logging
        cpu_percent = process.cpu_percent(interval=None)
        memory_mb = process.memory_info().rss / (1024 * 1024)

        detection_rows.append([
            frame_id,
            latency * 1000,
            count,
            ",".join(labels),
            cpu_percent,
            memory_mb
        ])

        frame_id += 1

        if time.perf_counter() - start_time >= BENCHMARK_SECONDS:
            break

    close_camera(args.camera, cam)

    # ==========================
    # Results
    # ==========================
    avg_latency = bench.average_latency_ms()
    p95_latency = bench.percentile_latency_ms()
    avg_fps = 1000 / avg_latency if avg_latency > 0 else 0

    # Aggregate detections
    all_labels = []
    for row in detection_rows:
        if row[3]:
            all_labels.extend(row[3].split(","))

    class_counts = Counter(all_labels)

    # ==========================
    # Save CSV
    # ==========================
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Frame",
            "Latency(ms)",
            "NumDetections",
            "DetectedClasses",
            "CPU(%)",
            "Memory(MB)"
        ])
        writer.writerows(detection_rows)

    # ==========================
    # Save Summary TXT
    # ==========================
    with open(summary_path, "w") as f:
        f.write("===== BENCHMARK SUMMARY =====\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Duration: {BENCHMARK_SECONDS} seconds\n")
        f.write(f"Frames processed: {frame_id}\n")
        f.write(f"Average latency: {avg_latency:.2f} ms\n")
        f.write(f"95th percentile latency: {p95_latency:.2f} ms\n")
        f.write(f"Average FPS: {avg_fps:.2f}\n\n")

        f.write("Detected objects summary:\n")
        for cls, count in class_counts.items():
            f.write(f"{cls}: {count}\n")

        if detection_rows:
            avg_cpu = sum(row[4] for row in detection_rows) / len(detection_rows)
            avg_memory = sum(row[5] for row in detection_rows) / len(detection_rows)
        else:
            avg_cpu = 0.0
            avg_memory = 0.0
        f.write(f"Average CPU usage: {avg_cpu:.2f} %\n")
        f.write(f"Average Memory usage: {avg_memory:.2f} MB\n")

    # ==========================
    # Print results
    # ==========================
    print("\n===== BENCHMARK COMPLETE =====")
    print(f"CSV saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")
    if not detection_rows:
        print("No frames were processed. Check camera selection/permissions.")


if __name__ == "__main__":
    main()
