import cv2
import argparse
import sys
import os
from benchmark import Benchmark
from models.yolo_decoder import YoloModel
from models.rf_detr_decoder import RfDetrModel


def load_model(path):
    if "rf-detr" in path.lower() or "rfdetr" in path.lower():
        return RfDetrModel(path)
    return YoloModel(path)


def open_camera(camera_type, camera_index):
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
    if camera_type == "rpi":
        frame_rgb = cam.capture_array()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return True, frame
    return cam.read()


def close_camera(camera_type, cam):
    if camera_type == "rpi":
        cam.stop()
    else:
        cam.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=["webcam", "rpi"], default="webcam")
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

    print("\n===== BENCHMARK RESULTS =====")
    print(f"Average latency: {bench.average_latency_ms():.2f} ms")
    print(f"95th percentile latency: {bench.percentile_latency_ms():.2f} ms")
    print(f"Average FPS: {1000 / bench.average_latency_ms():.2f}")
    bench.save_csv(f"{model_name}_{args.camera}_results.csv")


if __name__ == "__main__":
    main()
