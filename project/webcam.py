import cv2
from benchmark import Benchmark
from camera_source import open_best_camera
from model_selection import choose_model
from models.efficientdet_decoder import EfficientDetD0Model
from models.yolo_decoder import YoloModel
from models.rf_detr_decoder import RfDetrModel
from models.ssd_mobilenet_decoder import SsdMobilenetV1Model


def load_model(path):
    if "efficientdet" in path.lower():
        return EfficientDetD0Model(path)
    if "rf-detr" in path.lower() or "rfdetr" in path.lower():
        return RfDetrModel(path)
    if "ssd_mobilenet" in path.lower() or "ssd-mobilenet" in path.lower() or "ssdmobilenet" in path.lower():
        return SsdMobilenetV1Model(path)
    return YoloModel(path)


def main():
    try:
        model_name, model_path = choose_model()
    except RuntimeError as e:
        print(str(e))
        return

    print(f"Selected model: {model_name}")

    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {model_path}")
        print(f"Reason: {e}")
        return

    bench = Benchmark()

    try:
        cam = open_best_camera()
    except RuntimeError as e:
        print(str(e))
        return

    print(f"Using camera source: {cam.kind}")

    # Warmup
    for _ in range(20):
        ret, frame = cam.read()
        if ret:
            model(frame)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        bench.measure(model, frame)
        img = model.draw(frame)

        fps = bench.fps()
        cv2.putText(
            img,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

        cv2.imshow("Benchmark", img)

        if cv2.waitKey(1) == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
