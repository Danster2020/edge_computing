import cv2
from benchmark import Benchmark
from models.yolo_decoder import YoloModel
from models.rf_detr_decoder import RfDetrModel


def load_model(path):
    if "rf-detr" in path.lower() or "rfdetr" in path.lower():
        return RfDetrModel(path)
    return YoloModel(path)

MODEL_NAME = "rf-detr-base-coco" # yolo11n rf-detr-base-coco
MODEL_PATH = f"onnx_models/{MODEL_NAME}.onnx"
model = load_model(MODEL_PATH)
bench = Benchmark()

cap = cv2.VideoCapture(0)

# Warmup
for _ in range(20):
    ret, frame = cap.read()
    if ret:
        model(frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    bench.measure(model, frame)
    img = model.draw(frame)

    fps = bench.fps()
    cv2.putText(img, f"FPS: {fps:.1f}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,0,255),2)

    cv2.imshow("Benchmark", img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n===== BENCHMARK RESULTS =====")
print(f"Average latency: {bench.average_latency_ms():.2f} ms")
print(f"95th percentile latency: {bench.percentile_latency_ms():.2f} ms")
print(f"Average FPS: {1000 / bench.average_latency_ms():.2f}")

bench.save_csv("yolo11n_results.csv")
