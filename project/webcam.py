import cv2
from benchmark import Benchmark
from models.yolo_decoder import YoloModel

model = YoloModel("onnx_models/yolo11n.onnx")
bench = Benchmark()

cap = cv2.VideoCapture(0)

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
