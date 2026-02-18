import onnxruntime as ort
import numpy as np
import cv2

class BaseDetector:
    def detect(self, frame):
        raise NotImplementedError



class OnnxDetector(BaseDetector):
    def __init__(self, model_path, input_size=(640, 640)):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size

    def preprocess(self, frame):
        img = cv2.resize(frame, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess(self, outputs):
        # VERY MODEL DEPENDENT
        detections = []
        raw = outputs[0]

        for det in raw[0]:
            conf = det[4]
            if conf > 0.5:
                x1, y1, x2, y2 = det[:4]
                cls_id = int(det[5])

                detections.append({
                    "label": str(cls_id),
                    "confidence": float(conf),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)]
                })

        return detections

    def detect(self, frame):
        inp = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: inp})
        return self.postprocess(outputs)

def main():
    detector = OnnxDetector("yolo11n.onnx")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        for d in detections:
            x1, y1, x2, y2 = map(int, d["bbox"])
            label = f'{d["label"]} {d["confidence"]:.2f}'
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow("ONNX Detection", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

main()
