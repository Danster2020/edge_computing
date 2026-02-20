import cv2
import numpy as np
import onnxruntime as ort
from base_model import BaseModel
from coco_labels import COCO_CLASSES
import time


class YoloModel(BaseModel):
    def __init__(self, path):
        self.session = ort.InferenceSession(path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]
        
        self.class_names = COCO_CLASSES

        self.boxes = []
        self.scores = []
        self.class_ids = []

    def preprocess(self, frame):
        self.orig_h, self.orig_w = frame.shape[:2]

        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = img.transpose(2, 0, 1)

        return img[np.newaxis].astype(np.float32)

    def decode(self, outputs):
        # (1,84,8400) → (8400,84)
        preds = np.squeeze(outputs[0]).T

        boxes, scores, class_ids = [], [], []

        scale_x = self.orig_w / self.input_w
        scale_y = self.orig_h / self.input_h

        for p in preds:
            cx, cy, w, h = p[:4]
            class_scores = p[4:]

            cls_id = np.argmax(class_scores)
            score = class_scores[cls_id]

            if score < 0.4:
                continue

            # center → corners
            x1 = (cx - w/2) * scale_x
            y1 = (cy - h/2) * scale_y
            x2 = (cx + w/2) * scale_x
            y2 = (cy + h/2) * scale_y

            # clamp to image bounds
            x1 = max(0, min(x1, self.orig_w))
            y1 = max(0, min(y1, self.orig_h))
            x2 = max(0, min(x2, self.orig_w))
            y2 = max(0, min(y2, self.orig_h))

            # skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            scores.append(float(score))
            class_ids.append(int(cls_id))

        # Non-Max Suppression
        if boxes:
            idx = cv2.dnn.NMSBoxes(boxes, scores, 0.4, 0.5)
            if len(idx) > 0:
                idx = idx.flatten()
                self.boxes = [boxes[i] for i in idx]
                self.scores = [scores[i] for i in idx]
                self.class_ids = [class_ids[i] for i in idx]
            else:
                self.boxes, self.scores, self.class_ids = [], [], []
        else:
            self.boxes, self.scores, self.class_ids = [], [], []

    def __call__(self, frame):
        t0 = time.perf_counter()

        inp = self.preprocess(frame)
        t1 = time.perf_counter()

        outputs = self.session.run(None, {self.input_name: inp})
        t2 = time.perf_counter()

        self.decode(outputs)
        t3 = time.perf_counter()

        return {
            "preprocess": (t1 - t0) * 1000,
            "inference": (t2 - t1) * 1000,
            "postprocess": (t3 - t2) * 1000,
            "total": (t3 - t0) * 1000
        }


    def draw(self, frame):
        img = frame.copy()

        for box, score, cls_id in zip(self.boxes, self.scores, self.class_ids):
            x1, y1, x2, y2 = box
            label = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)

            text = f"{label} {score:.2f}"

            # Bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

            # Label background
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), (0,255,0), -1)

            # Label text
            cv2.putText(img, text,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,0,0), 2)

        return img

    def get_detections(self):
        return self.boxes, self.scores, self.class_ids
