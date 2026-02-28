import cv2
import numpy as np
import onnxruntime as ort
import time

from base_model import BaseModel
from coco_labels import COCO_CLASSES


class YoloModel(BaseModel):
    def __init__(self, path, conf_threshold=0.4, iou_threshold=0.5):
        self.session = ort.InferenceSession(path)
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_shape = input_meta.shape
        self.input_h = int(self.input_shape[2])
        self.input_w = int(self.input_shape[3])

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
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

    def _scale_and_clip_xyxy(self, x1, y1, x2, y2):
        max_abs = max(abs(float(x1)), abs(float(y1)), abs(float(x2)), abs(float(y2)))

        # Handle normalized outputs and outputs in model-input pixels.
        if max_abs <= 1.5:
            x1 *= self.orig_w
            x2 *= self.orig_w
            y1 *= self.orig_h
            y2 *= self.orig_h
        else:
            x1 *= self.orig_w / self.input_w
            x2 *= self.orig_w / self.input_w
            y1 *= self.orig_h / self.input_h
            y2 *= self.orig_h / self.input_h

        x1 = max(0, min(float(x1), self.orig_w))
        y1 = max(0, min(float(y1), self.orig_h))
        x2 = max(0, min(float(x2), self.orig_w))
        y2 = max(0, min(float(y2), self.orig_h))

        if x2 <= x1 or y2 <= y1:
            return None

        return [int(x1), int(y1), int(x2), int(y2)]

    def _collect_no_objectness(self, preds):
        boxes, scores, class_ids = [], [], []
        for p in preds:
            if p.shape[0] < 5:
                continue

            cx, cy, w, h = p[:4]
            class_scores = p[4:]
            if class_scores.size == 0:
                continue

            cls_id = int(np.argmax(class_scores))
            score = float(class_scores[cls_id])
            if score < self.conf_threshold:
                continue

            xyxy = self._scale_and_clip_xyxy(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            if xyxy is None:
                continue

            boxes.append(xyxy)
            scores.append(score)
            class_ids.append(cls_id)
        return boxes, scores, class_ids

    def _collect_with_objectness(self, preds):
        boxes, scores, class_ids = [], [], []
        for p in preds:
            if p.shape[0] < 6:
                continue

            cx, cy, w, h = p[:4]
            obj = float(p[4])
            if obj <= 0.0:
                continue

            class_scores = p[5:]
            if class_scores.size == 0:
                continue

            cls_id = int(np.argmax(class_scores))
            score = obj * float(class_scores[cls_id])
            if score < self.conf_threshold:
                continue

            xyxy = self._scale_and_clip_xyxy(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            if xyxy is None:
                continue

            boxes.append(xyxy)
            scores.append(score)
            class_ids.append(cls_id)
        return boxes, scores, class_ids

    def _collect_end2end_xyxy(self, preds):
        boxes, scores, class_ids = [], [], []
        for p in preds:
            if p.shape[0] < 6:
                continue

            x1, y1, x2, y2, score, cls_id = p[:6]
            score = float(score)
            if score < self.conf_threshold:
                continue

            cls_id = int(round(float(cls_id)))
            if cls_id < 0:
                continue

            # Fallback for exporters that might still emit xywh in 6-col format.
            if float(x2) <= float(x1) or float(y2) <= float(y1):
                cx, cy, w, h = float(x1), float(y1), float(x2), float(y2)
                x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

            xyxy = self._scale_and_clip_xyxy(x1, y1, x2, y2)
            if xyxy is None:
                continue

            boxes.append(xyxy)
            scores.append(score)
            class_ids.append(cls_id)
        return boxes, scores, class_ids

    def _apply_nms(self, boxes, scores, class_ids):
        if not boxes:
            self.boxes, self.scores, self.class_ids = [], [], []
            return

        nms_boxes = []
        for x1, y1, x2, y2 in boxes:
            nms_boxes.append([x1, y1, max(1, x2 - x1), max(1, y2 - y1)])

        idx = cv2.dnn.NMSBoxes(nms_boxes, scores, self.conf_threshold, self.iou_threshold)
        if idx is None or len(idx) == 0:
            self.boxes, self.scores, self.class_ids = [], [], []
            return

        keep = np.array(idx).reshape(-1)
        self.boxes = [boxes[i] for i in keep]
        self.scores = [float(scores[i]) for i in keep]
        self.class_ids = [int(class_ids[i]) for i in keep]

    def decode(self, outputs):
        self.boxes, self.scores, self.class_ids = [], [], []
        if not outputs:
            return

        preds = np.squeeze(np.array(outputs[0]))
        if preds.ndim != 2:
            return

        # Normalize to rows=predictions, cols=attributes.
        if preds.shape[0] < preds.shape[1] and preds.shape[0] <= len(self.class_names) + 5:
            preds = preds.T

        cols = preds.shape[1]
        num_classes = len(self.class_names)

        if cols == 6:
            boxes, scores, class_ids = self._collect_end2end_xyxy(preds)
        elif cols == num_classes + 4:
            boxes, scores, class_ids = self._collect_no_objectness(preds)
        elif cols == num_classes + 5:
            boxes, scores, class_ids = self._collect_with_objectness(preds)
        else:
            # Fallback for custom class counts: try both and keep denser parse.
            with_obj = self._collect_with_objectness(preds)
            no_obj = self._collect_no_objectness(preds)
            boxes, scores, class_ids = with_obj if len(with_obj[0]) >= len(no_obj[0]) else no_obj

        self._apply_nms(boxes, scores, class_ids)

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
