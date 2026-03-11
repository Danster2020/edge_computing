import cv2
import numpy as np
import onnxruntime as ort
import time

from base_model import BaseModel
from coco_labels import COCO_CLASSES
from coco_labels_dict import COCO_CLASSES2



class RfDetrModel(BaseModel):
    def __init__(self, path, conf_threshold=0.4):
        self.session = ort.InferenceSession(path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_h = int(self.input_shape[2])
        self.input_w = int(self.input_shape[3])
        self.conf_threshold = conf_threshold

        self.class_names = COCO_CLASSES2

        self.boxes = []
        self.scores = []
        self.class_ids = []

    def preprocess(self, frame):
        self.orig_h, self.orig_w = frame.shape[:2]

        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        return img[np.newaxis]

    def _scale_and_clamp_xyxy(self, box):
        x1, y1, x2, y2 = box
        x1 = x1 * self.orig_w
        x2 = x2 * self.orig_w
        y1 = y1 * self.orig_h
        y2 = y2 * self.orig_h

        x1 = max(0, min(float(x1), self.orig_w))
        y1 = max(0, min(float(y1), self.orig_h))
        x2 = max(0, min(float(x2), self.orig_w))
        y2 = max(0, min(float(y2), self.orig_h))

        if x2 <= x1 or y2 <= y1:
            return None

        return [int(x1), int(y1), int(x2), int(y2)]

    @staticmethod
    def _cxcywh_to_xyxy(box):
        cx, cy, w, h = box
        return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]

    def decode(self, outputs):
        self.boxes, self.scores, self.class_ids = [], [], []

        # Typical RF-DETR ONNX export: (pred_boxes, pred_logits)
        # pred_boxes: (1, num_queries, 4), pred_logits: (1, num_queries, num_classes)
        boxes_arr = None
        logits_arr = None
        scores_arr = None
        labels_arr = None

        for out in outputs:
            arr = np.array(out)
            if arr.ndim == 3 and arr.shape[-1] == 4:
                boxes_arr = arr[0]
            elif arr.ndim == 3 and arr.shape[-1] > 4:
                logits_arr = arr[0]
            elif arr.ndim == 2 and arr.shape[-1] == 4:
                boxes_arr = arr
            elif arr.ndim == 2 and arr.shape[-1] > 4:
                logits_arr = arr
            elif arr.ndim in (1, 2):
                flat = arr.reshape(-1)
                if np.issubdtype(flat.dtype, np.integer):
                    labels_arr = flat
                else:
                    scores_arr = flat

        if boxes_arr is None:
            return

        if logits_arr is not None:
            probs = 1.0 / (1.0 + np.exp(-logits_arr))
            class_ids = np.argmax(probs, axis=1)
            scores = probs[np.arange(probs.shape[0]), class_ids]
        else:
            if scores_arr is None or labels_arr is None:
                return
            count = min(len(boxes_arr), len(scores_arr), len(labels_arr))
            boxes_arr = boxes_arr[:count]
            scores = scores_arr[:count]
            class_ids = labels_arr[:count]

        for box, score, cls_id in zip(boxes_arr, scores, class_ids):
            score = float(score)
            cls_id = int(cls_id)
            if score < self.conf_threshold:
                continue

            if np.max(box) > 1.5:
                # Some exports already return pixel-space xyxy against model input size.
                x1, y1, x2, y2 = box
                x1 *= self.orig_w / self.input_w
                x2 *= self.orig_w / self.input_w
                y1 *= self.orig_h / self.input_h
                y2 *= self.orig_h / self.input_h
                scaled = [x1 / self.orig_w, y1 / self.orig_h, x2 / self.orig_w, y2 / self.orig_h]
            else:
                # RF-DETR can export boxes as normalized cxcywh or normalized xyxy.
                if box[0] < box[2] and box[1] < box[3]:
                    scaled = box
                else:
                    scaled = self._cxcywh_to_xyxy(box)

            xyxy = self._scale_and_clamp_xyxy(scaled)
            if xyxy is None:
                continue

            self.boxes.append(xyxy)
            self.scores.append(score)
            self.class_ids.append(cls_id)

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
            "total": (t3 - t0) * 1000,
        }

    def draw(self, frame):
        img = frame.copy()
        for box, score, cls_id in zip(self.boxes, self.scores, self.class_ids):
            x1, y1, x2, y2 = box
            label = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            text = f"{label} {score:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(
                img,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return img

    def get_detections(self):
        return self.boxes, self.scores, self.class_ids
