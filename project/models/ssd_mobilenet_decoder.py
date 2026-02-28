import cv2
import numpy as np
import onnxruntime as ort
import time

from base_model import BaseModel
from coco_labels import COCO_CLASSES


class SsdMobilenetV1Model(BaseModel):
    def __init__(self, path, conf_threshold=0.4, iou_threshold=0.5):
        self.session = ort.InferenceSession(path)
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_shape = input_meta.shape
        self.input_type = input_meta.type
        self.output_names = [out.name for out in self.session.get_outputs()]

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = COCO_CLASSES

        # COCO sparse category ids (1..90 with gaps) -> contiguous 0..79.
        self.coco_cat_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
            59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
        ]
        self.cat_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.coco_cat_ids)}

        self.nhwc = True
        self.input_h = None
        self.input_w = None

        if len(self.input_shape) == 4:
            if self.input_shape[-1] == 3:
                self.nhwc = True
                self.input_h = self._as_dim(self.input_shape[1])
                self.input_w = self._as_dim(self.input_shape[2])
            elif self.input_shape[1] == 3:
                self.nhwc = False
                self.input_h = self._as_dim(self.input_shape[2])
                self.input_w = self._as_dim(self.input_shape[3])

        self.boxes = []
        self.scores = []
        self.class_ids = []

    @staticmethod
    def _as_dim(v):
        return int(v) if isinstance(v, (int, np.integer)) else None

    def preprocess(self, frame):
        self.orig_h, self.orig_w = frame.shape[:2]

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.input_h is not None and self.input_w is not None:
            img = cv2.resize(img, (self.input_w, self.input_h))

        if not self.nhwc:
            img = img.transpose(2, 0, 1)

        # SSD-MobilenetV1-12 expects uint8 input in ONNX Model Zoo export.
        if "uint8" in self.input_type:
            img = img.astype(np.uint8)
        else:
            img = img.astype(np.float32)

        return img[np.newaxis]

    def _map_class_id(self, raw_cls_id):
        cls_id = int(round(float(raw_cls_id)))

        # Prefer sparse COCO mapping first (used by many SSD exports).
        if cls_id in self.cat_to_idx:
            return self.cat_to_idx[cls_id]
        # One-based contiguous indexing.
        if 1 <= cls_id <= len(self.class_names):
            return cls_id - 1
        # Zero-based contiguous indexing.
        if 0 <= cls_id < len(self.class_names):
            return cls_id
        return None

    def _box_to_xyxy(self, box):
        if len(box) != 4:
            return None

        y1, x1, y2, x2 = [float(v) for v in box]

        # detection_boxes are usually normalized yxyx.
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
            x1 *= self.orig_w
            x2 *= self.orig_w
            y1 *= self.orig_h
            y2 *= self.orig_h
        else:
            ref_w = self.input_w if self.input_w is not None else self.orig_w
            ref_h = self.input_h if self.input_h is not None else self.orig_h
            x1 *= self.orig_w / ref_w
            x2 *= self.orig_w / ref_w
            y1 *= self.orig_h / ref_h
            y2 *= self.orig_h / ref_h

        x1 = max(0, min(x1, self.orig_w))
        y1 = max(0, min(y1, self.orig_h))
        x2 = max(0, min(x2, self.orig_w))
        y2 = max(0, min(y2, self.orig_h))

        if x2 <= x1 or y2 <= y1:
            return None

        return [int(x1), int(y1), int(x2), int(y2)]

    @staticmethod
    def _flatten_batch(arr):
        if arr.ndim >= 2 and arr.shape[0] == 1:
            return arr[0]
        return arr

    def _extract_outputs(self, outputs):
        boxes_arr = None
        classes_arr = None
        scores_arr = None
        num_dets = None

        # Primary path: parse by output names.
        for name, out in zip(self.output_names, outputs):
            lname = name.lower()
            arr = np.array(out)
            if "box" in lname:
                boxes_arr = self._flatten_batch(arr)
            elif "class" in lname:
                classes_arr = self._flatten_batch(arr).reshape(-1)
            elif "score" in lname:
                scores_arr = self._flatten_batch(arr).reshape(-1)
            elif "num" in lname:
                num_dets = int(round(float(arr.reshape(-1)[0])))

        if boxes_arr is not None and classes_arr is not None and scores_arr is not None:
            return boxes_arr, classes_arr, scores_arr, num_dets

        # Fallback path: infer tensors by shape/content.
        for out in outputs:
            arr = np.array(out)
            if arr.ndim == 3 and arr.shape[-1] == 4:
                boxes_arr = arr[0]
            elif arr.ndim == 2 and arr.shape[-1] == 4:
                boxes_arr = arr
            elif arr.ndim in (1, 2):
                flat = arr.reshape(-1)
                if flat.size == 1:
                    num_dets = int(round(float(flat[0])))
                elif np.all((flat >= 0.0) & (flat <= 1.0)):
                    scores_arr = flat
                elif np.all(np.abs(flat - np.round(flat)) < 1e-4):
                    classes_arr = flat

        return boxes_arr, classes_arr, scores_arr, num_dets

    def decode(self, outputs):
        self.boxes, self.scores, self.class_ids = [], [], []

        boxes_arr, classes_arr, scores_arr, num_dets = self._extract_outputs(outputs)
        if boxes_arr is None or classes_arr is None or scores_arr is None:
            return

        count = min(len(boxes_arr), len(classes_arr), len(scores_arr))
        if num_dets is not None:
            count = min(count, num_dets)

        boxes, scores, class_ids = [], [], []
        for i in range(count):
            score = float(scores_arr[i])
            if score < self.conf_threshold:
                continue

            mapped_cls = self._map_class_id(classes_arr[i])
            if mapped_cls is None:
                continue

            xyxy = self._box_to_xyxy(boxes_arr[i])
            if xyxy is None:
                continue

            boxes.append(xyxy)
            scores.append(score)
            class_ids.append(mapped_cls)

        self._apply_nms(boxes, scores, class_ids)

    def _apply_nms(self, boxes, scores, class_ids):
        if not boxes:
            self.boxes, self.scores, self.class_ids = [], [], []
            return

        idx = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
        if len(idx) == 0:
            self.boxes, self.scores, self.class_ids = [], [], []
            return

        idx = idx.flatten()
        self.boxes = [boxes[i] for i in idx]
        self.scores = [float(scores[i]) for i in idx]
        self.class_ids = [int(class_ids[i]) for i in idx]

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
            label = (
                self.class_names[cls_id]
                if 0 <= cls_id < len(self.class_names)
                else str(cls_id)
            )
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
