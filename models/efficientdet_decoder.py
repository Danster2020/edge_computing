import cv2
import numpy as np
import onnxruntime as ort
import time

from base_model import BaseModel
from coco_labels import COCO_CLASSES


class EfficientDetD0Model(BaseModel):
    def __init__(self, path, conf_threshold=0.35, iou_threshold=0.5):
        self.session = ort.InferenceSession(path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        # Supports both NCHW and NHWC exports.
        if len(self.input_shape) == 4 and self.input_shape[1] == 3:
            self.input_h = int(self.input_shape[2])
            self.input_w = int(self.input_shape[3])
            self.nchw = True
        else:
            self.input_h = int(self.input_shape[1])
            self.input_w = int(self.input_shape[2])
            self.nchw = False

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = COCO_CLASSES

        # COCO 90-category ids mapped to contiguous 0..79 indices.
        self.coco_cat_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
            59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
        ]
        self.cat_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.coco_cat_ids)}

        self.boxes = []
        self.scores = []
        self.class_ids = []

    def preprocess(self, frame):
        self.orig_h, self.orig_w = frame.shape[:2]

        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.nchw:
            img = img.transpose(2, 0, 1)

        return img[np.newaxis]

    def _map_class_id(self, raw_cls_id):
        cls_id = int(raw_cls_id)

        # Contiguous COCO indexing.
        if 0 <= cls_id < len(self.class_names):
            return cls_id
        # One-based contiguous indexing.
        if 1 <= cls_id <= len(self.class_names):
            return cls_id - 1
        # Sparse COCO category id indexing.
        if cls_id in self.cat_to_idx:
            return self.cat_to_idx[cls_id]
        return None

    def _box_to_xyxy(self, box, assume_yxyx=True):
        if assume_yxyx:
            y1, x1, y2, x2 = box
        else:
            x1, y1, x2, y2 = box

        # Normalize/scaling to original frame.
        if max(abs(float(x1)), abs(float(y1)), abs(float(x2)), abs(float(y2))) <= 1.5:
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

    def _collect_from_boxes_scores_classes(self, boxes_arr, scores_arr, classes_arr):
        boxes, scores, class_ids = [], [], []

        count = min(len(boxes_arr), len(scores_arr), len(classes_arr))
        for i in range(count):
            score = float(scores_arr[i])
            if score < self.conf_threshold:
                continue

            mapped_cls = self._map_class_id(classes_arr[i])
            if mapped_cls is None:
                continue

            xyxy = self._box_to_xyxy(boxes_arr[i], assume_yxyx=True)
            if xyxy is None:
                xyxy = self._box_to_xyxy(boxes_arr[i], assume_yxyx=False)
            if xyxy is None:
                continue

            boxes.append(xyxy)
            scores.append(score)
            class_ids.append(mapped_cls)

        return boxes, scores, class_ids

    def decode(self, outputs):
        self.boxes, self.scores, self.class_ids = [], [], []

        arrays = [np.array(o) for o in outputs]

        # Common EfficientDet postprocessed format: (1,N,7) with:
        # [image_id, y1, x1, y2, x2, score, class_id].
        for arr in arrays:
            if arr.ndim == 3 and arr.shape[-1] == 7:
                detections = arr[0]
                boxes, scores, class_ids = [], [], []
                for row in detections:
                    score = float(row[5])
                    if score < self.conf_threshold:
                        continue
                    mapped_cls = self._map_class_id(row[6])
                    if mapped_cls is None:
                        continue
                    xyxy = self._box_to_xyxy(row[1:5], assume_yxyx=True)
                    if xyxy is None:
                        continue
                    boxes.append(xyxy)
                    scores.append(score)
                    class_ids.append(mapped_cls)
                self._apply_nms(boxes, scores, class_ids)
                return

        # Tensorflow style postprocessed outputs: boxes/scores/classes/num.
        boxes_arr = None
        scores_arr = None
        classes_arr = None
        num_dets = None
        logits_arr = None

        for arr in arrays:
            if arr.ndim == 3 and arr.shape[-1] == 4:
                boxes_arr = arr[0]
            elif arr.ndim == 3 and arr.shape[-1] > 4:
                logits_arr = arr[0]
            elif arr.ndim == 2 and arr.shape[-1] == 4:
                boxes_arr = arr
            elif arr.ndim in (2, 1):
                flat = arr.reshape(-1)
                if flat.size == 1:
                    num_dets = int(flat[0])
                elif np.issubdtype(flat.dtype, np.integer):
                    classes_arr = flat
                else:
                    # Heuristic: scores are usually in [0, 1].
                    if np.all((flat >= 0.0) & (flat <= 1.0)):
                        scores_arr = flat
                    else:
                        logits_arr = arr if arr.ndim == 2 else None

        # If model gives class logits per box.
        if boxes_arr is not None and logits_arr is not None:
            probs = 1.0 / (1.0 + np.exp(-logits_arr))
            raw_cls = np.argmax(probs, axis=1)
            scores = probs[np.arange(probs.shape[0]), raw_cls]
            boxes, scores, class_ids = self._collect_from_boxes_scores_classes(
                boxes_arr, scores, raw_cls
            )
            self._apply_nms(boxes, scores, class_ids)
            return

        # If model gives boxes + scores + classes.
        if boxes_arr is not None and scores_arr is not None and classes_arr is not None:
            if num_dets is not None:
                boxes_arr = boxes_arr[:num_dets]
                scores_arr = scores_arr[:num_dets]
                classes_arr = classes_arr[:num_dets]
            boxes, scores, class_ids = self._collect_from_boxes_scores_classes(
                boxes_arr, scores_arr, classes_arr
            )
            self._apply_nms(boxes, scores, class_ids)
            return

    def _apply_nms(self, boxes, scores, class_ids):
        if not boxes:
            self.boxes, self.scores, self.class_ids = [], [], []
            return
        idx = cv2.dnn.NMSBoxes(
            boxes, scores, self.conf_threshold, self.iou_threshold
        )
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
