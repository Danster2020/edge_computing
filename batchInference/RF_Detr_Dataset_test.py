import os
import time
import cv2
import numpy as np
import psutil
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import onnxruntime as ort
from tqdm import tqdm

# ==============================
# Configs
# ==============================
IMAGE_FOLDER = "val2017"
ANNOTATION_FILE = "annotations/instances_val2017.json"
MODEL_PATH = "../onnx_models/rf-detr-base-coco.onnx"
CONF_THRESH = 0.3
IOU_THRESH = 0.5
MAX_IMAGES = 0
# ==============================

# ==============================
# Load COCO
# ==============================
coco = COCO(ANNOTATION_FILE)
coco_cat_ids = coco.getCatIds()  # list of 90 valid COCO IDs
image_ids = list(coco.imgs.keys())
eval_image_ids = image_ids[:MAX_IMAGES] if MAX_IMAGES > 0 else image_ids
process = psutil.Process(os.getpid())

# ==============================
# Load ONNX model
# ==============================
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]
input_shape = session.get_inputs()[0].shape
INPUT_H, INPUT_W = input_shape[2], input_shape[3]

# ==============================
# IoU computation
# ==============================
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# ==============================
# NumPy softmax
# ==============================
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# ==============================
# Evaluation
# ==============================
def evaluate_rf_detr_onnx():
    tp_all, fp_all, fn_all = 0, 0, 0
    fps_list, inference_times = [], []
    cpu_usage_list, mem_usage_list = [], []
    iou_list = []
    coco_results = []

    for img_id in tqdm(eval_image_ids, desc="Evaluating RF-DETR ONNX"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(IMAGE_FOLDER, img_info['file_name'])
        image = cv2.imread(img_path)
        orig_h, orig_w, _ = image.shape

        # Ground truth boxes
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt_boxes = [[ann['bbox'][0], ann['bbox'][1],
                     ann['bbox'][0]+ann['bbox'][2],
                     ann['bbox'][1]+ann['bbox'][3]] for ann in anns]

        # Preprocess
        img_resized = cv2.resize(image, (INPUT_W, INPUT_H))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_np = img_rgb.astype(np.float32) / 255.0
        img_np = np.transpose(img_np, (2,0,1))[np.newaxis, ...]

        # CPU/memory before
        cpu_before = psutil.cpu_percent(interval=None) / 100
        mem_before = process.memory_info().rss / (1024**2)

        # Inference
        start = time.time()
        outputs = session.run(output_names, {input_name: img_np})
        end = time.time()
        inf_time = end - start
        inference_times.append(inf_time)

        cpu_after = psutil.cpu_percent(interval=None) / 100
        mem_after = process.memory_info().rss / (1024**2)
        cpu_usage_list.append(max(cpu_before, cpu_after)*100)
        mem_usage_list.append(max(mem_before, mem_after))
        fps_list.append(1/inf_time)

        # Postprocess
        pred_boxes  = outputs[0][0]  # [N,4] normalized cx,cy,w,h
        pred_logits = outputs[1][0]  # [N,91]

        # Convert cx,cy,w,h -> x1,y1,x2,y2
        cx, cy, w, h = pred_boxes.T
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Scale to original image
        boxes[:, [0,2]] *= orig_w
        boxes[:, [1,3]] *= orig_h

        # Softmax and labels
        probs = softmax(pred_logits[:, :-1], axis=-1)  # exclude background
        scores = np.max(probs, axis=1)
        labels = np.argmax(probs, axis=1)

        preds = []
        for box, score, cls in zip(boxes, scores, labels):
            if score < CONF_THRESH:
                continue
            if cls >= len(coco_cat_ids):
                continue  # skip invalid class
            category_id = coco_cat_ids[int(cls)]
            x1, y1, x2, y2 = box.astype(int)
            preds.append([x1, y1, x2, y2])

            # Save for COCO evaluation
            coco_results.append({
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [x1, y1, x2-x1, y2-y1],
                "score": float(score)
            })

        # IoU matching for manual metrics
        matched_gt = set()
        for pred_box in preds:
            best_iou, best_idx = 0, -1
            for idx, gt_box in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= IOU_THRESH:
                tp_all += 1
                matched_gt.add(best_idx)
                iou_list.append(best_iou)
            else:
                fp_all += 1
        fn_all += len(gt_boxes) - len(matched_gt)

    # Metrics
    recall = tp_all / (tp_all + fn_all + 1e-6)
    precision = tp_all / (tp_all + fp_all + 1e-6)
    f1 = 2*(precision*recall)/(precision+recall+1e-6)
    avg_iou = np.mean(iou_list) if iou_list else 0.0

    # COCO mAP
    coco_dt = coco.loadRes(coco_results)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.params.imgIds = eval_image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    map50_95 = coco_eval.stats[0]

    print("=== RF-DETR ONNX Evaluation ===")
    print(f"Images evaluated: {len(eval_image_ids)}")
    print(f"Average FPS: {np.mean(fps_list):.2f}")
    print(f"Average inference time: {np.mean(inference_times):.3f} sec")
    print(f"CPU usage: avg {np.mean(cpu_usage_list):.2f}%, max {np.max(cpu_usage_list):.2f}%")
    print(f"Memory usage: avg {np.mean(mem_usage_list):.2f} MB, max {np.max(mem_usage_list):.2f} MB")
    print(f"Recall (manual IoU={IOU_THRESH}): {recall:.4f}")
    print(f"Precision (manual IoU={IOU_THRESH}): {precision:.4f}")
    print(f"F1-score (manual IoU={IOU_THRESH}): {f1:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"COCO mAP (0.50:0.95 IoU): {map50_95:.4f}")


if __name__ == "__main__":
    evaluate_rf_detr_onnx()