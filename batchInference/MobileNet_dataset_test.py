import os
import time
import cv2
import numpy as np
import psutil
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import onnxruntime as ort

# ==============================
# Configs
# ==============================
IMAGE_FOLDER = "val2017"  # COCO val images
ANNOTATION_FILE = "annotations/instances_val2017.json"
MODEL_PATH = "../onnx_models/ssd_mobilenet_v1_12.onnx"
INPUT_SIZE = 300           # SSD MobileNet input size
CONF_THRESH = 0.3
IOU_THRESH = 0.5
MAX_IMAGES = 0
# ==============================

# Load COCO annotations
coco = COCO(ANNOTATION_FILE)
catId_to_name = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}
image_ids = list(coco.imgs.keys())
eval_image_ids = image_ids[:MAX_IMAGES] if MAX_IMAGES > 0 else image_ids

# Initialize ONNX Runtime session
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
process = psutil.Process(os.getpid())

# IoU computation
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# Evaluation function
def evaluate_ssd_mobilenet():
    tp_all, fp_all, fn_all = 0, 0, 0
    fps_list, inference_times = [], []
    cpu_usage_list, mem_usage_list = [], []
    iou_list = []
    coco_results = []

    for img_id in tqdm(eval_image_ids, desc="Evaluating"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(IMAGE_FOLDER, img_info['file_name'])
        image = cv2.imread(img_path)
        orig_h, orig_w, _ = image.shape

        # Ground truth boxes
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt_boxes = [[ann['bbox'][0], ann['bbox'][1], ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3]] for ann in anns]

        # Preprocess for SSD MobileNet (uint8 input)
        image_resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        input_tensor = np.expand_dims(image_rgb.astype(np.uint8), axis=0)

        # Record CPU/memory BEFORE inference
        cpu_before = psutil.cpu_percent(interval=None) / 100
        mem_before = process.memory_info().rss / (1024**2)

        # Run inference
        start = time.time()
        outputs = session.run(None, {input_name: input_tensor})
        end = time.time()
        inf_time = end - start
        inference_times.append(inf_time)

        # Record CPU/memory AFTER inference
        cpu_after = psutil.cpu_percent(interval=None) / 100
        mem_after = process.memory_info().rss / (1024**2)

        cpu_usage_list.append(max(cpu_before, cpu_after) * 100)
        mem_usage_list.append(max(mem_before, mem_after))
        fps_list.append(1/inf_time)

        # ONNX SSD outputs: [boxes, labels, scores]
        boxes = outputs[0][0]        # shape: [num_detections, 4], normalized xyxy
        scores = outputs[2][0]       # shape: [num_detections]
        classes = outputs[1][0].astype(int)
        num_detections = boxes.shape[0]

        preds = []
        for i in range(num_detections):
            if scores[i] < CONF_THRESH:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            x1 = int(xmin * orig_w)
            y1 = int(ymin * orig_h)
            x2 = int(xmax * orig_w)
            y2 = int(ymax * orig_h)
            preds.append([x1, y1, x2, y2])

            # Save for COCO evaluation
            coco_results.append({
                "image_id": img_id,
                "category_id": classes[i],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(scores[i])
            })

        # IoU matching
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
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    avg_iou = np.mean(iou_list) if iou_list else 0.0

    # COCO evaluation
    coco_dt = coco.loadRes(coco_results)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.params.imgIds = eval_image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    map50_95 = coco_eval.stats[0]

    print("=== SSD MobileNet v1 Evaluation ===")
    print(f"Images evaluated: {len(eval_image_ids)}")
    print(f"Average FPS: {np.mean(fps_list):.2f}")
    print(f"Average inference time: {np.mean(inference_times):.3f} sec")
    print(f"CPU usage (avg/max): {np.mean(cpu_usage_list):.2f}% / {np.max(cpu_usage_list):.2f}%")
    print(f"Memory usage (avg/max MB): {np.mean(mem_usage_list):.2f} / {np.max(mem_usage_list):.2f}")
    print(f"Recall (manual IoU={IOU_THRESH}): {recall:.4f}")
    print(f"Precision (manual IoU={IOU_THRESH}): {precision:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"COCO mAP (0.50:0.95 IoU): {map50_95:.4f}")

# Run evaluation
evaluate_ssd_mobilenet()