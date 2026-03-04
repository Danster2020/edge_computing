import os
import time
import cv2
import numpy as np
import psutil
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

# ==============================
# Config
# ==============================
IMAGE_FOLDER = "val2017"
ANNOTATION_FILE = "annotations/instances_val2017.json"
MODEL_PATH = "../onnx_models/yolo5n.onnx"  # Your ONNX model
INPUT_SIZE = 640
CONF_THRESH = 0.3
IOU_THRESH = 0.5
MAX_IMAGES = 0
# ==============================

# CPU threads
num_cores = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)

# Load COCO
coco = COCO(ANNOTATION_FILE)
image_ids = list(coco.imgs.keys())
eval_image_ids = image_ids[:MAX_IMAGES] if MAX_IMAGES > 0 else image_ids

# YOLO → COCO category mapping
coco_categories = coco.loadCats(coco.getCatIds())
coco_categories = sorted(coco_categories, key=lambda x: x["id"])
yolo_to_coco = {i: cat["id"] for i, cat in enumerate(coco_categories)}

# Load ONNX model
ort_session = ort.InferenceSession(
    MODEL_PATH,
    providers=['CPUExecutionProvider']
)

process = psutil.Process(os.getpid())

# ==============================
# IoU
# ==============================
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    return interArea / (boxAArea + boxBArea - interArea + 1e-6)

# ==============================
# Preprocess (letterbox)
# ==============================
def preprocess(img, input_size=640):
    h, w = img.shape[:2]
    scale = input_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    img_resized = cv2.resize(img, (nw, nh))

    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    top = (input_size - nh) // 2
    left = (input_size - nw) // 2
    canvas[top:top+nh, left:left+nw] = img_resized

    img_input = canvas[:, :, ::-1]
    img_input = img_input.transpose(2, 0, 1)
    img_input = img_input.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, 0)

    return img_input, scale, top, left

# ==============================
# ONNX Postprocess (NMS already fused)
# Output: [1, N, 6]
# [x1, y1, x2, y2, score, class]
# ==============================
def postprocess(preds, scale, top, left, conf_thresh=0.3):
    boxes, scores, classes = [], [], []
    
    for det in preds[0]:  # [num_boxes, 85]
        objectness = det[4]
        class_probs = det[5:]
        class_id = np.argmax(class_probs)
        class_conf = class_probs[class_id]
        conf = objectness * class_conf
        
        if conf < conf_thresh:
            continue
        
        # xywh to x1y1x2y2
        x_center, y_center, w, h = det[:4]
        x1 = (x_center - w / 2 - left) / scale
        y1 = (y_center - h / 2 - top) / scale
        x2 = (x_center + w / 2 - left) / scale
        y2 = (y_center + h / 2 - top) / scale
        
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
        scores.append(float(conf))
        classes.append(int(class_id))
        
    return boxes, scores, classes

# ==============================
# Evaluation
# ==============================
def evaluate():
    tp_all, fp_all, fn_all = 0, 0, 0
    fps_list, inference_times = [], []
    cpu_usage_list, mem_usage_list = [], []
    iou_list = []
    coco_results = []

    input_name = ort_session.get_inputs()[0].name

    for img_id in tqdm(eval_image_ids, desc="Evaluating YOLO ONNX"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(IMAGE_FOLDER, img_info['file_name'])
        image = cv2.imread(img_path)

        # Ground truth
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt_boxes = [
            [ann['bbox'][0],
             ann['bbox'][1],
             ann['bbox'][0] + ann['bbox'][2],
             ann['bbox'][1] + ann['bbox'][3]]
            for ann in anns
        ]

        # CPU/mem before
        cpu_before = psutil.cpu_percent(interval=None) / 100
        mem_before = process.memory_info().rss / (1024**2)

        # Inference
        img_input, scale, top, left = preprocess(image, INPUT_SIZE)

        start = time.time()
        preds = ort_session.run(None, {input_name: img_input})[0]
        end = time.time()

        inf_time = end - start
        inference_times.append(inf_time)
        fps_list.append(1 / inf_time)

        # CPU/mem after
        cpu_after = psutil.cpu_percent(interval=None) / 100
        mem_after = process.memory_info().rss / (1024**2)

        cpu_usage_list.append(max(cpu_before, cpu_after) * 100)
        mem_usage_list.append(max(mem_before, mem_after))

        # Postprocess
        pred_boxes, pred_scores, pred_classes = postprocess(preds, scale, top, left, CONF_THRESH)

        # COCO results
        for box, score, cls in zip(pred_boxes, pred_scores, pred_classes):
            x1, y1, x2, y2 = box
            category_id = yolo_to_coco[cls]

            coco_results.append({
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": score
            })

        # Manual IoU metrics
        matched_gt = set()
        for pred_box in pred_boxes:
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

    # Manual metrics
    recall = tp_all / (tp_all + fn_all + 1e-6)
    precision = tp_all / (tp_all + fp_all + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    avg_iou = np.mean(iou_list) if iou_list else 0

    # COCO mAP
    coco_dt = coco.loadRes(coco_results)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.params.imgIds = eval_image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map50_95 = coco_eval.stats[0]

    print("\n=== YOLO ONNX Evaluation ===")
    print(f"Images evaluated: {len(eval_image_ids)}")
    print(f"Average FPS: {np.mean(fps_list):.2f}")
    print(f"Average inference time: {np.mean(inference_times):.3f} sec")
    print(f"CPU usage: avg {np.mean(cpu_usage_list):.2f}%, max {np.max(cpu_usage_list):.2f}%")
    print(f"Memory usage: avg {np.mean(mem_usage_list):.2f} MB, max {np.max(mem_usage_list):.2f} MB")
    print(f"Recall (manual IoU={IOU_THRESH}): {recall:.4f}")
    print(f"Precision (manual IoU={IOU_THRESH}): {precision:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"COCO mAP (0.50:0.95 IoU): {map50_95:.4f}")

if __name__ == "__main__":
    evaluate()