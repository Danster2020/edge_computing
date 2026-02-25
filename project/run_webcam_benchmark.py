import time
import csv
import os
import sys
from collections import Counter
from benchmark import Benchmark
from camera_source import open_best_camera
from models.yolo_decoder import YoloModel
from models.rf_detr_decoder import RfDetrModel
from temp import get_cpu_temp
import psutil

MODEL_NAME = "rf-detr-base-coco" # yolo11n rf-detr-base-coco
MODEL_PATH = f"onnx_models/{MODEL_NAME}.onnx"
BENCHMARK_SECONDS = 60


def load_model(path):
    if "rf-detr" in path.lower() or "rfdetr" in path.lower():
        return RfDetrModel(path)
    return YoloModel(path)


def main():

    # ==========================
    # Get experiment name from CLI
    # ==========================
    if len(sys.argv) < 2:
        print("Usage: python3 run_webcam_benchmark.py <experiment_name>")
        sys.exit(1)

    experiment_name = sys.argv[1]

    # Extract model name from path
    model_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]

    # Create simulations folder if not exists
    os.makedirs("simulations", exist_ok=True)

    # ==========================
    # Setup
    # ==========================
    model = load_model(MODEL_PATH)
    bench = Benchmark()
    try:
        cam = open_best_camera()
    except RuntimeError as e:
        print(str(e))
        return
    print(f"Using camera source: {cam.kind}")

    # Output filenames
    csv_path = f"simulations/{model_name}_{cam.kind}_{experiment_name}.csv"
    summary_path = f"simulations/{model_name}_{cam.kind}_{experiment_name}_summary.txt"

    # Warmup
    for _ in range(20):
        ret, frame = cam.read()
        if ret:
            model(frame)

    print(f"\nRunning benchmark for {BENCHMARK_SECONDS} seconds...")
    start_time = time.perf_counter()

    frame_id = 0
    detection_rows = []
    process = psutil.Process()


    # ==========================
    # Benchmark loop
    # ==========================
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        latency = bench.measure(model, frame)
        boxes, scores, class_ids = model.get_detections()

        labels = [model.class_names[c] for c in class_ids]
        count = len(labels)
        
        # CPU/memory logging
        cpu_percent = process.cpu_percent(interval=None)
        memory_mb = process.memory_info().rss / (1024 * 1024)
        cpu_temp_c = get_cpu_temp()
        cpu_temp_c = cpu_temp_c if cpu_temp_c is not None else ""

        detection_rows.append([
            frame_id,
            latency * 1000,
            count,
            ",".join(labels),
            cpu_percent,
            memory_mb,
            cpu_temp_c
        ])

        frame_id += 1

        if time.perf_counter() - start_time >= BENCHMARK_SECONDS:
            break

    cam.release()

    # ==========================
    # Results
    # ==========================
    avg_latency = bench.average_latency_ms()
    p95_latency = bench.percentile_latency_ms()
    avg_fps = 1000 / avg_latency if avg_latency > 0 else 0

    # Aggregate detections
    all_labels = []
    for row in detection_rows:
        if row[3]:
            all_labels.extend(row[3].split(","))

    class_counts = Counter(all_labels)

    # ==========================
    # Save CSV
    # ==========================
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Frame",
            "Latency(ms)",
            "NumDetections",
            "DetectedClasses",
            "CPU(%)",
            "Memory(MB)",
            "CPU_Temp(C)"
        ])
        writer.writerows(detection_rows)

    # ==========================
    # Save Summary TXT
    # ==========================
    with open(summary_path, "w") as f:
        f.write("===== BENCHMARK SUMMARY =====\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Duration: {BENCHMARK_SECONDS} seconds\n")
        f.write(f"Frames processed: {frame_id}\n")
        f.write(f"Average latency: {avg_latency:.2f} ms\n")
        f.write(f"95th percentile latency: {p95_latency:.2f} ms\n")
        f.write(f"Average FPS: {avg_fps:.2f}\n\n")

        f.write("Detected objects summary:\n")
        for cls, count in class_counts.items():
            f.write(f"{cls}: {count}\n")
        
        if detection_rows:
            avg_cpu = sum(row[4] for row in detection_rows) / len(detection_rows)
            avg_memory = sum(row[5] for row in detection_rows) / len(detection_rows)
        else:
            avg_cpu = 0.0
            avg_memory = 0.0
        temp_values = [float(row[6]) for row in detection_rows if row[6] != ""]
        avg_cpu_temp = sum(temp_values) / len(temp_values) if temp_values else 0.0
        f.write(f"Average CPU usage: {avg_cpu:.2f} %\n")
        f.write(f"Average Memory usage: {avg_memory:.2f} MB\n")
        f.write(f"Average CPU temperature: {avg_cpu_temp:.2f} C\n")

    # ==========================
    # Print results
    # ==========================
    print("\n===== BENCHMARK COMPLETE =====")
    print(f"CSV saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")
    if not detection_rows:
        print("No frames were processed. Check camera connection and permissions.")


if __name__ == "__main__":
    main()
