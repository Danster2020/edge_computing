import time
import csv
import os
import sys
from collections import Counter
from benchmark import Benchmark
from camera_source import open_best_camera
from model_selection import choose_model
from models.efficientdet_decoder import EfficientDetD0Model
from models.yolo_decoder import YoloModel
from models.rf_detr_decoder import RfDetrModel
from models.ssd_mobilenet_decoder import SsdMobilenetV1Model
from temp import get_cpu_temp
import psutil

BENCHMARK_SECONDS = 2


def load_model(path):
    if "efficientdet" in path.lower():
        return EfficientDetD0Model(path)
    if "rf-detr" in path.lower() or "rfdetr" in path.lower():
        return RfDetrModel(path)
    if "ssd_mobilenet" in path.lower() or "ssd-mobilenet" in path.lower() or "ssdmobilenet" in path.lower():
        return SsdMobilenetV1Model(path)
    return YoloModel(path)


def main():

    # ==========================
    # Get experiment name from CLI
    # ==========================
    if len(sys.argv) < 2:
        print("Usage: python3 run_webcam_benchmark.py <experiment_name>")
        sys.exit(1)

    experiment_name = sys.argv[1]

    try:
        model_name, model_path = choose_model()
    except RuntimeError as e:
        print(str(e))
        return
    print(f"Selected model: {model_name}")

    # Create simulations folder if not exists
    os.makedirs("simulations", exist_ok=True)

    # ==========================
    # Setup
    # ==========================
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {model_path}")
        print(f"Reason: {e}")
        if "efficientdet" in model_path.lower():
            print(
                "efficientdet-d0.onnx appears invalid (missing graph weights). "
                "Re-export/download a valid ONNX file."
            )
        return
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
    print("\nWarming up...")
    for _ in range(20):
        ret, frame = cam.read()
        if ret:
            model(frame)
    print("Warmup DONE")

    print(f"\nRUNNING BENCHMARK FOR {BENCHMARK_SECONDS} SECONDS...")
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
        process = psutil.Process()
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
            "Inference latency(ms)",
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
        f.write(f"Average inference latency: {avg_latency:.2f} ms\n")
        f.write(f"95th percentile inference latency: {p95_latency:.2f} ms\n")
        f.write(f"Average FPS: {avg_fps:.2f}\n\n")
        
        if detection_rows:
            cpu_values = [row[4] for row in detection_rows]
            memory_values = [row[5] for row in detection_rows]
            temp_values = [float(row[6]) for row in detection_rows if row[6] != ""]

            avg_cpu = sum(cpu_values) / len(cpu_values)
            min_cpu = min(cpu_values)
            max_cpu = max(cpu_values)

            avg_memory = sum(memory_values) / len(memory_values)
            min_memory = min(memory_values)
            max_memory = max(memory_values)

            frames_with_detections = sum(1 for row in detection_rows if row[2] > 0)
            detection_pct = (frames_with_detections / len(detection_rows)) * 100

            if temp_values:
                avg_cpu_temp = sum(temp_values) / len(temp_values)
                min_cpu_temp = min(temp_values)
                max_cpu_temp = max(temp_values)
            else:
                avg_cpu_temp = None
                min_cpu_temp = None
                max_cpu_temp = None

        else:
            avg_cpu = min_cpu = max_cpu = 0.0
            avg_memory = min_memory = max_memory = 0.0
            detection_pct = 0.0
            avg_cpu_temp = min_cpu_temp = max_cpu_temp = None

        f.write(f"Frames with detections: {detection_pct:.2f} %\n")

        f.write(f"Average CPU usage: {avg_cpu:.2f} %\n")
        f.write(f"Min CPU usage: {min_cpu:.2f} %\n")
        f.write(f"Max CPU usage: {max_cpu:.2f} %\n")

        f.write(f"Average Memory usage: {avg_memory:.2f} MB\n")
        f.write(f"Min Memory usage: {min_memory:.2f} MB\n")
        f.write(f"Max Memory usage: {max_memory:.2f} MB\n")

        f.write(
            f"Average CPU temperature: {avg_cpu_temp:.2f} C\n"
            if avg_cpu_temp is not None else
            "Average CPU temperature: N/A\n"
        )

        f.write(
            f"Min CPU temperature: {min_cpu_temp:.2f} C\n"
            if min_cpu_temp is not None else
            "Min CPU temperature: N/A\n"
        )

        f.write(
            f"Max CPU temperature: {max_cpu_temp:.2f} C\n"
            if max_cpu_temp is not None else
            "Max CPU temperature: N/A\n"
        )
        
        f.write("\n---- Detected objects summary ----\n")
        total_detected_objects = sum(class_counts.values())
        for cls, count in class_counts.most_common():
            class_pct = (count / total_detected_objects) * 100 if total_detected_objects > 0 else 0.0
            f.write(f"{cls}: {count} ({class_pct:.2f}%)\n")
        if total_detected_objects == 0:
            f.write("No objects detected.\n")
        print()

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
