#!/usr/bin/env python3 
""" 
Benchmark شامل للنموذج (PyTorch vs ONNX)
""" 
import time 
import numpy as np 
import cv2 
from ultralytics import YOLO 
import onnxruntime as ort 
import os
import json

def benchmark_model(): 
    # المسارات
    pt_model_path = r'C:\Users\Admin\Documents\trae_projects\BunkerModel\waste_classification\yolov8n_cls_v15\weights\best.pt'
    onnx_model_path = r'C:\Users\Admin\Documents\trae_projects\BunkerModel\waste_classification\yolov8n_cls_v15\weights\best.onnx'
    test_dir = 'data/processed/test' 

    if not os.path.exists(pt_model_path) or not os.path.exists(onnx_model_path):
        print("Error: Model files not found.")
        return

    # 1. تحميل النماذج
    pt_model = YOLO(pt_model_path) 
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
    onnx_session = ort.InferenceSession(onnx_model_path, providers=providers) 
    
    # 2. تحميل صور الاختبار 
    test_images = [] 
    for cls in os.listdir(test_dir): 
        cls_dir = os.path.join(test_dir, cls) 
        if not os.path.isdir(cls_dir): continue
        for img_name in os.listdir(cls_dir)[:10]:  # 10 صور لكل فئة 
            img_path = os.path.join(cls_dir, img_name) 
            img = cv2.imread(img_path)
            if img is not None:
                test_images.append(img) 
    
    print(f"📸 اختبار على {len(test_images)} صورة...") 
    
    # 3. Benchmark PyTorch 
    print("\nBenchmarking PyTorch Model...")
    pt_latencies = [] 
    for img in test_images: 
        start = time.time() 
        _ = pt_model(img, verbose=False) 
        pt_latencies.append((time.time() - start) * 1000) 
    
    # 4. Benchmark ONNX 
    print("Benchmarking ONNX Model...")
    onnx_latencies = [] 
    input_name = onnx_session.get_inputs()[0].name
    for img in test_images: 
        # Preprocess
        img_resized = cv2.resize(img, (224, 224)) 
        img_normalized = img_resized.astype(np.float32) / 255.0 
        img_transposed = np.transpose(img_normalized, (2, 0, 1)) 
        img_batch = np.expand_dims(img_transposed, 0) 
        
        start = time.time() 
        _ = onnx_session.run(None, {input_name: img_batch}) 
        onnx_latencies.append((time.time() - start) * 1000) 
    
    # 5. طباعة النتائج بشكل جدول
    print("\n" + "="*60) 
    print(f"{'Metric':<20} | {'PyTorch (.pt)':<15} | {'ONNX (.onnx)':<15}")
    print("-" * 60)
    
    metrics = [
        ("Mean Latency", pt_latencies, onnx_latencies, "ms"),
        ("Std Latency", pt_latencies, onnx_latencies, "ms"),
        ("Min Latency", pt_latencies, onnx_latencies, "ms"),
        ("Max Latency", pt_latencies, onnx_latencies, "ms"),
    ]

    for label, pt_data, onnx_data, unit in metrics:
        if "Mean" in label:
            p_val, o_val = np.mean(pt_data), np.mean(onnx_data)
        elif "Std" in label:
            p_val, o_val = np.std(pt_data), np.std(onnx_data)
        elif "Min" in label:
            p_val, o_val = np.min(pt_data), np.min(onnx_data)
        elif "Max" in label:
            p_val, o_val = np.max(pt_data), np.max(onnx_data)
        
        print(f"{label:<20} | {p_val:>10.2f} {unit} | {o_val:>10.2f} {unit}")

    print("-" * 60)
    print(f"{'FPS (Mean)':<20} | {1000/np.mean(pt_latencies):>10.1f} FPS | {1000/np.mean(onnx_latencies):>10.1f} FPS")
    print("="*60) 
    
    # 6. التوقعات (Comparison with Industry Standards)
    print("\n" + "="*60) 
    print("Expected Performance Comparison:")
    print("="*60) 
    print(f"Current ONNX Speed: {1000/np.mean(onnx_latencies):.1f} FPS")
    print("TensorRT (Estimated): 400-600 FPS (on RTX 3050)")
    print("TensorRT (Jetson Nano): 40-60 FPS")
    print("="*60) 

    # 7. حفظ النتائج
    os.makedirs('results', exist_ok=True)
    with open('results/benchmark_report.json', 'w') as f:
        json.dump({
            "pytorch": {"mean": np.mean(pt_latencies), "fps": 1000/np.mean(pt_latencies)},
            "onnx": {"mean": np.mean(onnx_latencies), "fps": 1000/np.mean(onnx_latencies)},
            "speedup": np.mean(pt_latencies) / np.mean(onnx_latencies)
        }, f, indent=4)

if __name__ == '__main__': 
    benchmark_model()
