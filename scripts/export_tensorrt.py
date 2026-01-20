# Export YOLO model to TensorRT format (Run this on Jetson Nano)
import tensorrt as trt
import os
from pathlib import Path

def build_engine(onnx_path, engine_path, use_int8=False):
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found at {onnx_path}")
        return

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    
    # Define network flags
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    
    parser = trt.OnnxParser(network, logger)
    
    print(f"Parsing ONNX file: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    config = builder.create_builder_config()
    # Use recommended way to set memory pool limit (replacement for max_workspace_size)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 * 1024 * 1024) # 256MB

    if use_int8:
        if not builder.platform_has_fast_int8:
            print("Warning: This platform does not support fast INT8. Falling back to FP16.")
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        config.set_flag(trt.BuilderFlag.FP16)

    # Optimization Profile for static input shape
    profile = builder.create_optimization_profile()
    profile.set_shape('images', (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224))
    config.add_optimization_profile(profile)

    print(f"Building TensorRT engine (INT8={use_int8})... this may take a few minutes.")
    # build_serialized_network is the modern way (replaces build_engine)
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("Error: Failed to build the engine.")
        return

    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"✅ Success! Engine saved to: {engine_path}")

if __name__ == '__main__':
    # المسارات الافتراضية بناءً على هيكلة المشروع
    onnx_file = r'waste_classification/yolov8n_cls_v15/weights/best.onnx'
    
    print("--- Starting TensorRT Conversion ---")
    
    # 1. تحويل لـ FP16
    print("\n[1/2] Building FP16 Engine...")
    build_engine(onnx_file, 'models/best_fp16.engine', use_int8=False)
    
    # 2. تحويل لـ INT8
    print("\n[2/2] Building INT8 Engine...")
    build_engine(onnx_file, 'models/best_int8.engine', use_int8=True)
