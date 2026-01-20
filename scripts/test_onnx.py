import cv2 
import onnxruntime as ort 
import numpy as np 
import os

def test_onnx_inference():
    model_path = r'C:\Users\Admin\Documents\trae_projects\BunkerModel\waste_classification\yolov8n_cls_v15\weights\best.onnx'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Initialize Session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers) 
    
    # البحث عن أي صورة اختبار
    test_img_path = ""
    test_dir = 'data/processed/test'
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                test_img_path = os.path.join(root, file)
                break
        if test_img_path: break

    if not test_img_path:
        print("Error: No test image found in data/processed/test")
        return

    print(f"Testing with image: {test_img_path}")
    
    # تحميل صورة اختبار 
    img = cv2.imread(test_img_path) 
    img_resized = cv2.resize(img, (224, 224)) 
    img_normalized = img_resized.astype(np.float32) / 255.0 
    img_transposed = np.transpose(img_normalized, (2, 0, 1)) 
    img_batch = np.expand_dims(img_transposed, 0) 
    
    # Inference 
    # YOLOv8-CLS ONNX input name is usually 'images'
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_batch}) 
    
    print(f"Output shape: {output[0].shape}") 
    print(f"Top-1 Prediction (Softmax): {np.max(output[0])}")
    print(f"Predicted Class Index: {np.argmax(output[0])}")

if __name__ == '__main__':
    test_onnx_inference()
