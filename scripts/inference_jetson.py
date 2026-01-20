#!/usr/bin/env python3 
""" 
Inference script للعمل على Jetson Nano 
استخدم TensorRT (FP16/INT8) للأداء الأفضل
""" 

import cv2 
import numpy as np 
import tensorrt as trt 
import pycuda.driver as cuda 
import pycuda.autoinit 
import time 
import os

class WasteClassifier: 
    def __init__(self, engine_path): 
        self.logger = trt.Logger(trt.Logger.WARNING) 
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine file not found at {engine_path}")

        with open(engine_path, 'rb') as f: 
            runtime = trt.Runtime(self.logger) 
            self.engine = runtime.deserialize_cuda_engine(f.read()) 
            self.context = self.engine.create_execution_context() 

        # تخصيص Memory 
        self.inputs = [] 
        self.outputs = [] 
        self.bindings = [] 

        for binding in self.engine: 
            # Get binding shape and size
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape) 
            dtype = trt.nptype(self.engine.get_binding_dtype(binding)) 

            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(size, dtype) 
            device_mem = cuda.mem_alloc(host_mem.nbytes) 

            self.bindings.append(int(device_mem)) 

            if self.engine.binding_is_input(binding): 
                self.inputs.append({'host': host_mem, 'device': device_mem}) 
            else: 
                self.outputs.append({'host': host_mem, 'device': device_mem}) 

        self.class_names = [ 
            'PET', 'Aluminum', 'Cardboard', 'Paper', 
            'Glass', 'Metal', 'Organic', 'Non-Recyclable' 
        ] 
 
        self.recyclability_map = { 
            'PET': 'Recyclable', 
            'Aluminum': 'Recyclable', 
            'Cardboard': 'Recyclable', 
            'Paper': 'Recyclable', 
            'Glass': 'Recyclable', 
            'Metal': 'Recyclable', 
            'Organic': 'Organic', 
            'Non-Recyclable': 'Non-Recyclable' 
        } 

    def preprocess(self, img): 
        """تحضير الصورة للـinference (224x224 RGB)""" 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)) 
        img = img.astype(np.float32) / 255.0 
        img = np.transpose(img, (2, 0, 1)) # HWC to CHW
        img = np.expand_dims(img, 0) # Add batch dimension
        return np.ascontiguousarray(img) 

    def infer(self, img): 
        """تشغيل الـinference""" 
        preprocessed = self.preprocess(img) 

        # Copy input to device 
        np.copyto(self.inputs[0]['host'], preprocessed.ravel()) 
        cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host']) 

        # Inference 
        start = time.time() 
        self.context.execute_v2(self.bindings) 
        latency = (time.time() - start) * 1000 

        # Copy output from device 
        cuda.memcpy_dtoh(self.outputs[0]['host'], self.outputs[0]['device']) 

        # Process output (Softmax/Argmax)
        output = self.outputs[0]['host'] 
        class_id = np.argmax(output) 
        confidence = float(output[class_id]) 
        
        # Softmax simple implementation if needed (YOLOv8-cls usually outputs logits)
        exp_out = np.exp(output - np.max(output))
        softmax_out = exp_out / exp_out.sum()
        confidence = float(softmax_out[class_id])

        label = self.class_names[class_id]
        return { 
            'material': label, 
            'class_id': int(class_id), 
            'recyclability': self.recyclability_map.get(label, "Unknown"), 
            'confidence': confidence, 
            'latency_ms': latency 
        } 

# الاستخدام على Jetson Nano 
def run_live_inference(engine_path='models/best_fp16.engine'):
    try:
        classifier = WasteClassifier(engine_path) 
        print(f"Model loaded successfully from {engine_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(0) # 0 for USB camera or CSI camera
    
    print("Starting Live Inference... Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Run Inference
        result = classifier.infer(frame)

        # Draw Results
        label_text = f"{result['material']} ({result['confidence']:.1%})"
        rec_text = f"Status: {result['recyclability']}"
        fps_text = f"Latency: {result['latency_ms']:.1f}ms"

        cv2.putText(frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, rec_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, fps_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

        cv2.imshow("Waste Classification - Jetson Nano", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__': 
    # يمكنك تغيير المسار إلى best_int8.engine إذا قمت بإنشائه
    run_live_inference('models/best_fp16.engine')
