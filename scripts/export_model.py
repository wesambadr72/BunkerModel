from ultralytics import YOLO
import os

def export_to_tensorrt():
    # مسار أفضل نموذج تم تدريبه
    model_path = r'C:\Users\Admin\Documents\trae_projects\BunkerModel\waste_classification\yolov8n_cls_v15\weights\best.pt'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # تحميل النموذج
    model = YOLO(model_path)

    print("Starting export to TensorRT format...")
    # تصدير النموذج
    # ملاحظة: التصدير لـ TensorRT يتطلب وجود TensorRT مثبت محلياً
    # إذا لم يكن متاحاً على جهاز التدريب، سنقوم بالتصدير لـ ONNX أولاً
    try:
        model.export(format='engine', device=0) # engine = TensorRT
        print("Export to TensorRT successful!")
    except Exception as e:
        print(f"TensorRT export failed (usually due to missing local TensorRT libs): {e}")
        print("Exporting to ONNX instead for transfer to Jetson...")
        model.export(format='onnx')
        print("Export to ONNX successful!")

if __name__ == '__main__':
    export_to_tensorrt()
