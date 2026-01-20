from ultralytics import YOLO
import os

def train():
    # تحميل نموذج التصنيف (ليس Detection)
    model = YOLO('yolov8n-cls.pt')  # ⭐ استخدم -cls وليس عادي

    # التدريب
    results = model.train(
        data='data/processed',
        epochs=100,
        imgsz=224,           # ⭐ تقليل الحجم للـ classification
        batch=32,            # ⭐ يمكن زيادة batch
        patience=15,         # Early stopping
        optimizer='AdamW',
        lr0=0.001,
        weight_decay=0.0005,

        # Augmentation
        hsv_h=0.02,      
        hsv_s=0.8,       
        hsv_v=0.5,       
        degrees=25,      
        translate=0.15,  
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,      
        
        # ⭐ أضف هذه:
        copy_paste=0.1, 
        erasing=0.3,    

        # Hardware
        device=0,            
        workers=8,           

        # Validation
        val=True,
        save=True,
        project='waste_classification',
        name='yolov8n_cls_v1',
        verbose=True,
        plots=False          # تعطيل الرسوم البيانية لتجنب تعارض matplotlib
    )

    print(f"Best accuracy:{results.results_dict['metrics/accuracy_top1']}")

if __name__ == '__main__':
    train()