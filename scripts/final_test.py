from ultralytics import YOLO
import os

def run_final_test():
    # 1. تحميل النموذج
    model_path = r'C:\Users\Admin\Documents\trae_projects\BunkerModel\waste_classification\yolov8n_cls_v15\weights\best.pt'
    data_path = r'C:\Users\Admin\Documents\trae_projects\BunkerModel\data\waste_data.yaml'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model = YOLO(model_path)

    # 2. اختبار على test set
    print("Running evaluation on test set...")
    results = model.val(data='data/processed', split='test', project='results', name='final_test_eval', exist_ok=True)

    print("\n" + "="*30)
    print(f"Test Accuracy (Top-1): {results.top1:.2%}")
    print(f"Test Top-5 Accuracy: {results.top5:.2%}")
    print("="*30)

    # 3. حفظ Confusion Matrix
    # النتائج تحفظ تلقائياً في مجلد المشروع المحدد 'results/final_test_eval'
    print(f"Results and Confusion Matrix saved to: results/final_test_eval/")

if __name__ == '__main__':
    run_final_test()
