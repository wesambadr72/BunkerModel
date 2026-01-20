#!/usr/bin/env python3
"""
إعادة توازن Dataset - تقليل فئة Organic
"""
import os
import random
from pathlib import Path
import shutil

def balance_dataset():
    """تقليل Organic وزيادة الباقي"""
    
    # الهدف: 550 صورة لكل فئة تقريباً
    target_per_class = 550
    
    print("Starting dataset balancing...")
    
    # 1. تقليل Organic في مجلد التدريب
    organic_train = Path('data/processed/train/Organic')
    if organic_train.exists():
        organic_images = list(organic_train.glob('*.jpg')) + list(organic_train.glob('*.png')) + list(organic_train.glob('*.jpeg'))
        
        # اختر 385 صورة عشوائية (70% من 550)
        keep_count = min(385, len(organic_images))
        keep_train = random.sample(organic_images, keep_count)
        
        # احذف الباقي
        removed = 0
        for img in organic_images:
            if img not in keep_train:
                img.unlink()
                removed += 1
        
        print(f"✅ Organic (train) تم تقليلها: بقي {len(keep_train)} صورة، تم حذف {removed}")
    
    # 2. تقليل Organic في val و test
    for split in ['val', 'test']:
        split_dir = Path(f'data/processed/{split}/Organic')
        if split_dir.exists():
            images = list(split_dir.glob('*.jpg')) + list(split_dir.glob('*.png')) + list(split_dir.glob('*.jpeg'))
            
            keep_count = 83 if split == 'val' else 82  # 15% من 550
            keep_count = min(keep_count, len(images))
            keep = random.sample(images, keep_count)
            
            removed = 0
            for img in images:
                if img not in keep:
                    img.unlink()
                    removed += 1
            
            print(f"✅ Organic ({split}) تم تقليلها: بقي {len(keep)} صورة، تم حذف {removed}")

if __name__ == '__main__':
    random.seed(42) # For consistency
    balance_dataset()
