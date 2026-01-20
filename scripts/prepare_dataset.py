import os
import shutil
import random

# Mapping from TrashNet folders to our new classes
MAPPING = {
    'paper': 'Paper',
    'cardboard': 'Cardboard',
    'metal': 'Metal',
    'plastic': 'PET',
    'glass': 'Glass',
    'trash': 'Non-Recyclable',
    'organic': 'Organic',
    'aluminum': 'Aluminum'
}

# All 8 required classes for YOLO format
ALL_CLASSES = ['PET', 'Aluminum', 'Cardboard', 'Paper', 'Glass', 'Metal', 'Organic', 'Non-Recyclable']

def prepare_dataset(source_base, target_base, split_ratio=(0.70, 0.15, 0.15)):
    # Create target directories
    for split in ['train', 'val', 'test']:
        for cls in ALL_CLASSES:
            os.makedirs(os.path.join(target_base, split, cls), exist_ok=True)

    print(f"Starting dataset preparation from {source_base} to {target_base}...")

    # Process existing TrashNet classes
    for src_cls, target_cls in MAPPING.items():
        src_path = os.path.join(source_base, src_cls)
        if not os.path.exists(src_path):
            print(f"Warning: Source class {src_cls} not found at {src_path}")
            continue

        files = [f for f in os.listdir(src_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(files)

        total = len(files)
        train_idx = int(total * split_ratio[0])
        val_idx = train_idx + int(total * split_ratio[1])

        splits = {
            'train': files[:train_idx],
            'val': files[train_idx:val_idx],
            'test': files[val_idx:]
        }

        for split_name, split_files in splits.items():
            for f in split_files:
                src_file = os.path.join(src_path, f)
                dst_file = os.path.join(target_base, split_name, target_cls, f)
                shutil.copy2(src_file, dst_file)
        
        print(f"Processed {src_cls} -> {target_cls}: {total} images split into {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")

    print("\nDataset preparation complete.")
    print(f"Note: Folders for 'Metal', 'Organic' have been created in {target_base} but are currently empty.")

if __name__ == "__main__":
    random.seed(42) # For reproducibility
    source_dir = 'data/raw/dataset-resized'
    target_dir = 'data/processed'
    prepare_dataset(source_dir, target_dir)
