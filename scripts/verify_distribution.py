import os

def verify_distribution(base_path):
    splits = ['train', 'val', 'test']
    print(f"{'Split/Class':<25} | {'Count':<5}")
    print("-" * 35)
    
    for split in splits:
        split_path = os.path.join(base_path, split)
        if not os.path.exists(split_path):
            print(f"Directory {split_path} does not exist.")
            continue
            
        # Get classes (subdirectories)
        classes = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
        
        for cls in classes:
            cls_path = os.path.join(split_path, cls)
            count = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"{f'{split}/{cls}':<25} | {count:<5}")

if __name__ == "__main__":
    processed_path = 'data/processed'
    verify_distribution(processed_path)
