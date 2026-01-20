import os

def explore_data(base_path):
    print(f"Exploring data in: {base_path}")
    if not os.path.exists(base_path):
        print(f"Path {base_path} does not exist.")
        return

    for cls in os.listdir(base_path):
        cls_path = os.path.join(base_path, cls)
        if os.path.isdir(cls_path):
            count = len(os.listdir(cls_path))
            print(f"{cls}: {count} صورة")

if __name__ == "__main__":
    # Pointing to the unzipped TrashNet dataset
    raw_data_path = 'data/raw/dataset-resized'
    explore_data(raw_data_path)
