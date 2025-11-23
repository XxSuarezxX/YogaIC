import os
import shutil
import random

SOURCE_DIR = "../data/yoga_posture_dataset"
TARGET_DIR = "../data"

RATIOS = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}

def create_folders(base_path):
    for split in RATIOS.keys():
        split_path = os.path.join(base_path, split)
        os.makedirs(split_path, exist_ok=True)

def split_dataset():
    print("Dividiendo dataset...")

    create_folders(TARGET_DIR)

    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

    for cls in classes:
        print(f"Procesando clase: {cls}")

        cls_path = os.path.join(SOURCE_DIR, cls)
        images = [img for img in os.listdir(cls_path) if img.lower().endswith(("jpg", "jpeg", "png"))]

        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * RATIOS["train"])
        n_val = int(n_total * RATIOS["val"])
        # lo que quede va para test

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split_name, file_list in splits.items():
            split_folder = os.path.join(TARGET_DIR, split_name, cls)
            os.makedirs(split_folder, exist_ok=True)

            for file in file_list:
                src = os.path.join(cls_path, file)
                dst = os.path.join(split_folder, file)
                shutil.copyfile(src, dst)

    print("Listo. Dataset dividido correctamente.")

if __name__ == "__main__":
    split_dataset()
