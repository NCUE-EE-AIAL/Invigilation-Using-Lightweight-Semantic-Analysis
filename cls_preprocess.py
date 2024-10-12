import shutil
import random
import os
from PIL import Image

from conf import *

def split_dataset(source_dir, train_dir, val_dir, split_ratio=0.8):
    # Create target directories if they don't exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # Loop through each category
    for category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, category)
        if os.path.isdir(category_path):
            images = os.listdir(category_path)
            random.shuffle(images)

            split_point = int(len(images) * split_ratio)
            train_images = images[:split_point]
            val_images = images[split_point:]

            # Move files to train directory
            train_category_path = os.path.join(train_dir, category)
            val_category_path = os.path.join(val_dir, category)
            os.makedirs(train_category_path, exist_ok=True)
            os.makedirs(val_category_path, exist_ok=True)

            for img in train_images:
                shutil.move(os.path.join(category_path, img), os.path.join(train_category_path, img))

            # Move files to validation directory
            for img in val_images:
                shutil.move(os.path.join(category_path, img), os.path.join(val_category_path, img))

def remove_corrupted_images(directory):
    for category in os.listdir(directory):
        category_path = os.path.join(directory, category)
        if os.path.isdir(category_path):
            for img_file in os.listdir(category_path):
                img_path = os.path.join(category_path, img_file)
                try:
                    img = Image.open(img_path)  # Try to open the image
                    img.verify()  # Check for corruption
                except (IOError, SyntaxError) as e:
                    print(f"Removing corrupted file: {img_path}")
                    os.remove(img_path)

if __name__ == '__main__':
    split_dataset(source_dir, train_dir, val_dir, split_ratio=0.8)

    # Run this on both training and validation datasets
    remove_corrupted_images(train_dir)
    remove_corrupted_images(val_dir)
