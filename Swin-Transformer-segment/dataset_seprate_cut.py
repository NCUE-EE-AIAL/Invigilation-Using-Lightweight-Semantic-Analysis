import os
import shutil
from sklearn.model_selection import train_test_split

# 定义原始数据集路径和目标路径Name of the founder
original_images_dir = r"D:\NCUE_lab\Data\Segment\512x512\Image"
original_masks_dir = r"D:\NCUE_lab\Data\Segment\512x512\Label"

train_images_dir = r"D:\NCUE_lab\Data\Segment\512x512\Train\Images"
train_masks_dir = r"D:\NCUE_lab\Data\Segment\512x512\Train\Labels"
val_images_dir = r"D:\NCUE_lab\Data\Segment\512x512\Val\Images"
val_masks_dir = r"D:\NCUE_lab\Data\Segment\512x512\Val\Labels"
test_images_dir = r"D:\NCUE_lab\Data\Segment\512x512\Test\Images"
test_masks_dir = r"D:\NCUE_lab\Data\Segment\512x512\Test\Labels"

# 创建目标文件夹
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_masks_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_masks_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_masks_dir, exist_ok=True)

# 获取所有文件名
images = sorted([f for f in os.listdir(original_images_dir) if f.endswith('.jpg')])
masks = sorted([f for f in os.listdir(original_masks_dir) if f.endswith('.png')])

# 确保图像和掩码匹配
assert len(images) == len(masks), "Images and masks count do not match"

# 分割数据集
train_images, test_images, train_masks, test_masks = train_test_split(images, masks, test_size=0.3, random_state=42)
val_images, test_images, val_masks, test_masks = train_test_split(test_images, test_masks, test_size=1/3, random_state=42)

# 定义移动文件函数
def move_files(file_list, source_dir, target_dir):
    for file_name in file_list:
        shutil.move(os.path.join(source_dir, file_name), os.path.join(target_dir, file_name))

# 移动文件到目标文件夹
move_files(train_images, original_images_dir, train_images_dir)
move_files(train_masks, original_masks_dir, train_masks_dir)
move_files(val_images, original_images_dir, val_images_dir)
move_files(val_masks, original_masks_dir, val_masks_dir)
move_files(test_images, original_images_dir, test_images_dir)
move_files(test_masks, original_masks_dir, test_masks_dir)

print("Data split and move completed.")
