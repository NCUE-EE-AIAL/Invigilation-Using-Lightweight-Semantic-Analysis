import os
import shutil
import random

def split_test_set(source_folder, target_folder, test_ratio=0.2):
    """
    從每個子資料夾中隨機選取指定比例的PNG檔案來建立測試集。
    :param source_folder: 包含子資料夾的母資料夾路徑。
    :param target_folder: 存儲測試集的目標資料夾路徑。
    :param test_ratio: 用於測試集的檔案比例。
    """
    # 確保目標資料夾存在
    os.makedirs(target_folder, exist_ok=True)

    # 遍歷母資料夾中的所有子資料夾
    for subdir in os.listdir(source_folder):
        full_subdir_path = os.path.join(source_folder, subdir)
        if os.path.isdir(full_subdir_path):
            # 收集所有PNG檔案
            png_files = [f for f in os.listdir(full_subdir_path) if f.endswith('.png')]
            # 計算要移動到測試集的檔案數量
            num_test_files = int(len(png_files) * test_ratio)
            # 隨機選取檔案
            test_files = random.sample(png_files, num_test_files)
            
            # 確保對應的子資料夾在目標資料夾中存在
            target_subdir = os.path.join(target_folder, subdir)
            os.makedirs(target_subdir, exist_ok=True)
            
            # 移動選定的檔案到目標資料夾
            for file in test_files:
                src_file = os.path.join(full_subdir_path, file)
                dst_file = os.path.join(target_subdir, file)
                shutil.move(src_file, dst_file)
                print(f"Moved {src_file} to {dst_file}")

# 使用示例
source_directory = r'D:\NCUE_lab\Data\Classificaiton\Students_L_training'
target_directory = r'D:\NCUE_lab\Data\Classificaiton\Students_L_testing'
split_test_set(source_directory, target_directory)
