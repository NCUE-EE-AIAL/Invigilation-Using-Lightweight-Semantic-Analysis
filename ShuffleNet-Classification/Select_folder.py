import os
import shutil

def count_and_move_png_files(source_dir, target_dir, min_count=30):
    # 確保目標資料夾存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 準備一個檔案來記錄結果
    report_path = os.path.join(target_dir, "png_file_counts.txt")
    with open(report_path, "w") as report_file:
        # 遍歷所有子資料夾
        for subdir, dirs, files in os.walk(source_dir):
            png_count = sum(1 for file in files if file.endswith(".png"))
            # 寫入子資料夾和對應的 PNG 檔案數量
            report_file.write(f"{subdir}: {png_count} PNG files\n")
            
            # 如果 PNG 檔案數量超過指定數量，則移動整個子資料夾
            if png_count > min_count:
                # 創建相同的子資料夾路徑於目標資料夾
                new_subdir = os.path.join(target_dir, os.path.basename(subdir))
                if not os.path.exists(new_subdir):
                    shutil.move(subdir, new_subdir)
                else:
                    # 如果目標已存在，避免覆寫問題
                    print(f"Directory already exists: {new_subdir}")

# 使用範例
source_directory = r'D:\NCUE_lab\Data\Classificaiton\Students_L_face'
target_directory = r'D:\NCUE_lab\Data\Classificaiton\Students_L_training'
count_and_move_png_files(source_directory, target_directory)
