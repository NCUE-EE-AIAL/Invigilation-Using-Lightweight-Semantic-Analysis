import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def find_pixels_by_color(image_path, target_color):
    # 加載圖像
    image = Image.open(image_path)
    image_np = np.array(image)
    
    # 忽略Alpha通道，僅比較RGB通道
    if image_np.shape[-1] == 4:  # 如果圖像有Alpha通道
        image_np = image_np[:, :, :3]  # 去掉Alpha通道

    # 找到所有符合顏色條件的像素位置
    color_mask = np.all(image_np == target_color, axis=-1)
    y_coords, x_coords = np.where(color_mask)
    
    # 將結果存儲在字典中
    x_to_y = {}
    y_to_x = {}

    for x, y in zip(x_coords, y_coords):
        if x not in x_to_y:
            x_to_y[x] = []
        x_to_y[x].append(y)

        if y not in y_to_x:
            y_to_x[y] = []
        y_to_x[y].append(x)

    return x_to_y, y_to_x, x_coords, y_coords

def visualize_pixels(image_path, x_to_y, target_color, min_y_pixel=None, edges=None):
    # 加載圖像以獲取尺寸
    image = Image.open(image_path)
    width, height = image.size

    # 創建新圖像
    new_image_np = np.zeros((height, width, 3), dtype=np.uint8)  # 默認為黑色

    # 將目標顏色應用到提及的像素點
    for x, y_list in x_to_y.items():
        for y in y_list:
            new_image_np[y, x] = target_color

    if min_y_pixel is not None:
        new_image_np[min_y_pixel[1], :] = [255, 0, 0]  # 標記水平線
        new_image_np[:, min_y_pixel[0]] = [255, 0, 0]  # 標記垂直線

    if edges is not None:
        new_image_np[edges == 255] = [0, 255, 0]  # 用綠色標記邊界像素

    # 顯示新圖像
    plt.imshow(new_image_np)
    plt.axis('on')  # 顯示坐標軸
    plt.show()

def mark_min_y_pixel(x_coords, y_coords):
    if len(y_coords) == 0:  # 如果沒有找到匹配的像素
        return None

    # 找到 y 值最小的像素點
    min_y_index = np.argmin(y_coords)
    min_y_pixel = (x_coords[min_y_index], y_coords[min_y_index])

    return min_y_pixel

def find_color_edges(image_np, target_color):
    
    if image_np.shape[-1] == 4:  # 如果圖像有Alpha通道
        image_np = image_np[:, :, :3]  # 去掉Alpha通道

    mask = np.all(image_np == target_color, axis=-1).astype(np.uint8) * 255

    # 使用Canny邊緣檢測算法找到邊界
    edges = cv2.Canny(mask, 100, 200)

    return edges

# 使用示例
image_path = r'D:\NCUE_lab\sementic_segmentation\CamVid\labels_student\S1252001V1_0001.png'
target_color = (192, 64, 0)  # 替換為你指定的顏色 (R, G, B)

image = Image.open(image_path)
image_np = np.array(image)

x_to_y, y_to_x, x_coords, y_coords = find_pixels_by_color(image_path, target_color)
min_y_pixel = mark_min_y_pixel(x_coords, y_coords)
edges = find_color_edges(image_np, target_color)

# 可視化結果
visualize_pixels(image_path, x_to_y, target_color, min_y_pixel, edges)
print(f"Min y pixel location: {min_y_pixel}")
