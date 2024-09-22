import cv2
import os
from mtcnn import MTCNN
import torch
from torchvision import models, transforms
from torchvision.models import shufflenet_v2_x1_0
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 初始化MTCNN用于人脸检测
detector = MTCNN()

# 加载ShuffleNet模型
def load_model(model_path, num_classes):
    model = shufflenet_v2_x1_0(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_class_names(data_dir):
    """ 返回指定目录中所有子目录的名称，这些子目录表示不同的类别 """
    return [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# 指定你的类别目录
data_dir = r'D:\NCUE_lab\Data\Classificaiton\Students_L_training'
class_names = get_class_names(data_dir)
num_classes = len(class_names)  # 动态获取类别数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(r'D:\NCUE_lab\Exminationroom_dection\Shuffelnet_detection\classification_13.pth', num_classes).to(device)

cap = cv2.VideoCapture(0)

def update(frame_number):
    ret, frame = cap.read()
    if not ret:
        return

    # 使用MTCNN检测人脸
    faces = detector.detect_faces(frame)
    for face in faces:
        x, y, width, height = face['box']
        face_img = frame[y:y+height, x:x+width]
        face_img = cv2.resize(face_img, (224, 224))
        face_tensor = transforms.ToTensor()(face_img).unsqueeze(0).to(device)
        outputs = model(face_tensor)
        _, predicted = torch.max(outputs.data, 1)
        predicted_class = class_names[predicted.item()]  # 使用类别名称替代数字ID
        cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 0, 0), 2)
        cv2.putText(frame, f'ID: {predicted_class}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, interval=100)
plt.show()

cap.release()
