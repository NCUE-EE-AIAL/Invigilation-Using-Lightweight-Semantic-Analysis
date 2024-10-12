import cv2
import torch
import numpy as np
import torch.nn as nn
import tkinter as tk
from PIL import ImageTk
from torchvision.transforms import ToTensor
from PIL import Image
from transformers import Swinv2Model, Swinv2Config
import matplotlib.pyplot as plt

# Define the RGB to class mapping
class_map = {
    (192, 64, 0): 0,  # Human
    (64, 192, 0): 1,  # Background
    (0, 64, 192): 2   # Hands
}

model_weight_path = r'D:\NCUE_lab\Exminationroom_dection\Swin_V2_seg\swinv2_segmentation_model_5000_pre.pth'

# Inverse mapping
inverse_class_map = {v: k for k, v in class_map.items()}

# Define the segmentation model
class SwinV2Segmentation(nn.Module):
    def __init__(self, num_classes):
        super(SwinV2Segmentation, self).__init__()
        config = Swinv2Config()
        self.swin_v2 = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256", config=config)
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.segmentation_head = nn.Conv2d(config.hidden_size, num_classes, kernel_size=1)

    def forward(self, x):
        outputs = self.swin_v2(pixel_values=x, return_dict=True)
        last_hidden_state = outputs.last_hidden_state
        last_hidden_state = last_hidden_state.permute(0, 2, 1).contiguous()
        last_hidden_state = last_hidden_state.view(last_hidden_state.size(0), -1, int(last_hidden_state.size(-1)**0.5), int(last_hidden_state.size(-1)**0.5))
        upsampled_output = self.upsample(last_hidden_state)
        logits = self.segmentation_head(upsampled_output)
        return logits

# Define the number of segmentation classes
num_classes = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinV2Segmentation(num_classes).to(device)

# Load the trained model weights
model.load_state_dict(torch.load(model_weight_path))
model.eval()

def preprocess_image(image):
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = image.resize((256, 256))
    image = ToTensor()(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)

def postprocess_output(output, original_size):
    output = output.squeeze(0)  # Remove batch dimension
    output = torch.argmax(output, dim=0)  # Get the class with the highest score for each pixel
    output = output.cpu().numpy()
    height, width = output.shape
    segmented_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            segmented_image[i, j] = inverse_class_map[output[i, j]]
    
    # Resize the segmented image to the original size
    segmented_image = Image.fromarray(segmented_image)
    segmented_image = segmented_image.resize(original_size, Image.NEAREST)
    segmented_image = np.array(segmented_image)
    return segmented_image

# Open a connection to the camera
cap = cv2.VideoCapture(0)

# Create the main window
root = tk.Tk()
root.title("Real-time Segmentation")

# Create a label to display the images
label = tk.Label(root)
label.pack()

def update_frame():
    ret, frame = cap.read()
    if not ret:
        root.after(30, update_frame)
        return

    # Preprocess the frame
    input_image = preprocess_image(frame)

    # Run the model on the input image
    with torch.no_grad():
        output = model(input_image)

    # Postprocess the output to get the segmented image
    original_size = (frame.shape[1], frame.shape[0])
    segmented_image = postprocess_output(output, original_size)

    # Convert segmented image to BGR format for OpenCV
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)

    # Combine the original frame and the segmented frame
    combined_image = np.hstack((frame, segmented_image))

    # Convert the combined image to a format Tkinter can use
    combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(combined_image)
    imgtk = ImageTk.PhotoImage(image=img)

    # Update the label with the new image
    label.imgtk = imgtk
    label.configure(image=imgtk)

    # Schedule the next frame update
    root.after(30, update_frame)

# Start the frame update loop
update_frame()

# Run the Tkinter main loop
root.mainloop()

# Release the camera when done
cap.release()

