import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import os
from transformers import Swinv2Model, Swinv2Config

# Define the directories for the test image and model path
test_image_path = r"D:\NCUE_lab\sementic_segmentation\CamVid\images_student\S1252001V1_0002.png"
model_save_path = 'swinv2_segmentation_model_5000_pre.pth'

# Define the RGB to class mapping
class_map = {
    (192, 64, 0): 0,  # Human
    (64, 192, 0): 1,  # Background
    (0, 64, 192): 2   # Hands
}
# Inverse mapping
inverse_class_map = {v: k for k, v in class_map.items()}

# Define the model architecture
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
model.load_state_dict(torch.load(model_save_path))
model.eval()

# Preprocess the test image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = ToTensor()(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Postprocess the model output to get the segmented image
def postprocess_output(output):
    output = output.squeeze(0)  # Remove batch dimension
    output = torch.argmax(output, dim=0)  # Get the class with the highest score for each pixel
    output = output.cpu().numpy()
    height, width = output.shape
    segmented_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            segmented_image[i, j] = inverse_class_map[output[i, j]]
    return segmented_image

# Load and preprocess the test image
input_image = preprocess_image(test_image_path).to(device)

# Run the model on the input image
with torch.no_grad():
    output = model(input_image)

# Postprocess the output to get the segmented image
segmented_image = postprocess_output(output)

# Save or display the segmented image
segmented_image_pil = Image.fromarray(segmented_image)
segmented_image_pil.save(r"path_to_save_segmented_image.png")
segmented_image_pil.show()  # You can also display the image using a suitable viewer
