import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns   
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Compose
from transformers import Swinv2Model, Swinv2Config
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os



# Define paths
test_images_dir = r"D:\NCUE_lab\Data\Segment\512x512\Test\Images"
test_masks_dir = r"D:\NCUE_lab\Data\Segment\512x512\Test\Labels"
model_save_path = r"D:\NCUE_lab\Exminationroom_dection\Swin_V2_seg\swinv2_segmentation_model_complex_512.pth"

# Define the RGB to class mapping
class_map = {
    (192, 64, 0): 0,  # Human
    (64, 192, 0): 1,  # Background
    (0, 64, 192): 2   # Hands
}

# Inverse mapping
inverse_class_map = {v: k for k, v in class_map.items()}

# Define the Segmentation Dataset
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        
        # Convert RGB mask to class indices
        mask = np.array(mask)
        class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        for rgb, index in class_map.items():
            class_mask[(mask == rgb).all(axis=-1)] = index

        mask = torch.from_numpy(class_mask)
        if self.transform:
            image = self.transform(image)

        return image, mask

# Define the transformation
transform = Compose([ToTensor()])

# Create the test dataset and dataloader
test_dataset = SegmentationDataset(test_images_dir, test_masks_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Define the segmentation model
class SwinV2Segmentation(nn.Module):
    def __init__(self, num_classes):
        super(SwinV2Segmentation, self).__init__()
        config = Swinv2Config(image_size=512, patch_size=4)  #setting image input size
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

# Function to predict and generate confusion matrix
def evaluate_model(model, test_loader, num_classes):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy().flatten())
            all_labels.append(masks.cpu().numpy().flatten())

    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    
    # Compute accuracy
    accuracy = np.mean(all_preds == all_labels)
    
    return cm, accuracy

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names, accuracy):
    cm_prob = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_prob, annot=True, fmt='.2f', cmap=plt.cm.Blues, xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'Confusion Matrix as Probabilities\nAccuracy: {accuracy:.2%}')
    plt.show()

# Evaluate the model and compute confusion matrix
cm, accuracy = evaluate_model(model, test_loader, num_classes)

# Define class names for plotting
class_names = ['Human', 'Background', 'Hands']

# Plot the confusion matrix
plot_confusion_matrix(cm, class_names, accuracy)
