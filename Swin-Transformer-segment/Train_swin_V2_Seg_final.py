import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Compose
from transformers import Swinv2Model, Swinv2Config
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy


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

# Transformation
transform = Compose([ToTensor()])  # Only conversion to tensor is needed

# Define the directories for train, validation, and test datasets
train_images_dir = r"D:\NCUE_lab\Data\Segment\512x512\Train\Images"
train_masks_dir = r"D:\NCUE_lab\Data\Segment\512x512\Train\Labels"
val_images_dir = r"D:\NCUE_lab\Data\Segment\512x512\Val\Images"
val_masks_dir = r"D:\NCUE_lab\Data\Segment\512x512\Val\Labels"
test_images_dir = r"D:\NCUE_lab\Data\Segment\512x512\Test\Images"
test_masks_dir = r"D:\NCUE_lab\Data\Segment\512x512\Test\Labels"

# Create datasets and dataloaders
train_dataset = SegmentationDataset(train_images_dir, train_masks_dir, transform=transform)
val_dataset = SegmentationDataset(val_images_dir, val_masks_dir, transform=transform)
test_dataset = SegmentationDataset(test_images_dir, test_masks_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
model = SwinV2Segmentation(num_classes).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Adjust the learning rate

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, patience=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss}')

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks.long())
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss}')

        # Step the scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{num_epochs}], Learning Rate: {current_lr}')

        # Check for early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print('Early stopping triggered')
                break

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Function to save the trained model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20)
model_save_path = 'swinv2_segmentation_model_5000_pre.pth'
save_model(model, model_save_path)
