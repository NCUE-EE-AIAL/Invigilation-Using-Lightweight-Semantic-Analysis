from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

# Define the directories for the original images and labeled masks using relative paths
Ori_image = r"D:\NCUE_lab\sementic_segmentation\CamVid\images"
Label_image = r"D:\NCUE_lab\sementic_segmentation\CamVid\images_hand"

# Map RGB values to class names
class_map = {
    (192, 64, 0): "Human",
    (64, 192, 0): "Background",
    (0, 64, 192): "Hands"
}
transform = transforms.Compose([
    transforms.Resize((384, 384)),  # Resize images to 384x384 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

def calculate_class_distribution_gpu(mask_paths, class_map, device):
    # Convert RGB keys to a tensor and transfer to GPU
    keys = torch.tensor(list(class_map.keys()), dtype=torch.uint8, device=device)
    
    # Initialize a counter for each class
    total_counts = torch.zeros(len(class_map), device=device)
    
    for mask_path in mask_paths:
        mask = Image.open(mask_path).convert('RGB')  # Ensure you're opening a file, not a directory
        mask = np.array(mask, dtype=np.uint8)
        mask = torch.from_numpy(mask).to(device)
        
        mask = mask.unsqueeze(0)
        keys = keys.view(-1, 1, 1, 3)
        
        matches = (mask == keys).all(-1)
        total_counts += matches.sum([1, 2]).float()
    
    total_counts = total_counts / total_counts.sum()
    return total_counts.cpu().numpy()

def show_image_with_labels(image_tensor, mask_tensor, class_map):
    # Convert the image tensor to a PIL Image to use existing transformation functions
    image = TF.to_pil_image(image_tensor)
    mask = TF.to_pil_image(mask_tensor)
    
    # Set up plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show the image
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Show the mask
    ax[1].imshow(mask)
    ax[1].set_title('Mask Image')
    ax[1].axis('off')

    # Create a legend for the classes
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=np.array(color)/255.0, label=label) for color, label in class_map.items()]
    ax[1].legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # List only jpg files to avoid any confusion with other file types.
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.mask = [f for f in os.listdir(mask_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get the path for the original image
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.mask[idx])

        # Construct the mask path by changing the original image filename
        base_filename = os.path.splitext(f"{self.images[idx]}.jpg")[0]  # Removes the '.jpg', e.g., '000001'
        mask_filename = os.path.splitext(f"{self.mask[idx]}.png")[0]  # Constructs mask filename, e.g., 'label_000001.png'
        mask_path = os.path.join(self.mask_dir, mask_filename)

        # Debugging print statements to check file paths
        print(f"Image path: {img_path}")
        print(f"Mask path: {mask_path}")
        print(mask_filename)
        print(base_filename)

        # Load the image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        
        # Apply transformations, if any
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


    def convert_mask(self, mask):
        mask = np.array(mask)
        class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k, v in class_map.items():
            class_mask[np.all(mask == np.array(k, dtype=np.uint8), axis=-1)] = v
        return Image.fromarray(class_mask, 'L')


dataset = SegmentationDataset(Ori_image, Label_image, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Uncomment these lines after ensuring the dataset loads correctly:
mask_paths = [os.path.join(Label_image, fname) for fname in os.listdir(Label_image)]
distribution = calculate_class_distribution_gpu(mask_paths, class_map, device)
labels = list(class_map.values())
plt.bar(labels, distribution)
plt.xlabel('Classes')
plt.ylabel('Proportion')
plt.title('Class Distribution in Dataset')
plt.show()

#To display an image and mask, ensure this part is properly used:
sample_image, sample_mask = dataset[1]  # Get the first sample from the dataset
show_image_with_labels(sample_image, sample_mask, class_map)
