import os
import torch
import torch.optim as optim
import copy
from torchinfo import summary
#from torchsummary import summary
from torchvision import datasets, transforms
from torchvision import models
from torchvision.utils import save_image
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights

data_dir = r'D:\NCUE_lab\Data\Classificaiton\Students_L_training'



# Data augmentation and normalization for training
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Define a function to count files and ensure balance
def count_files_in_subfolders(directory):
    counts = {}
    for subdir in os.listdir(directory):
        subpath = os.path.join(directory, subdir)
        if os.path.isdir(subpath):
            counts[subdir] = len([f for f in os.listdir(subpath) if os.path.isfile(os.path.join(subpath, f))])
    return counts

def save_test_images(dataloader, output_dir):
    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over the DataLoader
    for i, (inputs, _) in enumerate(dataloader, 1):
        # Save each image
        for j, input in enumerate(inputs):
            # Filename for each image
            filename = os.path.join(output_dir, f'image_{i}_{j}.jpg')
            # Undo normalization and save the image
            save_image(input, filename)

# Load your dataset
def load_datasets(data_dir):
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms['train'])
    class_names = dataset.classes
    num_classes = len(class_names)
    counts = count_files_in_subfolders(data_dir)
    print(f"Class counts: {counts}")
    
    # Splitting the dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Apply different transforms for validation dataset
    val_dataset.dataset.transform = data_transforms['val']

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, num_classes

def count_subdirectories(directory):
    """Count number of subdirectories in a given directory."""
    return len([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])


def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, patience=5):
    best_acc = 0.0
    no_improve_epochs = 0  # 紀錄沒有改善的迭代次數

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    no_improve_epochs = 0  # 重置沒有改善的迭代次數
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= patience:  # 如果達到耐心限度
                        print(f'Early stopping after {epoch+1} epochs due to no improvement.')
                        model.load_state_dict(best_model_wts)
                        return model
        
    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model


# Example directory

train_loader, val_loader, num_classes = load_datasets(data_dir)

# After setting up DataLoaders
dataloaders = {
    'train': train_loader,
    'val': val_loader
}
# Load pre-trained ShuffleNet v2 model
model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)

# Assuming you have calculated the number of classes
num_classes = count_subdirectories(data_dir)  # Update this with your actual number of classes

# Modify the final fully connected layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.train()  # Set the model to training mode

print("Model modified and set to training mode.")

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Loss function and optimizer set up.")


#model = train_model(model, dataloaders, criterion, optimizer, num_epochs=20, patience=7)
#model_save_path = r'D:\NCUE_lab\Exminationroom_dection\Shuffelnet_detection\classification_13.pth'
#torch.save(model.state_dict(), model_save_path)

summary(model, input_size=(1, 3, 224, 224))
#print(model)