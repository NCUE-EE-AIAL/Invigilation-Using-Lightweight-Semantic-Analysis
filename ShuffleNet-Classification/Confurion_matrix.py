import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
from torchvision import models
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

def count_subdirectories(directory):
    """Count number of subdirectories in a given directory."""
    return len([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])

data_dir = r'D:\NCUE_lab\Data\Classificaiton\Students_L_training'
model_path = r'D:\NCUE_lab\Exminationroom_dection\Shuffelnet_detection\classification_13.pth'
test_path = r'D:\NCUE_lab\Data\Classificaiton\Students_L_testing'

## Load the model
# Define the model architecture (must match the architecture of the saved model)
# Initialize the model without pre-trained weights
model = shufflenet_v2_x1_0(weights=None)
num_classes = count_subdirectories(data_dir)  # Update this to the number of classes you have
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load the model weights

model.load_state_dict(torch.load(model_path))

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Define transformations for the test data
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root=test_path, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

## Generate Predictions and True Labels

all_preds = []
all_labels = []

model.eval()  # Ensure the model is in evaluation mode
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute the confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print(cm)

accuracy = np.trace(cm) / np.sum(cm)
print(f"Accuracy: {accuracy:.4f}")

# Function to plot the confusion matrix with accuracy
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          accuracy=None):  # Added accuracy parameter
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
    # Modify title to show accuracy on a new line below the main title
    plt.title(f'{title}\nAccuracy: {accuracy:.4f}', fontsize=16)  # Updated to show accuracy below the title
    
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# Get class names
class_names = test_dataset.classes
plot_confusion_matrix(cm, classes=class_names, title='Confusion Matrix', accuracy=accuracy)