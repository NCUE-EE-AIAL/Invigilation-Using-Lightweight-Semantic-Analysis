from facenet_pytorch import MTCNN
import torch
from PIL import Image
import os

# Setup the device and MTCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

def crop_faces(input_folder, output_folder, mtcnn):
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Process each file in the input folder
    files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    for file in files:
        img_path = os.path.join(input_folder, file)
        img = Image.open(img_path).convert('RGB')
        # Detect faces
        boxes, _ = mtcnn.detect(img)
        if boxes is not None:
            for i, box in enumerate(boxes):
                cropped_img = img.crop(box)
                cropped_img.save(os.path.join(output_folder, f"{os.path.splitext(file)[0]}_face_{i}.png"))

def process_subdirectories(base_input_folder, base_output_folder, mtcnn):
    # Get all subdirectories in the base input folder
    subdirs = [d for d in os.listdir(base_input_folder) if os.path.isdir(os.path.join(base_input_folder, d))]
    
    # Process each subdirectory
    for subdir in subdirs:
        input_subdir = os.path.join(base_input_folder, subdir)
        output_subdir = os.path.join(base_output_folder, subdir)
        
        # Ensure the output subdirectory exists
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        # Crop faces for each image in the subdirectory
        crop_faces(input_subdir, output_subdir, mtcnn)

# Example usage
base_input_folder = r'D:\NCUE_lab\Data\Classificaiton\Students_Picture'
base_output_folder = r'D:\NCUE_lab\Data\Classificaiton\Students_L_training'
process_subdirectories(base_input_folder, base_output_folder, mtcnn)
