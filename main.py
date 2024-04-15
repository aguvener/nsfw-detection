import os
import shutil
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoModelForImageClassification, ViTImageProcessor
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Initialize model and device
model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
processor = ViTImageProcessor.from_pretrained('Falconsai/nsfw_image_detection')
model.eval()
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)

# Define the directories
base_directory = '/Users/aguvener/Desktop/imgtag/photo-similarity-search/photos'
nsfw_directory = 'nsfw_directory'
error_directory = 'error_directory'
os.makedirs(nsfw_directory, exist_ok=True)
os.makedirs(error_directory, exist_ok=True)


# Function to process each image and copy if necessary
def process_image(image_path):
    try:
        img = Image.open(image_path)
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        predicted_label = logits.argmax(-1).item()
        label = model.config.id2label[predicted_label]
        if label.lower() == 'nsfw':
            shutil.copy(image_path, nsfw_directory)
        return image_path, label
    except:
        shutil.copy(image_path, error_directory)
        return image_path, 'error'

def process_images(image_paths):
    with ThreadPoolExecutor(max_workers=None) as executor:
        results = list(tqdm(executor.map(process_image, image_paths), total=len(image_paths)))
    return results

# Collect all image paths
image_files = [
    os.path.join(root, file)
    for root, _, files in os.walk(base_directory)
    for file in files
    if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))
]

results = process_images(image_files)