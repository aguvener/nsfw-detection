"""
This script is used to process images in a directory and classify them as NSFW or SFW.
The script uses a pre-trained model to classify images and copy them to the output directory if they are NSFW.
If an error occurs while processing an image, the image is copied to the error directory.

Usage: python main.py -I <input_directory> -O <output_directory> -E <error_directory>

Arguments:
    -I, --input: Input directory containing images to be processed.
    -O, --output: Output directory to copy NSFW images.
    -E, --error: Error directory to copy images that could not be processed.
"""

import os
import shutil
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse

# Initialize model and device
model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
processor = ViTImageProcessor.from_pretrained('Falconsai/nsfw_image_detection')
model.eval()
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)

# Get directories from command line arguments
# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-I', '--input', help='Input directory', required=True)
parser.add_argument('-O', '--output', help='Output directory')
parser.add_argument('-E', '--error', help='Error directory')
args = parser.parse_args()

# Set default values if directories are not specified
output_directory = args.output if args.output else 'output'
nsfw_directory = f'{output_directory}/nsfw'
error_directory = args.error if args.error else f'{output_directory}/error'

# Create directories if they don't exist
os.makedirs(output_directory, exist_ok=True)
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
    for root, _, files in os.walk(args.input)
    for file in files
    if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))
]

results = process_images(image_files)