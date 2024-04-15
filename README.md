# NSFW Image Detection

This script identifies and classifies images as NSFW (Not Safe For Work) or SFW (Safe For Work) using a pre-trained model. It processes images from a specified directory and moves NSFW images to an output directory. If an error occurs, the image is copied to an error directory.

## Model
The script employs the [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection) model.

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/aguvener/nsfwdetection
cd nsfwdetection
pip install -r requirements.txt
```

## Usage
Run the script using the following command:

```bash
python main.py -I <input_directory> -O <output_directory> -E <error_directory>
```

### Parameters
- `-I, --input`: Directory with images for processing.
- `-O, --output`: Directory to store NSFW images.
- `-E, --error`: Directory to store images that could not be processed due to errors.