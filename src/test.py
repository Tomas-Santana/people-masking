import random

import cv2
import numpy as np
import torch
from PIL import Image
from imutils import paths
from torchvision import transforms as TL
import src.config as config
from src.models.UNET import UNET

# Loads a model from a saved checkpoint
def load_model(checkpoint_path, device):
    model = UNET().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

# converts images to RGB, resizes, then converts to a tensor
def preprocess_image(image_path, input_size):
    transformations = TL.Compose([
        TL.Resize(input_size),
        TL.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    image = transformations(image)
    return image.unsqueeze(0)  

# Creates a visual for the result of the model with the actual image, actual mask, and predicted mask with postprocessing applied
def visualize_result(image_path, mask):
    cv2.namedWindow("Actual image")
    non_processed_image_path = image_path.replace("preprocessed_images", "images")
    non_processed_mask_path = image_path.replace("preprocessed_images", "masks")
    non_processed_image = cv2.imread(non_processed_image_path)
    
    actual_dimensions = (non_processed_image.shape[1], non_processed_image.shape[0])  # (width, height)
    
    mask = mask.squeeze().cpu().numpy()  
    mask = (mask * 255).astype(np.uint8)

    cv2.imshow("Actual image", non_processed_image)
    cv2.imshow("Actual Mask", cv2.imread(non_processed_mask_path, cv2.IMREAD_GRAYSCALE))

    # add gaussian blur to the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # remove salt and pepper noise
    processed_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel)

    processed_mask = cv2.resize(processed_mask, actual_dimensions)

    processed_mask = cv2.dilate(processed_mask, kernel, iterations=1)
    # add gaussian blur to the mask to hide pixelation    
    processed_mask = cv2.GaussianBlur(processed_mask, (31, 31), 0)
    cv2.imshow("Processed Mask", processed_mask)


def main(checkpoint_path):
    # Path to the image and model checkpoint
    images_dir = config.PREPROCESSED_TRAIN_IMAGES_DIR  # <- Change this to your actual directory
    images_path = list(paths.list_images(images_dir))

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)

    # Preprocess the image
    input_size = config.INPUT_IMAGE_SIZE  # Ensure this matches your training configuration

    while True:
        image_path = random.choice(images_path)  
        image = preprocess_image(image_path, input_size).to(device)
        
        # Predict the mask
        with torch.no_grad():
            predicted_mask = model(image)
            predicted_mask = (predicted_mask > 0.5).float()  # Binarize the mask

        visualize_result(image_path, predicted_mask)
        if cv2.waitKey(0) & 0xFF == ord('n'):
            continue    
        elif cv2.waitKey(0) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    CHECKPOINT_PATH = "weights/model_v3.pth"  # <- Replace with your model path
    main(CHECKPOINT_PATH)