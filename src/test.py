import numpy as np
import torch
from torchvision import transforms as TL
from PIL import Image
import matplotlib.pyplot as plt
from src.models.UNET import UNET
import src.config as config
# work with paths
from imutils import paths
import random
import torch.nn.functional as F
import cv2

def load_model(checkpoint_path, device):
    model = UNET().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path, input_size):
    transformations = TL.Compose([
        TL.Resize(input_size),
        TL.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    image = transformations(image)
    return image.unsqueeze(0)  # Add batch dimension

def preprocess_frame(frame, input_size):
    transformations = TL.Compose([
        TL.ToPILImage(),
        TL.Resize(input_size),
        TL.ToTensor()
    ])
    image = transformations(frame)
    return image.unsqueeze(0)  # Add batch dimension

def visualize_result(image_path, mask):
    cv2.namedWindow("actual image")
    non_processed_image_path = image_path.replace("preprocessed_images", "images")
    non_processed_mask_path = image_path.replace("preprocessed_images", "masks")
    non_processed_image = cv2.imread(non_processed_image_path)
    print(non_processed_image.shape)
    actual_dimsions = (non_processed_image.shape[1], non_processed_image.shape[0])  # (width, height)
    
    mask = mask.squeeze().cpu().numpy()  
    mask = (mask * 255).astype(np.uint8)
    resized_mask = cv2.resize(mask, actual_dimsions)

    cv2.imshow("actual image", non_processed_image)
    cv2.imshow("Predicted Mask", resized_mask)
    cv2.imshow("Actual Mask", cv2.imread(non_processed_mask_path, cv2.IMREAD_GRAYSCALE))

    # add gaussian blur to the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # remov salt and pepper noise
    processed_mask = cv2.morphologyEx(resized_mask, cv2.MORPH_OPEN, kernel)
    processed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    processed_mask = cv2.GaussianBlur(mask, (5, 5), 0)
    processed_mask = cv2.resize(processed_mask, actual_dimsions)
    cv2.imshow("Processed Mask", processed_mask)

    


def visualize_webcam_result(frame, mask):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Frame")
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(mask.squeeze().cpu().numpy(), cmap="gray")  # Use squeeze() to remove all dimensions of size 1
    plt.axis("off")
    plt.show()
    plt.pause(0.001)  # Pause to update the plot

def main():
    # Path to the image and model checkpoint
    images_dir = config.PREPROCESSED_TRAIN_IMAGES_DIR  # Directory containing test images
    images_path = list(paths.list_images(images_dir))
    #load a random image from the directory
    
    checkpoint_path = "checkpoints/model_v3_selfies+og_doubleconv.pth"  # Replace with your checkpoint path

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
            original_image = Image.open(image_path).convert("RGB")

            target_size = original_image.size[::-1]  # (width, height)

            
            predicted_mask = torch.nn.functional.interpolate(predicted_mask, size=target_size, mode="bilinear", align_corners=False)

        visualize_result(image_path, predicted_mask)
        if cv2.waitKey(0) & 0xFF == ord('n'):
            continue    
        elif cv2.waitKey(0) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()