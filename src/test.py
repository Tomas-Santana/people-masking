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
    original_image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(mask.squeeze().cpu().numpy(), cmap="gray")  # Use squeeze() to remove all dimensions of size 1
    plt.axis("off")

    

    plt.show()

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
    images_dir = config.TRAIN_IMAGES_DIR  # Directory containing test images
    images_path = list(paths.list_images(images_dir))
    #load a random image from the directory
    
    checkpoint_path = "checkpoints/model.pth"  # Replace with your checkpoint path

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
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()