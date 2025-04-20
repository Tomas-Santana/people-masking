import torch
from torchvision import transforms as TL
from PIL import Image
import matplotlib.pyplot as plt
from src.models.UNET import UNET
import src.config as config

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

def visualize_result(image_path, mask):
    original_image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(mask.squeeze(0).cpu().numpy(), cmap="gray")
    plt.show()

def main():
    # Path to the image and model checkpoint
    image_path = "dataset/images/ds1_bow-tie-businessman-fashion-man.png"  # Replace with your image path
    checkpoint_path = "checkpoints/model.pth"  # Replace with your checkpoint path

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)

    # Preprocess the image
    input_size = config.INPUT_IMAGE_SIZE  # Ensure this matches your training configuration
    image = preprocess_image(image_path, input_size).to(device)

    # Generate mask
    with torch.no_grad():
        predicted_mask = model(image)
        predicted_mask = (predicted_mask > 0.5).float()  # Binarize the mask

    # Visualize the result
    visualize_result(image_path, predicted_mask)

if __name__ == "__main__":
    main()