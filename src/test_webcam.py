import torch
from torchvision import transforms as TL
from PIL import Image
from src.models.UNET import UNET
import src.config as config
import cv2
import numpy as np

def load_model(checkpoint_path, device):
    model = UNET().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def preprocess_frame(frame, input_size):
    transformations = TL.Compose([
        TL.ToPILImage(),
        TL.Resize(input_size),
        TL.ToTensor()
    ])
    image = transformations(frame)
    return image.unsqueeze(0)  # Add batch dimension

def main():
    # Open webcam
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Could not open webcam.")
        return

    # Load model
    checkpoint_path = "checkpoints/model.pth" 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)

    # Input size for preprocessing
    input_size = config.INPUT_IMAGE_SIZE  # Ensure this matches your training configuration

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Preprocess the frame
        image = preprocess_frame(frame, input_size).to(device)

        # Predict the mask
        with torch.no_grad():
            predicted_mask = model(image)
            predicted_mask = (predicted_mask > 0.5).float()  # Binarize the mask

            # Resize the mask to match the original frame size
            target_size = (frame.shape[1], frame.shape[0])  # (width, height)
            predicted_mask = torch.nn.functional.interpolate(predicted_mask, size=target_size, mode="bilinear", align_corners=False)

        # Convert the mask to a format suitable for OpenCV
        mask = predicted_mask.squeeze().cpu().numpy()  # Remove batch and channel dimensions
        mask = (mask * 255).astype(np.uint8)  # Scale to 0-255
        mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)  # Apply a colormap for better visualization

        # Resize the mask_colored to match the frame dimensions
        mask_colored = cv2.resize(mask_colored, (frame.shape[1], frame.shape[0]))  # Resize to (width, height)

        # Resize the binary mask to match the frame dimensions
        binary_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))  # Resize to (width, height)
        binary_mask = binary_mask // 255  # Convert mask to binary (0 or 1)
        # add gaussian noise to the binary mask
        noise = np.random.normal(0, 0.1, binary_mask.shape).astype(np.uint8)
        binary_mask = cv2.add(binary_mask, noise)
        binary_mask = np.clip(binary_mask, 0, 1)
        binary_mask = binary_mask.astype(np.uint8)
        

        # Create a masked version of the frame
        masked_frame = frame * binary_mask[:, :, np.newaxis]  # Apply the mask to the frame

        # Combine the original frame, the mask, and the masked frame side by side
        combined = np.hstack((mask_colored, masked_frame))

        # Display the combined frame
        cv2.imshow("Webcam, Predicted Mask, and Masked Frame", combined)


        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()