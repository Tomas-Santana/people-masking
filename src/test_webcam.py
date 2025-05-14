import cv2
import numpy as np
import torch
from torchvision import transforms as TL

import src.config as config
from src.models.UNET import UNET


def load_model(checkpoint_path, device):
    model = UNET().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def preprocess_frame(frame, input_size):
    """Tranform the frame to match the model input"""
    transformations = TL.Compose([
        TL.ToPILImage(),
        TL.Resize(input_size),
        TL.ToTensor(),
    ])
    image = transformations(frame)
    return image.unsqueeze(0) 

def main(checkpoint_path):
    # Open webcam
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_EXPOSURE, -2)
    if not webcam.isOpened():
        print("Error: Could not open webcam.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)

    input_size = config.INPUT_IMAGE_SIZE 
    blur = True
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        image = preprocess_frame(frame, input_size).to(device)

        with torch.no_grad():
            predicted_mask = model(image)
            predicted_mask = (predicted_mask > 0.5).float()  

             
        mask = predicted_mask.squeeze().cpu().numpy() 

        # Remove salt and pepper noise 
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Resize the mask to match the frame size
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        # Add dilation to the mask to pad the edges
        mask = cv2.dilate(mask, kernel, iterations=1)

        # add gaussian blur to the mask to hide pixelation
        mask = cv2.GaussianBlur(mask, (31, 31), 0) 
        
        if blur:
            bg = frame * (1 - mask[:, :, None])
            bg = cv2.GaussianBlur(bg, ksize=(31, 31), sigmaX=0)

        else:
            # Use a background image
            bg = cv2.imread("assets/bg.jpg")
            bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))
            bg = bg * (1 - mask[:, :, None])

        # Apply the mask to the frame
        frame = frame * mask[:, :, None]
        frame += bg
        frame[frame > 255] = 255

        cv2.imshow('Webcam', np.uint8(frame))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('b'):
            blur = not blur

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    CHECKPOINT_PATH = "weights/model_v3.pth"  # <- Replace with your checkpoint path
    main(checkpoint_path=CHECKPOINT_PATH)