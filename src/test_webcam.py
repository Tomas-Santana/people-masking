import torch
from torchvision import transforms as TL
from PIL import Image
from src.models.UNET import UNET
from src.models.FastSCNN import FastSCNN
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
        TL.ToTensor(),
        # TL.GaussianBlur(kernel_size=5),
    ])
    image = transformations(frame)
    return image.unsqueeze(0) 

def main():
    # Open webcam
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_EXPOSURE, -4)
    if not webcam.isOpened():
        print("Error: Could not open webcam.")
        return

    checkpoint_path = "checkpoints/model_v3_selfies+og_doubleconv.pth" 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)

    input_size = config.INPUT_IMAGE_SIZE 
    blur = True
    kernel_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_10 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
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
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_5)
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        mask = cv2.dilate(mask, kernel_10, iterations=1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0) 
        
        if blur:
            bg = frame * (1 - mask[:, :, None])
            bg = cv2.GaussianBlur(bg, ksize=(31, 31), sigmaX=0)

        else:
            # open bg/stout-bg
            bg = cv2.imread("assets/bg.jpg")
            bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))
            bg = bg * (1 - mask[:, :, None])

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
    main()