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
    if not webcam.isOpened():
        print("Error: Could not open webcam.")
        return

    checkpoint_path = "checkpoints/model_v3_selfies+og.pth" 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)

    input_size = config.INPUT_IMAGE_SIZE 

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        image = preprocess_frame(frame, input_size).to(device)

        with torch.no_grad():
            predicted_mask = model(image)
            predicted_mask = (predicted_mask > 0.5).float()  

            target_size = (frame.shape[1], frame.shape[0]) 
            predicted_mask = torch.nn.functional.interpolate(predicted_mask, size=target_size, mode="bilinear", align_corners=False, antialias=True)

        mask = predicted_mask.squeeze().cpu().numpy()  
        mask = (mask * 255).astype(np.uint8)  
        mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)  

        mask_colored = cv2.resize(mask_colored, (frame.shape[1], frame.shape[0]))  

        binary_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))  
        binary_mask = binary_mask // 255  

        noise = np.random.normal(0, 0.1, binary_mask.shape).astype(np.uint8)
        binary_mask = cv2.add(binary_mask, noise)
        binary_mask = np.clip(binary_mask, 0, 1)
        binary_mask = binary_mask.astype(np.uint8)
        

        masked_frame = frame * binary_mask[:, :, np.newaxis]  

        combined = np.hstack((mask_colored, masked_frame))

        cv2.imshow("Webcam, Predicted Mask, and Masked Frame", combined)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()