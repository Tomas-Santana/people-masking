import torch
from src.models.UNET import UNET
import torchvision
import cv2


checkpont = torch.load("checkpoints/model.pth")
model = UNET()
model.load_state_dict(checkpont)

sample_image_path = "dataset/images/ds1_bow-tie-businessman-fashion-man.png"

image = torchvision.io.read_image(sample_image_path).unsqueeze(0).float() / 255.0
image = torchvision.transforms.functional.resize(image, (128, 128))
grayscale_image = torchvision.transforms.functional.rgb_to_grayscale(image, num_output_channels=1)
image = grayscale_image.unsqueeze(0)

image = image.to("cpu")

model.eval()

with torch.no_grad():
    output = model(image)
    output = torch.sigmoid(output)
    output = (output > 0.5).float()
    output_image = output.squeeze().cpu().numpy()

    output_image = (output_image * 255).astype("uint8")
    output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Output", output_image)
    cv2.waitKey(0)
