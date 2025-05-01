import random

import cv2
from numpy.f2py.auxfuncs import throw_error

from src.data import PeopleMaskingDataset
from src.models.UNET import UNET
import src.config as config
import src.params as params
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import torch
import time
from torch.optim.lr_scheduler import StepLR
import numpy as np


def main(checkpoint_path=None):

    masks: list[str] = list(paths.list_images(config.PREPROCESSED_MASKS_DIR))
    images = list(paths.list_images(config.PREPROCESSED_IMAGES_DIR))
    if (len(masks) != len(images)):
        throw_error(f"Number of masks and images do not match")

    image_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    ])
    dataset = PeopleMaskingDataset(images, masks, image_transforms)
    loader = DataLoader(dataset, batch_size=params.BATCH_SIZE, shuffle=True, num_workers=params.NUM_WORKERS)

    unet = UNET(in_channels=1, out_channels=1).to(config.DEVICE)
    if checkpoint_path:
        unet.load_state_dict(torch.load(checkpoint_path))
    loss_fn = BCEWithLogitsLoss()
    optimizer = Adam(unet.parameters(), lr=params.LR, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    steps = len(loader)
    print(f"Number of training steps: {steps}")

    # Loss Calculation
    running_loss = []
    print("Starting training...")
    start_time = time.time()
    for epoch in range(params.NUM_EPOCHS):
        unet.train()

        epoch_loss = 0
        for _, (images, masks) in enumerate(tqdm(loader)):

            # Prediction and loss calculation
            predictions = unet(images)
            loss = loss_fn(predictions, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # display
            image, mask = dataset.__getitem__(random.randint(0, len(dataset) - 1))

            with torch.no_grad():
                unet.eval()
                out = (unet(image.view(1, 1, *config.INPUT_IMAGE_SIZE).to(config.DEVICE)).view(
                    *config.INPUT_IMAGE_SIZE).cpu().numpy() > 0) * 1.0
                unet.train()
            mask = mask.view(*config.INPUT_IMAGE_SIZE).cpu().numpy()
            out = np.concatenate((out, mask), axis=1)
            cv2.imshow("Predicted / Actual", np.uint8(255 * out))
            cv2.waitKey(1)

        scheduler.step()
        running_loss.append(epoch_loss / steps)

        print(f"Epoch {epoch + 1}/{params.NUM_EPOCHS} - loss: {running_loss[-1]:.4f}")
        
        torch.save(unet.state_dict(), config.MODEL_PATH)
        print(f"Model saved to {config.MODEL_PATH}")

        print(f"Epoch completed in {time.time() - start_time:.2f} seconds")
    
    print("Training finished!")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

            

        