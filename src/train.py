import random
import cv2
import numpy as np
from src.data import PeopleMaskingDataset
from src.models.UNET import UNET
import src.config as config
import src.params as params
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import torch
import time
from torch.optim.lr_scheduler import StepLR


def main(checkpoint_path=None):

    masks: list[str] = list(paths.list_images(config.PREPROCESSED_TRAIN_MASKS_DIR))
    images = [str(p).replace("masks", "images") for p in masks]
    # random sample 25% of the images and masks
    # images, _, masks, _ = train_test_split(images, masks, test_size=0.75, random_state=42)
    
    
    

    image_transforms = transforms.Compose([
        transforms.ToPILImage(), 
        # transforms.Resize(config.INPUT_IMAGE_SIZE), 
        # transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0], std=[1]),
        # transforms.GaussianBlur(kernel_size=5),
        # Augmentation
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.6, 1))
        # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    ])
    train_dataset = PeopleMaskingDataset(images, masks, image_transforms, read_mode=cv2.IMREAD_COLOR_BGR)

    train_loader = DataLoader(train_dataset, batch_size=params.BATCH_SIZE, shuffle=True, num_workers=params.NUM_WORKERS)


    unet = UNET().to(config.DEVICE)
    if checkpoint_path:
        unet.load_state_dict(torch.load(checkpoint_path))
    loss_fn = BCEWithLogitsLoss(reduction="sum")
    optimizer = Adam(unet.parameters(), lr=params.LR, weight_decay=1e-5)
    cv2.namedWindow("Predicted / Actual")
    train_steps = len(train_loader)
    unet.train()
    print(f"Number of training steps: {train_steps}")

    print("Starting training...")
    start_time = time.time()
    for epoch in range(params.NUM_EPOCHS):

        epoch_loss = 0
        epoch_accuracy = 0
        for _, (images, masks) in enumerate(tqdm(train_loader)):
            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE)

            # Prediction and loss calculation
            predictions = unet(images)
            loss = loss_fn(predictions, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            epoch_accuracy += ((predictions > 0.5) == masks).float().mean().item()
            
            # display
            image, mask = train_dataset.__getitem__(random.randint(0, len(train_dataset) - 1))
            
            with torch.no_grad():
                unet.eval()
                out = (unet(image.view(1, 3, *config.INPUT_IMAGE_SIZE).to(config.DEVICE)).view(*config.INPUT_IMAGE_SIZE).cpu().numpy() > 0) * 1.0
                unet.train()
            original = image.permute(1, 2, 0).cpu().numpy()  # Convert from [C, H, W] to [H, W, C]
            original = np.uint8(255 * original)
            mask = mask.view(*config.INPUT_IMAGE_SIZE).cpu().numpy()
            out = np.concatenate((out, mask), axis=1)
            cv2.imshow("Predicted / Actual", np.uint8(255 * out))
            cv2.imshow("Original", np.uint8(original))
            cv2.waitKey(1)


        epoch_loss = epoch_loss / train_steps * params.BATCH_SIZE


        print(f"Epoch {epoch + 1}/{params.NUM_EPOCHS} - Train loss: {epoch_loss:.4f}")
        
        torch.save(unet.state_dict(), config.MODEL_PATH)
        print(f"Model saved to {config.MODEL_PATH}")

        print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    print("Training finished!")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main("checkpoints/model_v3_selfies.pth")

            

        