import random

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
from monai.losses import DiceLoss
import cv2 as cv
import numpy as np

class BCEDiceLoss(torch.nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = BCEWithLogitsLoss()
        self.dice = DiceLoss(sigmoid=True)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def main(checkpoint_path=None, display=True):
    image_paths = list(paths.list_images(config.IMAGES_DIR))
    mask_paths = list(paths.list_images(config.MASKS_DIR))
    images_train, images_test, masks_train, masks_test = train_test_split(
        image_paths, mask_paths, test_size=params.TRAIN_TEST_SPLIT, random_state=42
    )

    image_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(num_output_channels=1), transforms.Resize(config.INPUT_IMAGE_SIZE), transforms.ToTensor()])
    train_dataset = PeopleMaskingDataset(images_train, masks_train, image_transforms, augment=False)
    test_dataset = PeopleMaskingDataset(images_test, masks_test, image_transforms, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=params.BATCH_SIZE, shuffle=True, num_workers=params.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=params.BATCH_SIZE, shuffle=False, num_workers=params.NUM_WORKERS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path:
        unet = UNET(in_channels=1, out_channels=1).to(config.DEVICE)
        unet.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    else:
        unet = UNET(in_channels=1, out_channels=1).to(config.DEVICE)
    loss_fn = BCEWithLogitsLoss()
    optimizer = Adam(unet.parameters(), lr=params.LR)
    train_steps = len(train_loader)
    test_steps = len(test_loader)

    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    print("Starting training...")
    start_time = time.time()

    if display:
        cv.namedWindow("Predicted / Actual")

    for epoch in tqdm(range(params.NUM_EPOCHS)):
        unet.train()

        epoch_loss = 0
        epoch_accuracy = 0
        train_batch_count = 0
        epoch_test_loss = 0
        epoch_test_accuracy = 0
        for _, (images, masks) in enumerate(train_loader):
            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE)

            predictions = unet(images)
            loss = loss_fn(predictions, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # display
            image, mask = train_dataset.__getitem__(random.randint(0, len(train_dataset) - 1))

            with torch.no_grad():
                pred_binary = (predictions > 0.5).float()
                epoch_accuracy += (pred_binary == masks).float().mean().item()
                train_batch_count += 1

                out = (unet(image.view(1, 1, *config.INPUT_IMAGE_SIZE).to(config.DEVICE)).view(
                    *config.INPUT_IMAGE_SIZE).cpu().numpy() > 0) * 1.0
                unet.train()
            mask = mask.view(*config.INPUT_IMAGE_SIZE).cpu().numpy()
            out = np.concatenate((out, mask), axis=1)
            cv.imshow("Predicted / Actual", np.uint8(255 * out))
            cv.waitKey(1)
        with torch.no_grad():
            unet.eval()
            for batch_idx, (images, masks) in enumerate(test_loader):
                images = images.to(config.DEVICE)
                masks = masks.to(config.DEVICE)

                predictions = unet(images)
                loss = loss_fn(predictions, masks)

                epoch_test_loss += loss.item()
                epoch_test_accuracy += ((predictions > 0.5) == masks).float().mean().item()

        train_loss.append(epoch_loss / train_steps)
        test_loss.append(epoch_test_loss / test_steps)

        train_accuracy.append(epoch_accuracy / train_steps)
        test_accuracy.append(epoch_test_accuracy / test_steps)

        if display:
            image, mask = test_dataset.__getitem__(random.randint(0, len(test_dataset) - 1))
            with torch.no_grad():
                unet.eval()
                h, w = config.INPUT_IMAGE_SIZE
                output = unet(image.view(1, 1, h, w).to(device))
                output = torch.sigmoid(output).view(h, w).cpu().numpy()
                pred_mask = (output > 0.5).astype(np.uint8) * 255
                true_mask = mask.view(h, w).cpu().numpy().astype(np.uint8) * 255
                unet.train()

            combined = np.concatenate((pred_mask, true_mask), axis=1)
            cv.imshow("Predicted / Actual", combined)
            cv.waitKey(1)

        print(f"Epoch {epoch + 1}/{params.NUM_EPOCHS} - Train loss: {train_loss[-1]:.4f} - Test loss: {test_loss[-1]:.4f} - Train accuracy: {train_accuracy[-1]:.4f} - Test accuracy: {test_accuracy[-1]:.4f}")
        torch.save(unet.state_dict(), config.MODEL_PATH + f"unet_{epoch + 1}_epoch.pth")
        print(f"Model saved to {config.MODEL_PATH}")

        print(f"Training completed in {time.time() - start_time:.2f} seconds")

    if display:
        cv.destroyAllWindows()
    
    print("Training finished!")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")
    print(f"Model saved to {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main(None)

            

        