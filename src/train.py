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


def main(checkpoint_path=None):
    now = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    image_paths = list(paths.list_images(config.IMAGES_DIR))
    mask_paths = list(paths.list_images(config.MASKS_DIR))
    images_train, images_test, masks_train, masks_test = train_test_split(
        image_paths, mask_paths, test_size=params.TRAIN_TEST_SPLIT, random_state=42
    )

    image_transforms = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize(config.INPUT_IMAGE_SIZE), 
        transforms.ToTensor(),
    ])
    train_dataset = PeopleMaskingDataset(images_train, masks_train, image_transforms)
    test_dataset = PeopleMaskingDataset(images_test, masks_test, image_transforms)
    train_loader = DataLoader(train_dataset, batch_size=params.BATCH_SIZE, shuffle=True, num_workers=params.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=params.BATCH_SIZE, shuffle=False, num_workers=params.NUM_WORKERS)

    if checkpoint_path:
        unet = UNET(in_channels=3, out_channels=1).to(config.DEVICE)
        unet.load_state_dict(torch.load(checkpoint_path))
    else:
        unet = UNET(in_channels=3, out_channels=1).to(config.DEVICE)
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
    for epoch in tqdm(range(params.NUM_EPOCHS)):
        unet.train()

        epoch_loss = 0
        epoch_accuracy = 0
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
        with torch.no_grad():
            unet.eval()
            for batch_idx, (images, masks) in enumerate(test_loader):
                images = images.to(config.DEVICE)
                masks = masks.to(config.DEVICE)

                predictions = unet(images)
                loss = loss_fn(predictions, masks)

                epoch_test_loss += loss.item()
                epoch_test_accuracy += ((predictions > 0.5) == masks).float().mean().item()
                epoch_accuracy += ((predictions > 0.5) == masks).float().mean().item()

        train_loss.append(epoch_loss / train_steps)
        test_loss.append(epoch_test_loss / test_steps)

        train_accuracy.append(epoch_accuracy / train_steps)
        test_accuracy.append(epoch_test_accuracy / test_steps)

        print(f"Epoch {epoch + 1}/{params.NUM_EPOCHS} - Train loss: {train_loss[-1]:.4f} - Test loss: {test_loss[-1]:.4f} - Train accuracy: {train_accuracy[-1]:.4f} - Test accuracy: {test_accuracy[-1]:.4f}")
        
        torch.save(unet.state_dict(), config.MODEL_PATH)
        print(f"Model saved to {config.MODEL_PATH}")

        print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    print("Training finished!")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")
    print(f"Model saved to {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()

            

        