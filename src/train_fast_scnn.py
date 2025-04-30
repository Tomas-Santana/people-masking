import cv2
from src.data import PeopleMaskingDataset
from src.models.FastSCNN import get_fast_scnn
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

def train():
    image_paths = list(paths.list_images(config.RESIZED_IMAGES_DIR))
    mask_paths = list(paths.list_images(config.RESIZED_MASKS_DIR))
    images_train, images_test, masks_train, masks_test = train_test_split(
        image_paths, mask_paths, test_size=params.TRAIN_TEST_SPLIT, random_state=42
    )

    image_transforms = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize(config.INPUT_IMAGE_SIZE), 
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])
    train_dataset = PeopleMaskingDataset(images_train, masks_train, image_transforms)
    test_dataset = PeopleMaskingDataset(images_test, masks_test, image_transforms)
    train_loader = DataLoader(train_dataset, batch_size=params.BATCH_SIZE, shuffle=True, num_workers=params.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=params.BATCH_SIZE, shuffle=False, num_workers=params.NUM_WORKERS)
    model = get_fast_scnn(checkpoint_path=None, device=config.DEVICE, num_classes=1)

    loss_fn = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=params.LR)
    train_steps = len(train_loader)

    test_steps = len(test_loader)

    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    print("Starting training...")
    start_time = time.time()
    for epoch in tqdm(range(params.NUM_EPOCHS)):
        model.train()

        epoch_loss = 0
        epoch_accuracy = 0
        epoch_test_loss = 0
        epoch_test_accuracy = 0
        for _, (images, masks) in enumerate(train_loader):
            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE)

            predictions = model(images)

            print(type(predictions[0]))
            
            loss = loss_fn(predictions[0], masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        with torch.no_grad():
            model.eval()
            for batch_idx, (images, masks) in enumerate(test_loader):
                images = images.to(config.DEVICE)
                masks = masks.to(config.DEVICE)

                predictions = model(images)
                loss = loss_fn(predictions[0], masks)

                epoch_test_loss += loss.item()
                # Calculate accuracy
                predictions = torch.sigmoid(predictions[0])
                predictions = (predictions > 0.5).float()
                epoch_test_accuracy += ((predictions == masks).float()).mean().item()
                epoch_accuracy += ((predictions == masks).float()).mean().item()


        epoch_loss /= train_steps
        epoch_accuracy /= train_steps
        epoch_test_loss /= test_steps
        epoch_test_accuracy /= test_steps

        train_loss.append(epoch_loss)
        test_loss.append(epoch_test_loss)

        train_accuracy.append(epoch_accuracy)
        test_accuracy.append(epoch_test_accuracy)

        print(f"Epoch {epoch + 1}/{params.NUM_EPOCHS} - Train loss: {train_loss[-1]:.4f} - Test loss: {test_loss[-1]:.4f} - Train accuracy: {train_accuracy[-1]:.4f} - Test accuracy: {test_accuracy[-1]:.4f}")
        torch.save(model.state_dict(), config.MODEL_PATH)
        print(f"Model saved to {config.MODEL_PATH}")
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
if __name__ == "__main__":
    train()