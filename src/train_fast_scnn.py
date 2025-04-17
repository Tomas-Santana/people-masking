import cv2
from data import PeopleMaskingDataset
from models.FastSCNN import fastscnn
import config
import params
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

    transformations = transforms.Compose([transforms.ToPILImage(), transforms.Resize(config.INPUT_IMAGE_SIZE), transforms.ToTensor()])
    train_dataset = PeopleMaskingDataset(images_train, masks_train, transformations)
    test_dataset = PeopleMaskingDataset(images_test, masks_test, transformations)
    train_loader = DataLoader(train_dataset, batch_size=params.BATCH_SIZE, shuffle=True, num_workers=params.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=params.BATCH_SIZE, shuffle=False, num_workers=params.NUM_WORKERS)
    model = fastscnn().to(config.DEVICE)

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
            loss = loss_fn(predictions, masks)
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
                loss = loss_fn(predictions, masks)

                epoch_test_loss += loss.item()
                # Calculate accuracy
                predictions = torch.sigmoid(predictions)
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

def demo():
    model = fastscnn().to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.eval()

    image_path = "path_to_your_image.jpg"
    image = cv2.imread(image_path)
    image = cv2.resize(image, config.INPUT_IMAGE_SIZE)
    image = transforms.ToTensor()(image).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output)
        output = (output > 0.5).float()

    output_image = output.squeeze().cpu().numpy()
    cv2.imshow("Output", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    train()