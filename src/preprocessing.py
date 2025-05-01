import os
from imutils import paths
import cv2
import src.config as config
import tqdm
from torchvision import transforms

images_paths = list(paths.list_images(config.IMAGES_DIR))
masks_paths = list(paths.list_images(config.MASKS_DIR))

image_transforms = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.Resize(config.INPUT_IMAGE_SIZE), 
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0], std=[1]),
    transforms.GaussianBlur(kernel_size=5),
    transforms.ToPILImage(),
])

def preprocess(images, masks, transforms, desc):
    for image_path, mask_path in tqdm.tqdm(zip(images, masks), total=len(images), desc=desc):
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = transforms(image)
        mask = transforms(mask)

        # Save the preprocessed images and masks
        image_name = os.path.basename(image_path)
        mask_name = os.path.basename(mask_path)
        image.save(os.path.join(config.PREPROCESSED_IMAGES_DIR, image_name))
        mask.save(os.path.join(config.PREPROCESSED_MASKS_DIR, mask_name))


if __name__ == "__main__":
    os.makedirs(config.PREPROCESSED_IMAGES_DIR, exist_ok=True)
    os.makedirs(config.PREPROCESSED_MASKS_DIR, exist_ok=True)

    preprocess(images_paths, masks_paths, image_transforms, "Preprocessing images and masks")
    os.makedirs(config.PREPROCESSED_IMAGES_DIR, exist_ok=True)
    print("Preprocessing completed.")