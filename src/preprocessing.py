import os

import cv2
import tqdm
from imutils import paths
from torchvision import transforms

import src.config as config

# Change the paths to your dataset train directories
TRAIN_DIR = config.TRAIN_IMAGES_DIR
TRAIN_MASKS_DIR = config.TRAIN_MASKS_DIR

# Change the paths to your dataset test directories
TEST_DIR = config.TEST_IMAGES_DIR
TEST_MASKS_DIR = config.TEST_MASKS_DIR

# Change the paths to your dataset train and test preprocessed directories
P_TRAIN_DIR = config.PREPROCESSED_TRAIN_IMAGES_DIR
P_TRAIN_MASKS_DIR = config.PREPROCESSED_TRAIN_MASKS_DIR
P_TEST_DIR = config.PREPROCESSED_TEST_IMAGES_DIR
P_TEST_MASKS_DIR = config.PREPROCESSED_TEST_MASKS_DIR

images_paths = list(paths.list_images(TRAIN_DIR))
masks_paths = list(paths.list_images(TRAIN_MASKS_DIR))

test_images_paths = list(paths.list_images(TEST_DIR))
test_masks_paths = list(paths.list_images(TEST_MASKS_DIR))

# Defines the augmentation being applied to every image in the dataset
image_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(config.INPUT_IMAGE_SIZE), 
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToPILImage(),
    
])

# Reads each image and its mask, applies the transformation, and stores them in the specified directories
def preprocess(images, masks, transforms, desc, images_dir=None, masks_dir=None):
    for image_path, mask_path in tqdm.tqdm(zip(images, masks), total=len(images), desc=desc):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

        image = transforms(image)
        mask = transforms(mask)

        # Save the preprocessed images and masks
        image_name = os.path.basename(image_path)
        mask_name = os.path.basename(mask_path)
        image.save(os.path.join(images_dir, image_name))
        mask.save(os.path.join(masks_dir, mask_name))

# Makes image directories and calls the preprocess function
if __name__ == "__main__":
    os.makedirs(P_TEST_DIR, exist_ok=True)
    os.makedirs(P_TRAIN_MASKS_DIR, exist_ok=True)
    os.makedirs(P_TEST_DIR, exist_ok=True)
    os.makedirs(P_TEST_MASKS_DIR, exist_ok=True)

    preprocess(images_paths, masks_paths, image_transforms, "Preprocessing train images and masks", config.PREPROCESSED_TRAIN_SELFIES_DIR, config.PREPROCESSED_TRAIN_SELFIES_MASKS_DIR)
    preprocess(test_images_paths, test_masks_paths, image_transforms, "Preprocessing test images and masks", config.PREPROCESSED_TEST_IMAGES_DIR, config.PREPROCESSED_TEST_MASKS_DIR)
    print("Preprocessing completed.")