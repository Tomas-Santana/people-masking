import os
from imutils import paths
import cv2
import src.config as config
import tqdm
from torchvision import transforms

images_paths = list(paths.list_images(config.TRAIN_SELFIES_DIR))
masks_paths = list(paths.list_images(config.TRAIN_SELFIES_MASKS_DIR))

# test_images_paths = list(paths.list_images(config.TEST_SELFIES_DIR))
# test_masks_paths = list(paths.list_images(config.TEST_MASKS_DIR))

image_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(config.INPUT_IMAGE_SIZE), 
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0], std=[1]),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToPILImage(),
    
])


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


if __name__ == "__main__":
    os.makedirs(config.PREPROCESSED_TRAIN_SELFIES_DIR, exist_ok=True)
    os.makedirs(config.PREPROCESSED_TRAIN_SELFIES_MASKS_DIR, exist_ok=True)

    preprocess(images_paths, masks_paths, image_transforms, "Preprocessing train images and masks", config.PREPROCESSED_TRAIN_SELFIES_DIR, config.PREPROCESSED_TRAIN_SELFIES_MASKS_DIR)
    # os.makedirs(config.PREPROCESSED_TEST_IMAGES_DIR, exist_ok=True)
    # os.makedirs(config.PREPROCESSED_TEST_MASKS_DIR, exist_ok=True)
    # preprocess(test_images_paths, test_masks_paths, image_transforms, "Preprocessing test images and masks", config.PREPROCESSED_TEST_IMAGES_DIR, config.PREPROCESSED_TEST_MASKS_DIR)
    print("Preprocessing completed.")