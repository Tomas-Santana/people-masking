import os
from imutils import paths
import cv2
import config
import tqdm

images_paths = list(paths.list_images(config.IMAGES_DIR))
masks_paths = list(paths.list_images(config.MASKS_DIR))


def resize_images_and_masks(images_paths: list[str], masks_paths: list[str], size: tuple[int]) -> None:
    for image_path in tqdm.tqdm(images_paths, desc="Resizing images"):
        filename = image_path.split(os.path.sep)[-1]
        image = cv2.imread(image_path)
        image = cv2.resize(image, size)
        cv2.imwrite(os.path.join(config.RESIZED_IMAGES_DIR, filename), image)

    for mask_path in tqdm.tqdm(masks_paths, desc="Resizing masks"):
        filename = mask_path.split(os.path.sep)[-1]
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, size)
        cv2.imwrite(os.path.join(config.RESIZED_MASKS_DIR, filename), mask)


if __name__ == "__main__":
    os.makedirs(config.RESIZED_IMAGES_DIR, exist_ok=True)
    os.makedirs(config.RESIZED_MASKS_DIR, exist_ok=True)
    resize_images_and_masks(images_paths, masks_paths, config.INPUT_IMAGE_SIZE)