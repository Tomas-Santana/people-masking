import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_IMAGE_SIZE = (128, 128)

# First dataset: supervisely person masking dataset
# Train data
TRAIN_IMAGES_DIR = os.path.join("dataset_images","train", "images")
TRAIN_IMAGES_DIR = os.path.abspath(TRAIN_IMAGES_DIR)
TRAIN_MASKS_DIR = os.path.join("dataset_images", "train", "masks")
TRAIN_MASKS_DIR = os.path.abspath(TRAIN_MASKS_DIR)

PREPROCESSED_TRAIN_IMAGES_DIR = os.path.join("dataset_images", "train", "preprocessed_images")
PREPROCESSED_TRAIN_IMAGES_DIR = os.path.abspath(PREPROCESSED_TRAIN_IMAGES_DIR)
PREPROCESSED_TRAIN_MASKS_DIR = os.path.join("dataset_images", "train", "preprocessed_masks")
PREPROCESSED_TRAIN_MASKS_DIR = os.path.abspath(PREPROCESSED_TRAIN_MASKS_DIR)

# Test data
TEST_IMAGES_DIR = os.path.join("dataset_images", "test", "images")
TEST_IMAGES_DIR = os.path.abspath(TEST_IMAGES_DIR)
TEST_MASKS_DIR = os.path.join("dataset_images", "test", "masks")
TEST_MASKS_DIR = os.path.abspath(TEST_MASKS_DIR)

PREPROCESSED_TEST_IMAGES_DIR = os.path.join("dataset_images", "test", "preprocessed_images")
PREPROCESSED_TEST_IMAGES_DIR = os.path.abspath(PREPROCESSED_TEST_IMAGES_DIR)
PREPROCESSED_TEST_MASKS_DIR = os.path.join("dataset_images", "test", "preprocessed_masks")
PREPROCESSED_TEST_MASKS_DIR = os.path.abspath(PREPROCESSED_TEST_MASKS_DIR)

# Second dataset: Selfie or not dataset with yolo masks
# Train data
TRAIN_SELFIES_DIR = os.path.join("dataset_selfies", "train", "images")
TRAIN_SELFIES_DIR = os.path.abspath(TRAIN_SELFIES_DIR)
TRAIN_SELFIES_MASKS_DIR = os.path.join("dataset_selfies", "train", "masks")
TRAIN_SELFIES_MASKS_DIR = os.path.abspath(TRAIN_SELFIES_MASKS_DIR)

PREPROCESSED_TRAIN_SELFIES_DIR = os.path.join("dataset_selfies", "train", "preprocessed_images")
PREPROCESSED_TRAIN_SELFIES_DIR = os.path.abspath(PREPROCESSED_TRAIN_SELFIES_DIR)
PREPROCESSED_TRAIN_SELFIES_MASKS_DIR = os.path.join("dataset_selfies", "train", "preprocessed_masks")
PREPROCESSED_TRAIN_SELFIES_MASKS_DIR = os.path.abspath(PREPROCESSED_TRAIN_SELFIES_MASKS_DIR)

# Test data
TEST_SELFIES_DIR = os.path.join("dataset_selfies", "test", "images")
TEST_SELFIES_DIR = os.path.abspath(TEST_SELFIES_DIR)
TEST_SELFIES_MASKS_DIR = os.path.join("dataset_selfies", "test", "masks")
TEST_SELFIES_MASKS_DIR = os.path.abspath(TEST_SELFIES_MASKS_DIR)

PREPROCESSED_TEST_SELFIES_DIR = os.path.join("dataset_selfies", "test", "preprocessed_images")
PREPROCESSED_TEST_SELFIES_DIR = os.path.abspath(PREPROCESSED_TEST_SELFIES_DIR)
PREPROCESSED_TEST_SELFIES_MASKS_DIR = os.path.join("dataset_selfies", "test", "preprocessed_masks")
PREPROCESSED_TEST_SELFIES_MASKS_DIR = os.path.abspath(PREPROCESSED_TEST_SELFIES_MASKS_DIR)


CHECKPOINTS_DIR = os.path.join("checkpoints")
CHECKPOINTS_DIR = os.path.abspath(CHECKPOINTS_DIR)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

OUTPUT_DIR = os.path.join("output")
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_OUTPUT = os.path.sep.join([OUTPUT_DIR, "output"])

MODEL_PATH = os.path.abspath(os.path.join(CHECKPOINTS_DIR, "model_v3_selfies+images_doubleconv.pth"))
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])


