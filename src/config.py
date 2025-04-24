import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_IMAGE_SIZE = (128, 128)

IMAGES_DIR = os.path.join("dataset", "images")
IMAGES_DIR = os.path.abspath(IMAGES_DIR)
RESIZED_IMAGES_DIR = os.path.join("dataset", "resized_images")
RESIZED_IMAGES_DIR = os.path.abspath(RESIZED_IMAGES_DIR)

MASKS_DIR = os.path.join("dataset", "masks")
MASKS_DIR = os.path.abspath(MASKS_DIR)
RESIZED_MASKS_DIR = os.path.join("dataset", "resized_masks")
RESIZED_MASKS_DIR = os.path.abspath(RESIZED_MASKS_DIR)

CHECKPOINTS_DIR = os.path.join("checkpoints")
CHECKPOINTS_DIR = os.path.abspath(CHECKPOINTS_DIR)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

OUTPUT_DIR = os.path.join("output")
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_OUTPUT = os.path.sep.join([OUTPUT_DIR, "output"])

MODEL_PATH = os.path.abspath(os.path.join(CHECKPOINTS_DIR, "model_v2.pth"))
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])


