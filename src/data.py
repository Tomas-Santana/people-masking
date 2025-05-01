import torch
from torch.utils.data import Dataset
import cv2
import torchvision.transforms.functional as TF
from PIL import Image
import random


class PeopleMaskingDataset(Dataset):
    def __init__(self, image_paths: list[str], mask_paths: list[str], augment = True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # Convert to tensors early, since TF.* expects tensors in your version
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0  # [1, H, W]
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0    # [1, H, W]

        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            angle = random.uniform(-15, 15)
            translate = [random.uniform(-0.1, 0.1) * image.shape[2], random.uniform(-0.1, 0.1) * image.shape[1]]
            scale = random.uniform(0.8, 1.2)
            shear = random.uniform(-10, 10)
            image = TF.affine(image, angle=angle, translate=translate, scale=scale, shear=[shear, 0])
            mask = TF.affine(mask, angle=angle, translate=translate, scale=scale, shear=[shear, 0])

        mask = (mask > 0.5).float()
        return image, mask