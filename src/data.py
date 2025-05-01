import torchvision
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter
import torchvision.transforms.v2 as TVtransforms
import random
import torch
import cv2


class PeopleMaskingDataset(Dataset):
    def __init__(self, image_paths: list[str], mask_paths: list[str], transform=None, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.augment = augment
        self.color_jitter = ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path: str = self.image_paths[idx]
        mask_path: str = self.mask_paths[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Apply data augmentation
        if self.augment:
            if random.random() > 0.5:
                if random.random() > 0.25:
                    image = TF.hflip(image)
                    mask = TF.hflip(mask)

                if random.random() > 0.25:
                    image = TF.vflip(image)
                    mask = TF.vflip(mask)

                if random.random() > 0.5:
                    angle = random.uniform(-15, 15)
                    image = TF.rotate(image, angle)
                    mask = TF.rotate(mask, angle)

                if random.random() > 0.5:
                    image = self.color_jitter(image)

                if random.random() > 0.5:
                    gaussian_noise = TVtransforms.GaussianNoise(mean=0, sigma=0.1)
                    image = gaussian_noise(image)

        return image, mask

        