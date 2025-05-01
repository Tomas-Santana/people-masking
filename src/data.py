import torch
from torch.utils.data import Dataset
import cv2
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import random


image_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    ])

class PeopleMaskingDataset(Dataset):
    def __init__(self, image_paths: list[str], mask_paths: list[str], augment = True, read_mode=cv2.IMREAD_GRAYSCALE):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment
        self.read_mode = read_mode

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path: str = self.image_paths[idx]
        image = cv2.imread(image_path, self.read_mode)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0

        if self.augment:
            seed = torch.randint(0, 999999, (1,)).item()

            torch.manual_seed(seed)
            if torch.rand(1).item() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            angle = torch.empty(1).uniform_(-15, 15).item()
            translate = [torch.empty(1).uniform_(-0.1, 0.1).item() * image.shape[2],
                         torch.empty(1).uniform_(-0.1, 0.1).item() * image.shape[1]]
            scale = torch.empty(1).uniform_(0.8, 1.2).item()
            shear = torch.empty(1).uniform_(-10, 10).item()

            image = TF.affine(image, angle=angle, translate=translate, scale=scale,
                              shear=[shear, 0], interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.affine(mask, angle=angle, translate=translate, scale=scale,
                             shear=[shear, 0], interpolation=TF.InterpolationMode.NEAREST)

        mask = (mask > 0.5).float()
        return image.clamp(0, 1), mask