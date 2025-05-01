import torch
from torch.utils.data import Dataset
import cv2


class PeopleMaskingDataset(Dataset):
    def __init__(self, image_paths: list[str], mask_paths: list[str], transform=None, read_mode=cv2.IMREAD_COLOR):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.read_mode = read_mode

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path: str = self.image_paths[idx]
        mask_path: str = self.mask_paths[idx]

        image = cv2.imread(image_path, self.read_mode)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)

        return image, mask

        