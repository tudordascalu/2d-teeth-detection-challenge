import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.nn.functional import pad

from src.utils.label_encoder import LabelEncoder


class ToothDataset(Dataset):
    def __init__(self, dataset, image_dir, transform=None):
        self.dataset = dataset
        self.image_dir = image_dir
        self.encoder = LabelEncoder()
        self.transform = transform

    def __getitem__(self, idx):
        # Select sample
        sample = self.dataset[idx]

        # Load image, box, label
        image = read_image(f"{self.image_dir}/{sample['file_name']}")
        box = torch.tensor(sample["annotation"]["bbox"], dtype=torch.int32)
        label = sample["annotation"]["category_id_3"]

        # Crop image and normalize image
        image = image[0, box[1]:box[3], box[0]:box[2]].unsqueeze(0) / 255

        image = self.transform(image)

        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int64)

        return dict(image=image, label=label)

    @staticmethod
    def collate_fn(batch):
        # Unpack batch
        images = [b["image"] for b in batch]
        labels = [b["label"] for b in batch]

        # Determine the max height and width
        max_h = max([img.shape[1] for img in images])
        max_w = max([img.shape[2] for img in images])

        # Pad all images to match the max height and width
        images = [pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1])) for img in images]

        # Convert to tensor
        images = torch.stack(images)
        labels = torch.stack(labels)

        return {"image": images, "label": labels}

    def __len__(self):
        return len(self.dataset)
