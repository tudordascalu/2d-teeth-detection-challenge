import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.nn.functional import pad

from src.utils.label_encoder import LabelEncoder


class ToothDataset(Dataset):
    def __init__(self, dataset, image_dir, transform=None):
        self.dataset = self._unpack_dataset(dataset)
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

        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int32)

        return dict(image=image, label=label)

    @staticmethod
    def _unpack_dataset(x):
        """
        Ensure that each box is its own item.
        :return: list of dictionaries with keys "file_name" and "annotation"
        """
        return [dict(file_name=sample["file_name"], annotation=annotation) for sample in x for annotation in
                sample["annotations"]]

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
        images = torch.tensor(images, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int32)

        return {"image": images, "label": labels}

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    y = np.load(f"data/final/y_quadrant_enumeration_disease_train.npy", allow_pickle=True)
    dataset = ToothDataset(y, "data/raw/training_data/quadrant_enumeration_disease/xrays", transform=None)
    sample = dataset[0]
    print(sample)
