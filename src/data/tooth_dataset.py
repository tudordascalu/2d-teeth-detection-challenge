import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from src.utils.label_encoder import LabelEncoder


class ToothDataset(Dataset):
    def __init__(self, dataset, image_dir, transform=None, device="cpu"):
        self.dataset = dataset
        self.image_dir = image_dir
        self.encoder = LabelEncoder()
        self.transform = transform
        self.device = device

    def __getitem__(self, idx):
        # Select sample
        sample = self.dataset[idx]

        # Load image, box, label
        image = read_image(f"{self.image_dir}/{sample['file_name']}")
        box = torch.tensor(sample["annotation"]["bbox"], dtype=torch.int32)
        disease = sample["annotation"]["category_id_3"]

        # Crop image and normalize image
        image = image[:, box[1]:box[3], box[0]:box[2]] / 255

        if self.transform is not None:
            image = self.transform(image)

        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(disease, dtype=torch.float32).unsqueeze(-1)

        return dict(image=image,
                    label=label,
                    file_name=sample["file_name"])

    def __len__(self):
        return len(self.dataset)
