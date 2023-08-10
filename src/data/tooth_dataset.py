import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from src.utils.label_encoder import LabelEncoder
from src.utils.one_hot import OneHotMultiLabel


class ToothDataset(Dataset):
    def __init__(self, dataset, image_dir, transform=None, n_classes=4):
        self.dataset = dataset
        self.image_dir = image_dir
        self.encoder = LabelEncoder()
        self.transform = transform
        self.n_classes = n_classes
        self.one_hot_multilabel = OneHotMultiLabel()

    def __getitem__(self, idx):
        # Select sample
        sample = self.dataset[idx]

        # Load image, box, label
        image = read_image(f"{self.image_dir}/{sample['file_name']}")
        box = torch.tensor(sample["annotation"]["bbox"], dtype=torch.int32)
        disease = sample["annotation"]["category_id_3"]
        disease_list = sample["annotation"]["category_id_3_list"]

        # Crop image and normalize image
        image = image[:, box[1]:box[3], box[0]:box[2]] / 255

        if self.transform is not None:
            image = self.transform(image)

        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32)

        if self.n_classes > 1:
            label = self.one_hot_multilabel(
                torch.tensor(disease_list, dtype=torch.int64).unsqueeze(0),
                num_classes=self.n_classes
            ).squeeze()
        else:
            label = torch.tensor(disease, dtype=torch.float32).unsqueeze(-1)

        return dict(image=image,
                    label=label,
                    file_name=sample["file_name"])

    def __len__(self):
        return len(self.dataset)
