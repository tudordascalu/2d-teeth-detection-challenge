import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from src.utils.label_encoder import LabelEncoder


class PanoramicDataset(Dataset):
    def __init__(self, dataset, image_dir, transforms=lambda x: x):
        self.dataset = dataset
        self.image_dir = image_dir
        self.transforms = transforms
        self.encoder = LabelEncoder()

    def __getitem__(self, idx):
        image_data = self.dataset[idx]
        image = read_image(f"{self.image_dir}/{image_data['file_name']}")[0].unsqueeze(0) / 255
        image = self.transforms(image)
        targets = self._get_targets(image_data["annotations"])
        return image, targets

    def _get_targets(self, annotations):
        boxes = torch.tensor(list(map(lambda x: x["bbox"], annotations)), dtype=torch.float32)
        labels = list(map(lambda x: x["category_id_1"] * 10 + x["category_id_2"] + 1, annotations))
        labels_encoded = self.encoder.transform(labels)
        labels_encoded = torch.tensor(labels_encoded, dtype=torch.int64)
        return dict(boxes=boxes, labels=labels_encoded)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def collate_fn(batch):
        images = []
        targets = []
        for b in batch:
            images.append(b[0])
            targets.append(b[1])
        return images, targets
