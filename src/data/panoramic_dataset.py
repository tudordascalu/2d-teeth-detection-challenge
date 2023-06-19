import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from src.utils.label_encoder import LabelEncoder


class PanoramicDataset(Dataset):
    def __init__(self, dataset, image_dir, data_type="quadrant"):
        self.dataset = dataset
        self.image_dir = image_dir
        self.data_type = data_type
        self.encoder = LabelEncoder()

    def __getitem__(self, idx):
        image_data = self.dataset[idx]
        image = read_image(f"{self.image_dir}/{image_data['file_name']}")[0].unsqueeze(0) / 255
        targets = dict(boxes=self._get_target_boxes(image_data["annotations"]),
                       labels=self._get_target_labels(image_data["annotations"]))
        return {"image": image, "targets": targets, "id": image_data['file_name'].split(".")[0]}

    def _get_target_labels(self, annotations):
        if self.data_type == "quadrant":
            return self._get_target_labels_quadrant(annotations=annotations)
        elif self.data_type == "quadrant_enumeration":
            return self._get_target_labels_quadrant_enumeration(annotations=annotations)
        else:
            raise "Invalid data_type! It should be \"quadrant\" or \"enumeration\"."

    def _get_target_boxes(self, annotations):
        boxes = torch.tensor(list(map(lambda x: x["bbox"], annotations)), dtype=torch.float32)
        return boxes

    def _get_target_labels_quadrant_enumeration(self, annotations):
        labels = list(map(lambda x: x["category_id_1"] * 10 + x["category_id_2"] + 1, annotations))
        # Transform labels from 1, 2, .., 8, 11, 12, .. 18, .. into 1, 2, .., 8, 9, 10, .., 16, ..
        labels = torch.tensor(self.encoder.transform(labels), dtype=torch.int64)
        return labels

    def _get_target_labels_quadrant(self, annotations):
        labels = torch.tensor(list(map(lambda x: x["category_id"] + 1, annotations)), dtype=torch.int64)
        return labels

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def collate_fn(batch):
        images = []
        targets = []
        ids = []
        for b in batch:
            images.append(b["image"])
            targets.append(b["targets"])
            ids.append(b["id"])
        return {"image": images, "targets": targets, "id": ids}
