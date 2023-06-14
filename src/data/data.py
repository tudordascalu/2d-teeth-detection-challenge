import numpy as np
import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchvision.io import read_image

from src.utils.label_encoder import LabelEncoder
from src.utils.multi_object_labeling import InterObjectDistanceMapper, InterObjectScoreMapper, \
    MultiObjectCentroidMapper


class PanoramicDataset(Dataset):
    def __init__(self, dataset, image_dir, transforms=lambda x: x, data_type="quadrant"):
        self.dataset = dataset
        self.image_dir = image_dir
        self.transforms = transforms
        self.data_type = data_type
        self.encoder = LabelEncoder()

    def __getitem__(self, idx):
        image_data = self.dataset[idx]
        image = read_image(f"{self.image_dir}/{image_data['file_name']}")[0].unsqueeze(0) / 255
        image = self.transforms(image)
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


class MultiObjectDataset(Dataset):
    def __init__(self, dataset, inter_object_distance_mat_mean, inter_object_distance_mat_std, transforms=lambda x: x):
        self.dataset = dataset
        self.transforms = transforms
        self.encoder = LabelEncoder()
        self.multi_object_centroid_mapper = MultiObjectCentroidMapper()
        self.inter_object_distance_mapper = InterObjectDistanceMapper()
        self.inter_object_score_mapper = InterObjectScoreMapper(inter_object_distance_mat_mean,
                                                                inter_object_distance_mat_std)

    def __getitem__(self, idx):
        # Extract sample, boxes and corresponding labels
        sample = self.dataset[idx]
        boxes = self._get_boxes(sample["annotations"])
        box_labels = self._get_labels(sample["annotations"])

        # Compute centroids and labels
        object_centroids = self.multi_object_centroid_mapper(boxes, box_labels)

        # The centroids are in order; right upper jaw -> left upper jaw -> left lower jaw -> right lower jaw
        labels = np.arange(0, 32)

        # Augment data by performing random shift, swap, removal of objects
        object_centroids, labels = self.transforms((object_centroids, labels))

        # Compute distance matrix and score map
        inter_object_distance_mat = self.inter_object_distance_mapper(object_centroids)
        inter_object_score_mat = self.inter_object_score_mapper(object_centroids, inter_object_distance_mat)

        # Convert numpy array to tensors
        labels = torch.tensor(labels, dtype=torch.int64)
        inter_object_distance_mat = torch.tensor(inter_object_distance_mat, dtype=torch.float32)
        inter_object_score_mat = torch.tensor(inter_object_score_mat, dtype=torch.float32)

        # Encode labels
        labels = one_hot(labels, num_classes=32).squeeze(1).type(torch.float32)

        # Merge distance and score matrices
        inputs = torch.concat((inter_object_distance_mat, inter_object_score_mat), dim=-1).permute(2, 0, 1)
        return inputs, labels

    def _get_boxes(self, annotations):
        boxes = torch.tensor(list(map(lambda x: x["bbox"], annotations)), dtype=torch.float32)
        return boxes

    def _get_labels(self, annotations):
        labels = list(map(lambda x: x["category_id_1"] * 10 + x["category_id_2"] + 1, annotations))
        # Transform labels from 1, 2, .., 8, 11, 12, .. 18, .. into 1, 2, .., 8, 9, 10, .., 16, ..
        labels = torch.tensor(self.encoder.transform(labels), dtype=torch.int64)
        return labels

    def __len__(self):
        return len(self.dataset)
