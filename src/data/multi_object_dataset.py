import numpy as np
import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset

from src.utils.label_encoder import LabelEncoder
from src.utils.multi_object_labeling import InterObjectDistanceMapper, InterObjectScoreMapper, \
    MultiObjectCentroidMapper


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
        input = torch.concat((inter_object_distance_mat, inter_object_score_mat), dim=-1).permute(2, 0, 1)
        return input, labels

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
