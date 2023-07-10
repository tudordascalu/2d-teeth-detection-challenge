"""
This script is designed to enrich the "quadrant_enumeration_disease" dataset by introducing labels for healthy teeth.
The first stage of the script operates by utilizing the Faster-RCNN model to derive predictions for every sample in the dataset.
Following this, any teeth that do not exhibit a substantial overlap with the identified unhealthy teeth are added to the dataset.
"""
import os

import numpy as np
import torch
from torchvision.io import read_image
from torchvision.ops import box_iou
from tqdm import tqdm

from src.model.faster_rcnn.faster_rcnn import FasterRCNN
from src.utils.processors import UniqueClassNMSProcessor


class FilterOverlappingBoxes:
    def __init__(self, iou_threshold=.75):
        self.iou_threshold = iou_threshold
        pass

    def __call__(self, boxes_1, boxes_2):
        """
        Returns indices of non overlapping between boxes_1, boxes_2 corresponding to boxes 1
        :param boxes_1: torch.tensor
        :param boxes_2: torch.tensor
        :return:
        """
        # Compute the IoU matrix
        iou_matrix = box_iou(boxes_1, boxes_2)

        # Find boxes in the first list that overlap with any box in the second list
        non_overlapping_indices = torch.all(iou_matrix < self.iou_threshold, dim=1).nonzero().squeeze().tolist()

        # Return indices of overlapping boxes as a list
        return non_overlapping_indices


if __name__ == "__main__":
    # Load data
    y = np.load("data/processed/y_quadrant_enumeration_disease.npy", allow_pickle=True)

    # Load model
    checkpoint = dict(version="version_", model="epoch=epoch=192-val_loss=val_loss=0.84.ckpt")
    model = FasterRCNN.load_from_checkpoint(
        "checkpoints/faster_rcnn/version_3/checkpoints/epoch=epoch=88-val_loss=val_loss=0.81.ckpt")
    model.eval()
    # Initialize helpers
    unique_class_nms_processor = UniqueClassNMSProcessor(iou_threshold=.75)
    filter_overlapping_boxes = FilterOverlappingBoxes(iou_threshold=.5)

    y_processed = []

    # Make predictions for each sample
    for sample in tqdm(y, total=len(y)):
        # Load image
        image = read_image(f"data/raw/training_data/quadrant_enumeration_disease/xrays/{sample['file_name']}")
        image = image.type(torch.float32)[0].unsqueeze(0) / 255

        # Make predictions for healthy teeth
        with torch.no_grad():
            predictions = model(image.unsqueeze(0))[0]
        predictions = unique_class_nms_processor(predictions)

        # Extract boxes, scores
        boxes = torch.stack(predictions["boxes"])
        scores = predictions["scores"]

        # Get affected tooth boxes with maximum confidence score
        boxes_affected = torch.tensor([annotation["bbox"] for annotation in sample["annotations"]],
                                      dtype=torch.float32)
        labels_affected = torch.tensor([annotation["category_id_3"] for annotation in sample["annotations"]],
                                       dtype=torch.int64)

        # Remove overlapping boxes
        if len(boxes_affected) > 0:
            indices = filter_overlapping_boxes(boxes, boxes_affected)
            boxes = boxes[indices]

        # Prepare labels for healthy teeth
        labels = torch.ones(len(boxes), dtype=torch.int64) * 4

        # Concatenate healthy and affected teeth
        boxes = torch.cat((boxes, boxes_affected))
        labels = torch.cat((labels, labels_affected))

        # Update annotations to include healthy teeth
        annotations_processed = [dict(bbox=box, category_id_3=label) for box, label in zip(boxes, labels)]
        sample["annotations"] = annotations_processed

        # Accumulate processed samples
        y_processed.append(sample)

    # Save dataset
    np.save("data/processed/y_quadrant_enumeration_disease_with_healthy_samples_2.npy", y_processed)

    # Unpack dataset
    y_processed_unpacked = []
    for sample in y_processed:
        for annotation in sample["annotations"]:
            if torch.is_tensor(annotation["category_id_3"]):
                annotation["category_id_3"] = annotation["category_id_3"].item()
            y_processed_unpacked.append(dict(file_name=sample["file_name"], annotation=annotation))

    # Save unpacked dataset
    np.save("data/processed/y_quadrant_enumeration_disease_with_healthy_samples_unpacked_2.npy", y_processed_unpacked)
