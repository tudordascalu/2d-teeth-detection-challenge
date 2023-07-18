import json

import numpy as np
import torch
from torchvision.io import read_image
from torchvision.ops import box_iou
from tqdm import tqdm

from src.model.faster_rcnn.faster_rcnn import FasterRCNN
from src.utils.label_encoder import LabelEncoder
from src.utils.processors import UniqueClassNMSProcessor


class FilterOverlappingBoxes:
    def __init__(self, iou_threshold=.75):
        self.iou_threshold = iou_threshold
        pass

    def __call__(self, boxes1, boxes2):
        """
        Returns indices of non overlapping between boxes_1, boxes_2 corresponding to boxes 1.
        """
        # Compute the IoU matrix
        iou_matrix = box_iou(boxes1, boxes2)

        # Find boxes in the first list that overlap with any box in the second list
        non_overlapping_indices = torch.all(iou_matrix < self.iou_threshold, dim=1).nonzero().squeeze().tolist()

        # Return indices of overlapping boxes as a list
        return non_overlapping_indices


if __name__ == "__main__":
    # GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test split
    with open("data/processed/train_quadrant_enumeration_disease.json") as f:
        X = json.load(f)

    # Model
    model = FasterRCNN.load_from_checkpoint(
        "checkpoints/faster_rcnn/version_3/checkpoints/epoch=epoch=88-val_loss=val_loss=0.81.ckpt")
    model.eval()

    # Helpers
    unique_class_nms_processor = UniqueClassNMSProcessor(iou_threshold=.75)
    filter_overlapping_boxes = FilterOverlappingBoxes(iou_threshold=.5)
    encoder = LabelEncoder()

    for sample in tqdm(X, total=len(X)):
        # Load image
        file_name = sample["file_name"]
        image = read_image(f"data/raw/training_data/quadrant_enumeration_disease/xrays/{file_name}").type(torch.float32)
        image = image[0, ...].unsqueeze(0) / 255

        # Predict
        with torch.no_grad():
            predictions = model(image.unsqueeze(0))[0]
        predictions = unique_class_nms_processor(predictions)

        # Extract data from predictions
        predicted_scores = predictions["scores"]
        predicted_boxes = torch.stack(predictions["boxes"])
        predicted_labels = predictions["labels"]
        predicted_labels_decoded = encoder.inverse_transform(predicted_labels)
        predicted_teeth = [int((label - 1) % 10) for label in predicted_labels_decoded]
        predicted_quadrants = [int((label - 1) / 10) for label in predicted_labels_decoded]
        predicted_annotations = []
        for box, score, label, tooth, quadrant in zip(predicted_boxes, predicted_scores, predicted_labels,
                                                      predicted_teeth, predicted_quadrants):
            predicted_annotations.append(
                dict(bbox=box.tolist(),
                     score=score.item(),
                     label=label,
                     category_id_1=quadrant,
                     category_id_2=tooth,
                     category_id_3=4,
                     category_id_3_list=[4]))

        # Remove annotations overlapping with existing annotations
        boxes = torch.tensor([annotation["bbox"] for annotation in sample["annotations"]])
        if len(boxes) > 0:
            indices_non_overlapping = filter_overlapping_boxes(predicted_boxes, boxes)
            predicted_annotations = np.array(predicted_annotations)[indices_non_overlapping].tolist()

        sample["annotations"].extend(predicted_annotations)

    with open("data/processed/train_quadrant_enumeration_disease_healthy.json", 'w') as f:
        json.dump(X, f, indent=4)
