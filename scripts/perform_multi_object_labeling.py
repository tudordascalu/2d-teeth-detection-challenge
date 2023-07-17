"""
This script aims to take the outputs of the Faster-RCNN model and correct them using the MultiObjectLabelingCNN.
"""
import glob
import pickle

import numpy as np
import torch
from tqdm import tqdm

from src.model.multi_object_labeling_cnn.multi_object_labeling_cnn import MultiObjectLabelingCNN
from src.utils.multi_object_labeling import MultiObjectCentroidMapper, InterObjectDistanceMapper, \
    InterObjectScoreMapper, AssignmentSolver
from src.utils.processors import UniqueClassNMSProcessor

if __name__ == "__main__":
    # Load statistics
    inter_object_distance_mat_mean = np.load("../data/final/inter_object_distance_mat_mean.npy")
    inter_object_distance_mat_std = np.load("../data/final/inter_object_distance_mat_std.npy")

    # Initialize helpers
    unique_class_nms_processor = UniqueClassNMSProcessor(iou_threshold=.75)
    multi_object_centroid_mapper = MultiObjectCentroidMapper()
    inter_object_distance_mapper = InterObjectDistanceMapper()
    inter_object_score_mapper = InterObjectScoreMapper(inter_object_distance_mat_mean,
                                                       inter_object_distance_mat_std)
    assignment_solver = AssignmentSolver()

    # Compute sample names
    data_path = "../output/version_3__epoch=epoch=88-val_loss=val_loss=0"
    samples = list(map(lambda x: x.split("/")[-1], glob.glob(f"{data_path}/*")))

    for sample in tqdm(samples, total=len(samples)):
        # Load predictions
        predictions = np.load(f"{data_path}/{sample}/predictions.npy", allow_pickle=True)

        # Overlap targets and boxes on image
        predictions = unique_class_nms_processor(predictions)

        # Extract boxes and labels
        scores = predictions["scores"]
        boxes = predictions["boxes"]
        labels = predictions["labels"]

        # Compute centroids
        object_centroids = multi_object_centroid_mapper(boxes, labels)
        inter_object_distance_mat = inter_object_distance_mapper(object_centroids)
        inter_object_score_mat = inter_object_score_mapper(object_centroids, inter_object_distance_mat)

        # Convert to tensors
        inter_object_distance_mat = torch.tensor(inter_object_distance_mat, dtype=torch.float32)
        inter_object_score_mat = torch.tensor(inter_object_score_mat, dtype=torch.float32)

        # Compute input
        input = torch.concat((inter_object_distance_mat, inter_object_score_mat), dim=-1).permute(2, 0, 1)

        # Load model
        model = MultiObjectLabelingCNN.load_from_checkpoint(
            "../checkpoints/multi_object_labeling_cnn/version_5/checkpoints/epoch=epoch=71-val_loss=val_loss=0.00.ckpt")

        # Predict
        labels_predicted = model(input.unsqueeze(0))

        # Process labels
        labels_predicted = labels_predicted.detach().cpu().numpy()
        labels_predicted, _ = assignment_solver(labels_predicted)
        labels_predicted = labels_predicted[0]

        # Use predicted labels to correct labels
        labels = [labels_predicted[label - 1] + 1 for label in labels]

        # Save dict
        predictions_processed = dict(boxes=boxes, labels=labels, scores=scores)
        with open(f"{data_path}/{sample}/predictions_processed.npy", "wb") as file:
            pickle.dump(predictions_processed, file)
