import glob
import os
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt, patches
from torchvision.ops import box_iou
from tqdm import tqdm

from src.utils.label_encoder import LabelEncoder
from src.utils.processors import UniqueClassNMSProcessor


def assign_prediction_eval_labels(targets, predictions, iou_threshold=0.5):
    # Initialize label encoder for decoding labels
    encoder = LabelEncoder()
    # Load boxes and lables into varrs
    target_boxes = targets["boxes"]
    target_labels = encoder.inverse_transform(np.array(targets["labels"]))
    prediction_boxes = predictions["boxes"]
    prediction_labels = encoder.inverse_transform(np.array(predictions["labels"]))
    # Initialize label accumulator
    prediction_eval_labels = np.zeros(len(prediction_labels), dtype=np.int32)
    iou_mat = box_iou(torch.tensor(target_boxes), torch.tensor(prediction_boxes))
    for i, (target_box, target_label) in enumerate(zip(target_boxes, target_labels)):
        target_tooth_label = int(target_label % 10)
        target_tooth_quadrant = int(target_label / 10)
        try:
            j = np.argmax(iou_mat[i])
            iou = iou_mat[i, j]
            prediction_label = prediction_labels[j]
            # Check if target box was found
            if iou > iou_threshold:
                label = 0
                prediction_tooth_label = int(prediction_label % 10)
                prediction_tooth_quadrant = int(prediction_label / 10)
                # Check if both tooth and quadrant labels were
                if prediction_tooth_label == target_tooth_label and prediction_tooth_quadrant == target_tooth_quadrant:
                    label = 3
                # Check if tooth label was correct
                elif prediction_tooth_label == target_tooth_label:
                    label = 2
                # Check if quadrant lable was correct
                elif prediction_tooth_quadrant == target_tooth_quadrant:
                    label = 1
                # Assign label
                prediction_eval_labels[j] = label
        except:
            pass
    return prediction_eval_labels


if __name__ == "__main__":
    suffix = "processed"
    checkpoint = "version_3__epoch=epoch=88-val_loss=val_loss=0"
    task = "quadrant_enumeration"
    colors = np.load("../data/assets/colors.npy")
    if not os.path.exists(f"../visualizations/{checkpoint}"):
        os.mkdir(f"../visualizations/{checkpoint}")
    output_path = f"../output/{checkpoint}"
    data_path = "../data/raw/training_data"
    ids = [path.split("/")[-1] for path in glob.glob(f"{output_path}/*")]
    # unique_class_nms_processor = UniqueClassNMSProcessor(iou_threshold=.75)
    for id in tqdm(ids, total=len(ids)):
        # Load data
        image = plt.imread(f"{data_path}/{task}/xrays/{id}.png")[:, :, 0]
        with open(f"{output_path}/{id}/targets.npy", "rb") as file:
            targets = pickle.load(file)
        with open(f"{output_path}/{id}/predictions_{suffix}.npy", "rb") as file:
            predictions = pickle.load(file)
        # Overlap targets and boxes on image
        # predictions = unique_class_nms_processor(predictions)
        prediction_eval_labels = assign_prediction_eval_labels(targets, predictions, iou_threshold=.6)
        fig, ax = plt.subplots(nrows=2, ncols=1)
        # Display the image
        ax[0].imshow(image, cmap='gray')
        ax[1].imshow(image, cmap='gray')
        for box, label in zip(targets["boxes"], targets["labels"]):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="g",
                                     facecolor='none')
            ax[0].add_patch(rect)
        edge_colors = ["r", "y", "b", "g"]
        for box, label, binary_label in zip(predictions["boxes"], predictions["labels"], prediction_eval_labels):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=edge_colors[binary_label],
                                     facecolor='none')
            ax[1].add_patch(rect)
        ax[0].set_title("Ground truth")
        ax[1].set_title("Predictions (G - correct, R - wrong)")
        fig.subplots_adjust(hspace=.3)
        fig.savefig(f"../visualizations/{checkpoint}/{id}_{suffix}.png", dpi=300)
