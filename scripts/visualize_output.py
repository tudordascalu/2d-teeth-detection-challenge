import glob
import os
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt, patches
from torchvision.ops import box_iou
from tqdm import tqdm

from src.utils.processors import UniqueClassNMSProcessor


def assign_prediction_binary_labels(targets, predictions, iou_threshold=0.5):
    target_boxes = targets["boxes"]
    target_labels = np.array(targets["labels"])
    prediction_boxes = predictions["boxes"]
    prediction_labels = np.array(predictions["labels"])
    prediction_binary_labels = np.zeros(len(prediction_labels))
    iou_mat = box_iou(torch.tensor(target_boxes), torch.tensor(prediction_boxes))
    for i, (target_box, target_label) in enumerate(zip(target_boxes, target_labels)):
        try:
            j = np.where(prediction_labels == target_label)[0][0]
            iou = iou_mat[i, j]
            prediction_binary_labels[j] = int(iou > iou_threshold)
        except:
            pass
    return prediction_binary_labels


if __name__ == "__main__":
    colors = np.load("../data/assets/colors.npy")
    checkpoint = "version_3__epoch=76-step=2156"
    if not os.path.exists(f"../visualizations/{checkpoint}"):
        os.mkdir(f"../visualizations/{checkpoint}")
    output_path = f"../output/{checkpoint}"
    data_path = "../data/raw/training_data"
    ids = [path.split("/")[-1] for path in glob.glob(f"{output_path}/*")]
    unique_class_nms_processor = UniqueClassNMSProcessor(iou_threshold=.5)
    for id in tqdm(ids, total=len(ids)):
        # Load data
        image = plt.imread(f"{data_path}/quadrant_enumeration/xrays/{id}.png")[:, :, 0]
        with open(f"{output_path}/{id}/targets.npy", "rb") as file:
            targets = pickle.load(file)
        with open(f"{output_path}/{id}/predictions.npy", "rb") as file:
            predictions = pickle.load(file)
        # Overlap targets and boxes on image
        predictions = unique_class_nms_processor(predictions)
        prediction_binary_labels = assign_prediction_binary_labels(targets, predictions, iou_threshold=.6)
        fig, ax = plt.subplots(nrows=2, ncols=1)
        # Display the image
        ax[0].imshow(image, cmap='gray')
        ax[1].imshow(image, cmap='gray')
        for box, label in zip(targets["boxes"], targets["labels"]):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="g",
                                     facecolor='none')
            ax[0].add_patch(rect)
        for box, label, binary_label in zip(predictions["boxes"], predictions["labels"], prediction_binary_labels):
            if binary_label == 1:
                edge_color = "g"
            else:
                edge_color = "r"
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=edge_color,
                                     facecolor='none')
            ax[1].add_patch(rect)
        ax[0].set_title("Ground truth")
        ax[1].set_title("Predictions (G - correct, R - wrong)")
        fig.subplots_adjust(hspace=.3)
        fig.savefig(f"../visualizations/{checkpoint}/{id}.png", dpi=300)
