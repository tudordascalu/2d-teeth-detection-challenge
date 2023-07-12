import json

import pandas as pd
from matplotlib import pyplot as plt
from torchvision.io import read_image
from tqdm import tqdm

from src.visualization.scripts.draw_image_with_boxes import DrawImageWithBoxes

if __name__ == "__main__":
    # Load data
    with open("output/predictions.json") as f:
        predictions = json.load(f)
    validation_meta = pd.read_excel("data/raw/validation_data/validation_data.xlsx")

    # Helpers
    draw_image_with_boxes = DrawImageWithBoxes(colors=["r", "g", "b", "y", "orange"])

    # Group by "image_id" and convert each group to a dictionary
    df = pd.DataFrame(predictions)
    predictions_processed = [
        dict(group[["bbox", "category_id_3"]].to_dict('list'), image_id=image_id)
        for image_id, group in df.groupby("image_id")
    ]

    # Draw and save figures
    for sample in tqdm(predictions_processed, total=len(predictions_processed)):
        file_name = validation_meta[validation_meta["Image id"] == sample["image_id"]]["File name"].values[0]
        image = read_image(f"data/raw/validation_data/quadrant_enumeration_disease/xrays/{file_name}")[0]
        boxes = [[box[0], box[1], box[2] + box[0], box[3] + box[1]] for box in sample["bbox"]]
        labels = sample["category_id_3"]
        fig = draw_image_with_boxes(image, boxes, labels)
        plt.title(f"Red - Impacted; Green - Caries; Blue - Periapical lesion; Yellow - deep caries")
        fig.savefig(f"visualizations/predictions/{file_name}")
