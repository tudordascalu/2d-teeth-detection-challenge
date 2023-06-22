import numpy as np
from torchvision.io import read_image
from tqdm import tqdm

from src.visualization.scripts.draw_image_with_boxes import DrawImageWithBoxes

if __name__ == "__main__":
    # Set configuration
    data_type = "quadrant_enumeration_disease"
    colors = np.load("data/assets/colors.npy")

    y = np.load(f"data/processed/y_{data_type}_with_healthy_samples_processed.npy", allow_pickle=True)
    draw_image_with_boxes = DrawImageWithBoxes(colors=np.array(["r", "g", "b", "y", "orange", "purple"]))

    for sample in tqdm(y[:3], total=len(y[:3])):
        image = read_image(f"data/raw/training_data/quadrant_enumeration_disease/xrays/{sample['file_name']}")[0]
        boxes = [annotation["bbox"] for annotation in sample["annotations"]]
        labels = [annotation["category_id_3"] for annotation in sample["annotations"]]
        fig = draw_image_with_boxes(image, boxes, labels)
        fig.savefig(f"visualizations/{sample['file_name']}", dpi=300)
