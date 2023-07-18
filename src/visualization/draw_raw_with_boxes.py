import json
import os

from PIL import ImageDraw
from torchvision.io import read_image
from torchvision.transforms import transforms
from tqdm import tqdm

if __name__ == "__main__":
    # Load data
    with open("data/processed/train_quadrant_enumeration_disease_healthy.json") as f:
        data = json.load(f)

    # Helpers
    to_pil = transforms.ToPILImage()
    colors = ["red", "green", "blue", "yellow", "orange"]

    for sample in tqdm(data, total=len(data)):
        file_name = sample["file_name"]
        image = read_image(f"data/raw/training_data/quadrant_enumeration_disease/xrays/{file_name}")

        # Convert image to pil draw
        image_pil = to_pil(image)
        draw = ImageDraw.Draw(image_pil)

        # Extract labels and bounding boxes
        boxes = [annotation["bbox"] for annotation in sample["annotations"]]
        labels = [annotation["category_id_3"] for annotation in sample["annotations"]]

        # Draw boxes
        for box, label in zip(boxes, labels):
            draw.rectangle(box, outline=colors[label], width=4)

        # Save image
        if not os.path.exists("visualizations/quadrant_enumeration_disease_healthy"):
            os.mkdir("visualizations/quadrant_enumeration_disease_healthy")
        image_pil.save(f"visualizations/quadrant_enumeration_disease_healthy/{file_name}")
