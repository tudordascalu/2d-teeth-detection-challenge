import numpy as np
from PIL import ImageDraw
from torchvision.io import read_image
from torchvision.transforms import transforms
from tqdm import tqdm

if __name__ == "__main__":
    # Load data
    y = np.load(f"data/processed/y_quadrant_enumeration_disease.npy", allow_pickle=True)
    colors = ["red", "green", "blue", "yellow", "orange"]

    # Helpers
    to_pil = transforms.ToPILImage()

    for sample in tqdm(y, total=len(y)):
        file_name = sample['file_name']
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
        image_pil.save(f"visualizations/quadrant_enumeraton_disease/{file_name}")
