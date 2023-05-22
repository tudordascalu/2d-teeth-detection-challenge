"""
Nest annotations inside images.
"""
import json

import numpy as np
from tqdm import tqdm


def get_annotations(image_id, y):
    return list(filter(lambda x: x["image_id"] == image_id, y["annotations"]))


def process_annotations(annotations):
    """
    :param annotations:
    :return:
    """
    for annotation in annotations:
        x, y, w, h = annotation["bbox"]
        annotation["bbox"] = [x, y, x + w, y + h]
    return annotations


def process_labels(y):
    """
    Nest annotations inside images.
    """
    y_processed = []
    for image in tqdm(y["images"]):
        annotations = get_annotations(image["id"], y)
        annotations_processed = process_annotations(annotations)
        image["annotations"] = annotations_processed
        y_processed.append(image)
    return y_processed


if __name__ == "__main__":
    with open("../data/raw/training_data/quadrant/train_quadrant.json") as f:
        # Load JSON data from file
        y_quadrant = json.load(f)
    with open("../data/raw/training_data/quadrant_enumeration/train_quadrant_enumeration.json") as f:
        # Load JSON data from file
        y_quadrant_enumeration = json.load(f)
    with open("../data/raw/training_data/quadrant_enumeration_disease/train_quadrant_enumeration_disease.json") as f:
        # Load JSON data from file
        y_quadrant_enumeration_disease = json.load(f)
    # Process labels
    y_quadrant_processed = process_labels(y_quadrant)
    y_quadrant_enumeration_processed = process_labels(y_quadrant_enumeration)
    y_quadrant_enumeration_disease_processed = process_labels(y_quadrant_enumeration_disease)
    # Save labels
    np.save("../data/processed/y_quadrant.npy", y_quadrant_processed)
    np.save("../data/processed/y_quadrant_enumeration.npy", y_quadrant_enumeration_processed)
    np.save("../data/processed/y_quadrant_enumeration_disease.npy", y_quadrant_enumeration_disease_processed)
