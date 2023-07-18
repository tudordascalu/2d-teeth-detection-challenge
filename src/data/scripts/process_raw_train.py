"""
Nest annotations inside images.
"""
import json

from tqdm import tqdm


def get_annotations(image_id, X):
    return list(filter(lambda x: x["image_id"] == image_id, X["annotations"]))


def process_annotations(annotations):
    """
    :return: list of annotations with different format for bbox, i.e (x1,y1,x2,y2) instead of (x1,y1,w,h)
    """
    for annotation in annotations:
        x, y, w, h = annotation["bbox"]
        annotation["bbox"] = [x, y, x + w, y + h]
    return annotations


def process_labels(X):
    """
    Nest annotations inside images.
    """
    X_processed = []
    for image in tqdm(X["images"]):
        annotations = get_annotations(image["id"], X)
        annotations_processed = process_annotations(annotations)
        image["annotations"] = annotations_processed
        X_processed.append(image)
    return X_processed


def merge_targets_for_overlapping_boxes(X):
    """
    Collects all diseases assigned to a particular tooth and assigns them to all boxes.
    
    :param X: list of dictionaries representing Dentex samples
    :return: list of dictionaries with extra "category_id_3_list" property 
    """
    # Loop through X
    for sample in X:
        # Collect annotations in new list
        annotations = sample["annotations"]

        # Loop through annotations
        for annotation1 in annotations:
            annotation1["category_id_3_list"] = []
            for annotation2 in annotations:
                if annotation1["category_id_1"] == annotation2["category_id_1"] \
                        and annotation1["category_id_2"] == annotation2["category_id_2"]:
                    annotation1["category_id_3_list"].append(annotation2["category_id_3"])
    return X


if __name__ == "__main__":
    # Load data
    with open("data/raw/training_data/quadrant/train_quadrant.json") as f:
        train_quadrant = json.load(f)
    with open("data/raw/training_data/quadrant_enumeration/train_quadrant_enumeration.json") as f:
        train_quadrant_enumeration = json.load(f)
    with open("data/raw/training_data/quadrant_enumeration_disease/train_quadrant_enumeration_disease.json") as f:
        train_quadrant_enumeration_disease = json.load(f)

    # Process labels
    train_quadrant_processed = process_labels(train_quadrant)
    train_quadrant_enumeration_processed = process_labels(train_quadrant_enumeration)
    train_quadrant_enumeration_disease_processed = process_labels(train_quadrant_enumeration_disease)

    # Merge boxes corresponding to the same teeth
    train_quadrant_enumeration_disease_processed = merge_targets_for_overlapping_boxes(
        train_quadrant_enumeration_disease_processed
    )

    # Save labels
    with open("data/processed/train_quadrant.json", 'w') as f:
        json.dump(train_quadrant_processed, f, indent=4)
    with open("data/processed/train_quadrant_enumeration.json", 'w') as f:
        json.dump(train_quadrant_enumeration_processed, f, indent=4)
    with open("data/processed/train_quadrant_enumeration_disease.json", 'w') as f:
        json.dump(train_quadrant_enumeration_disease_processed, f, indent=4)
