import numpy as np
from sklearn.model_selection import train_test_split


def get_train_validation_test_splits(x, p_val, p_test, y=None):
    n = len(x)
    if y is None:
        # Perform split with no stratification
        x_train, x_test = train_test_split(x, test_size=int(n * p_test), random_state=42)
        x_train, x_val = train_test_split(x_train, test_size=int(n * p_val), random_state=42)
    else:
        # Perform split with  stratification
        x_train, x_test, y_train, _ = train_test_split(x, y, test_size=int(n * p_test), stratify=y, random_state=42)
        x_train, x_val = train_test_split(x_train, test_size=int(n * p_val), stratify=y_train, random_state=42)
    return x_train, x_val, x_test


if __name__ == "__main__":
    # Set split size
    p_val = 0.15
    p_test = 0.15

    # Load data
    x = np.load("data/processed/y_quadrant_enumeration_disease_unpacked.npy", allow_pickle=True)

    # Extract combination of tooth-disease labels for stratification
    tooth_disease_combination = [
        sample["annotation"]["category_id_3"] * 10 + sample["annotation"]["category_id_2"] + 1
        for sample in x
    ]

    # Compute train test split
    x_train, x_val, x_test = get_train_validation_test_splits(x, p_val, p_test, tooth_disease_combination)

    # Save results
    for split, x_split in zip(["train", "val", "test"], [x_train, x_val, x_test]):
        np.save(f"data/final/y_quadrant_enumeration_disease_unpacked_{split}.npy", x_split)
