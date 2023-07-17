"""
Split data into training and testing.
"""
import numpy as np
from sklearn.model_selection import train_test_split


def get_train_validation_test_splits(x, p_val, p_test, y=None):
    n = len(x)
    if y is None:
        # Perform split with no stratification
        x_train, x_test = train_test_split(x, test_size=int(n * p_test))
        x_train, x_val = train_test_split(x_train, test_size=int(n * p_val))
    else:
        # Perform split with  stratification
        x_train, x_test, y_train, _ = train_test_split(x, y, test_size=int(n * p_test), stratify=y)
        x_train, x_val = train_test_split(x_train, test_size=int(n * p_val), stratify=y_train)
    return x_train, x_val, x_test


if __name__ == "__main__":
    p_val = 0.1
    p_test = 0.2
    # y_quadrant = np.load("../data/processed/y_quadrant.npy", allow_pickle=True)
    # y_quadrant_enumeration = np.load("../data/processed/y_quadrant_enumeration.npy", allow_pickle=True)
    # y_quadrant_enumeration_disease = np.load("../data/processed/y_quadrant_enumeration_disease.npy", allow_pickle=True)
    y_quadrant_enumeration_disease_unpacked = np.load("../data/processed/y_quadrant_enumeration_disease_unpacked.npy",
                                                      allow_pickle=True)
    y_quadrant_enumeration_disease_unpacked_labels = [sample["annotation"]["category_id_3"] for sample in
                                                      y_quadrant_enumeration_disease_unpacked]
    # y_quadrant_enumeration_disease_with_healthy_samples_unpacked = np.load(
    #     "../data/processed/y_quadrant_enumeration_disease_with_healthy_samples_unpacked.npy", allow_pickle=True)
    # y_quadrant_enumeration_disease_with_healthy_samples_unpacked_labels = [sample["annotation"]["category_id_3"] for
    #                                                                        sample in
    #                                                                        y_quadrant_enumeration_disease_with_healthy_samples_unpacked]
    # Train test split
    # y_quadrant_train, y_quadrant_val, y_quadrant_test = get_train_validation_test_splits(y_quadrant, p_val, p_test)
    # y_quadrant_enumeration_train, y_quadrant_enumeration_val, y_quadrant_enumeration_test = get_train_validation_test_splits(
    #     y_quadrant_enumeration, p_val, p_test)
    # y_quadrant_enumeration_disease_train, y_quadrant_enumeration_disease_val, y_quadrant_enumeration_disease_test = get_train_validation_test_splits(
    #     y_quadrant_enumeration_disease, p_val, p_test)
    y_quadrant_enumeration_disease_unpacked_train, y_quadrant_enumeration_disease_unpacked_val, y_quadrant_enumeration_disease_unpacked_test = get_train_validation_test_splits(
        y_quadrant_enumeration_disease_unpacked, p_val, p_test, y_quadrant_enumeration_disease_unpacked_labels)
    # y_quadrant_enumeration_disease_with_healthy_samples_unpacked_train, y_quadrant_enumeration_disease_with_healthy_samples_unpacked_val, y_quadrant_enumeration_disease_with_healthy_samples_unpacked_test = get_train_validation_test_splits(
    #     y_quadrant_enumeration_disease_with_healthy_samples_unpacked, p_val, p_test,
    #     y=y_quadrant_enumeration_disease_with_healthy_samples_unpacked_labels)

    # Save results
    # np.save("../data/final/y_quadrant_train.npy", y_quadrant_train)
    # np.save("../data/final/y_quadrant_val.npy", y_quadrant_train)
    # np.save("../data/final/y_quadrant_test.npy", y_quadrant_train)
    # np.save("../data/final/y_quadrant_enumeration_train.npy", y_quadrant_enumeration_train)
    # np.save("../data/final/y_quadrant_enumeration_val.npy", y_quadrant_enumeration_val)
    # np.save("../data/final/y_quadrant_enumeration_test.npy", y_quadrant_enumeration_test)

    # np.save("../data/final/y_quadrant_enumeration_disease_train.npy", y_quadrant_enumeration_disease_train)
    # np.save("../data/final/y_quadrant_enumeration_disease_val.npy", y_quadrant_enumeration_disease_val)
    # np.save("../data/final/y_quadrant_enumeration_disease_test.npy", y_quadrant_enumeration_disease_test)

    np.save("../data/final/y_quadrant_enumeration_disease_unpacked_train.npy", y_quadrant_enumeration_disease_unpacked_train)
    np.save("../data/final/y_quadrant_enumeration_disease_unpacked_val.npy", y_quadrant_enumeration_disease_unpacked_val)
    np.save("../data/final/y_quadrant_enumeration_disease_unpacked_test.npy", y_quadrant_enumeration_disease_unpacked_test)

    # np.save("../data/final/y_quadrant_enumeration_disease_with_healthy_samples_unpacked_train.npy",
    #         y_quadrant_enumeration_disease_with_healthy_samples_unpacked_train)
    # np.save("../data/final/y_quadrant_enumeration_disease_with_healthy_samples_unpacked_val.npy",
    #         y_quadrant_enumeration_disease_with_healthy_samples_unpacked_val)
    # np.save("../data/final/y_quadrant_enumeration_disease_with_healthy_samples_unpacked_test.npy",
    #         y_quadrant_enumeration_disease_with_healthy_samples_unpacked_test)
