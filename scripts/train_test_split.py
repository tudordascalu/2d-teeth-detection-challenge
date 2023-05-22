"""
Split data into training and testing.
"""
import numpy as np
from sklearn.model_selection import train_test_split


def get_train_validation_test_splits(x, p_val, p_test):
    n = len(x)
    x_train, x_test = train_test_split(x, test_size=int(n * p_test))
    x_train, x_val = train_test_split(x_train, test_size=int(n * p_val))
    return x_train, x_val, x_test


if __name__ == "__main__":
    p_val = 0.1
    p_test = 0.2
    y_quadrant = np.load("../data/processed/y_quadrant.npy", allow_pickle=True)
    y_quadrant_enumeration = np.load("../data/processed/y_quadrant_enumeration.npy", allow_pickle=True)
    y_quadrant_enumeration_disease = np.load("../data/processed/y_quadrant_enumeration_disease.npy", allow_pickle=True)
    # Train test split
    y_quadrant_train, y_quadrant_val, y_quadrant_test = get_train_validation_test_splits(y_quadrant, p_val, p_test)
    y_quadrant_enumeration_train, y_quadrant_enumeration_val, y_quadrant_enumeration_test = get_train_validation_test_splits(
        y_quadrant_enumeration, p_val, p_test)
    y_quadrant_enumeration_disease_train, y_quadrant_enumeration_disease_val, y_quadrant_enumeration_disease_test = get_train_validation_test_splits(
        y_quadrant_enumeration_disease, p_val, p_test)
    # Save results
    np.save("../data/final/y_quadrant_train.npy", y_quadrant_train)
    np.save("../data/final/y_quadrant_val.npy", y_quadrant_train)
    np.save("../data/final/y_quadrant_test.npy", y_quadrant_train)
    np.save("../data/final/y_quadrant_enumeration_train.npy", y_quadrant_enumeration_train)
    np.save("../data/final/y_quadrant_enumeration_val.npy", y_quadrant_enumeration_val)
    np.save("../data/final/y_quadrant_enumeration_test.npy", y_quadrant_enumeration_test)
    np.save("../data/final/y_quadrant_enumeration_disease_train.npy", y_quadrant_enumeration_disease_train)
    np.save("../data/final/y_quadrant_enumeration_disease_val.npy", y_quadrant_enumeration_disease_val)
    np.save("../data/final/y_quadrant_enumeration_disease_test.npy", y_quadrant_enumeration_disease_test)
