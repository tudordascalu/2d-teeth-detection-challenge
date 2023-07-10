import numpy as np

if __name__ == "__main__":
    # Load data
    x = np.load("data/final/y_quadrant_enumeration_disease_unpacked_train.npy", allow_pickle=True)

    # Extract disease and tooth labels
    disease = [sample["annotation"]["category_id_3"] for sample in x]
    tooth = [sample["annotation"]["category_id_2"] for sample in x]

    # Only keep deep caries and apical lesions
    x_filtered = list(filter(lambda a: a["annotation"]["category_id_3"] in [2, 3], x))

    # Select 400 samples with caries
    x_caries = np.array(list(filter(lambda a: a["annotation"]["category_id_3"] in [1], x)))[:400]

    x_filtered.extend(x_caries)

    # Save filtered
    np.save("data/final/y_quadrant_enumeration_disease_unpacked_train_filtered.npy", x_filtered)
