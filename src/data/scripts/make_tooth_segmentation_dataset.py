import glob

import numpy as np

if __name__ == "__main__":
    # Select healthy samples from "n" patients
    n = 5
    X_healthy = np.load("data/processed/y_quadrant_enumeration_disease_with_healthy_samples_unpacked.npy",
                        allow_pickle=True)
    diseases_healthy = np.array([sample["annotation"]["category_id_3"] for sample in X_healthy])
    X_healthy = X_healthy[np.where(diseases_healthy == 4)]
    file_names_healthy = np.array([sample["file_name"] for sample in X_healthy])
    X_healthy_selection = []

    for file_name in file_names_healthy[:n]:
        X_healthy_sample = X_healthy[file_names_healthy == file_name]
        X_healthy_selection.extend(X_healthy_sample)

    # # Loop through all tooth - category combinations and select "n" samples
    # for quadrant in np.arange(0, 4):
    #     for tooth in np.arange(0, 8):
    #         X_quadrant_tooth = X_healthy[np.where((quadrants == quadrant) & (teeth == tooth))]
    #         X_healthy_selection.extend(X_quadrant_tooth[:n])

    # Select all samples with diseases and masks available
    X = np.load("data/processed/y_quadrant_enumeration_disease_unpacked.npy", allow_pickle=True)
    diseases = np.array([sample["annotation"]["category_id_3"] for sample in X])
    teeth = np.array([sample["annotation"]["category_id_2"] for sample in X])
    quadrants = np.array([sample["annotation"]["category_id_1"] for sample in X])
    file_names = np.array([sample["file_name"] for sample in X])

    masks = list(map(lambda x: x.split("/")[-1].split(".")[0], glob.glob("data/final/masks/*")))

    X_disease_selection = []
    for mask_name in masks:
        disease = mask_name.split("_")[0]
        tooth = mask_name.split("_")[1]
        quadrant = mask_name.split("_")[2]
        id = mask_name.split('_')[4]

        # Find sample
        sample = X[
            np.where(
                (quadrants == int(quadrant)) &
                (teeth == int(tooth)) &
                (diseases == int(disease)) &
                (file_names == f"train_{id}.png")
            )
        ][0]

        sample["mask_name"] = mask_name
        X_disease_selection.append(sample)

    X_tooth_segmentation = np.concatenate((X_disease_selection, X_healthy_selection))
    np.save("data/processed/y_quadrant_enumeration_disease_with_healthy_samples_and_segmentation_unpacked.npy",
            X_tooth_segmentation)
