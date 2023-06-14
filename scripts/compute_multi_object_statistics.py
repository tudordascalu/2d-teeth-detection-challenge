"""
Compute mean and standard deviation for inter-tooth distances.
"""
import numpy as np
from tqdm import tqdm

from src.utils.label_encoder import LabelProcessor, LabelEncoder
from src.utils.multi_object_labelling import ObjectCentroidMapper, InterObjectDistanceMapper

if __name__ == "__main__":
    # Load data
    y_quadrant_enumeration_train = np.load("../data/final/y_quadrant_enumeration_train.npy", allow_pickle=True)
    label_processor = LabelProcessor()
    label_encoder = LabelEncoder()
    object_centroid_mapper = ObjectCentroidMapper()
    inter_object_distance_mapper = InterObjectDistanceMapper()
    inter_object_distance_mat_acc = []
    for sample in tqdm(y_quadrant_enumeration_train, total=len(y_quadrant_enumeration_train)):
        boxes = []
        labels = []
        centroids = np.zeros((32, 2), dtype=np.float32)
        for annotation in sample["annotations"]:
            # Process label
            label = label_processor(annotation["category_id_1"], annotation["category_id_2"])
            label = label_encoder.transform([label])[0] - 1  # Subtract 1 as teeth go from 1-32
            centroids[label] = object_centroid_mapper(annotation["bbox"])
        inter_teeth_distance_mat = inter_object_distance_mapper(centroids)
        inter_object_distance_mat_acc.append(inter_teeth_distance_mat)
    # Compute mean, std for inter teeth distances for each coordinate
    inter_object_teeth_distance_mat_acc_masked = np.ma.masked_equal(inter_object_distance_mat_acc, 0)
    inter_object_distance_mat_mean = np.ma.filled(np.ma.mean(inter_object_teeth_distance_mat_acc_masked, axis=0),
                                                  fill_value=0)
    inter_object_distance_mat_std = np.ma.filled(np.ma.mean(inter_object_teeth_distance_mat_acc_masked, axis=0),
                                                 fill_value=0)
    np.save("../data/final/inter_object_distance_mat_mean.npy", inter_object_distance_mat_mean)
    np.save("../data/final/inter_object_distance_mat_std.npy", inter_object_distance_mat_std)
