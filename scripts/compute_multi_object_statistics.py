"""
Compute mean and standard deviation for inter-tooth distances.
"""
import numpy as np
from tqdm import tqdm

from src.utils.label_encoder import LabelProcessor, LabelEncoder


class ObjectCentroidMapper:
    def __init__(self):
        pass

    def __call__(self, box):
        """

        :param box: list with 4 elements corresponding to x1y1x2y2
        :return: (x1+x2)/2, (y1+y2)/2
        """
        return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


class InterObjectDistanceMapper:
    def __init__(self):
        pass

    def __call__(self, centroids):
        """
        :param centroids: list of centroids of shape (32, 2)
        :return: matrix including inter teeth distances for each axis
        """
        inter_teeth_distance_mat = np.zeros((32, 32, 2), dtype=np.float32)
        labels = np.where(~(centroids == [0, 0]).all(axis=1))
        for label, centroid in enumerate(centroids):
            inter_teeth_distance_mat[label, labels] = centroids[labels] - centroid
        return inter_teeth_distance_mat


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
