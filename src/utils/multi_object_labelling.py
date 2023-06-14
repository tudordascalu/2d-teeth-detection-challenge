import math

import numpy as np
from scipy.stats import norm
from scipy.optimize import linear_sum_assignment


class InterObjectScoreMapper:
    def __init__(self, inter_object_distance_mat_mean, inter_object_distance_mat_std, n_teeth=32, missing_score=-1,
                 diagonal_score=0):
        self.inter_object_distance_mat_mean = inter_object_distance_mat_mean
        self.inter_object_distance_mat_std = inter_object_distance_mat_std
        self.n_teeth = n_teeth
        self.missing_score = missing_score
        self.diagonal_score = diagonal_score

    def __call__(self, centroids, inter_object_distance_mat):
        """

        :param inter_object_distance_mat: np.array of shape (32, 2) featuring labels for each instance
        :param inter_object_distance_mat_mean: np.array of shape (32, 32, 2) featuring mean distances between tooth-tooth pairs
        :param inter_object_distance_mat_std: np.array of shape (32, 32, 2) featuring stds for tooth-tooth pair distances
        :return: np.array of shape (32, 32) denoting tooth-tooth probabilities based on distances
        """
        inter_objects_score_mat = np.zeros((self.n_teeth, self.n_teeth, 1))
        # Navigate through all instances [0, 17]
        for i in range(self.n_teeth):
            for j in range(self.n_teeth):
                inter_object_distance = inter_object_distance_mat[i, j]
                inter_object_distance_mean = self.inter_object_distance_mat_mean[i, j]
                inter_object_distance_std = self.inter_object_distance_mat_std[i, j]
                try:
                    x_score = norm.pdf(inter_object_distance[0], inter_object_distance_mean[0],
                                       inter_object_distance_std[0])
                    y_score = norm.pdf(inter_object_distance[1], inter_object_distance_mean[1],
                                       inter_object_distance_std[1])
                    if math.isnan(x_score) or math.isnan(y_score):
                        raise ValueError("The result of norm.pdf is NaN")
                    score = x_score + y_score
                except:
                    score = 0
                inter_objects_score_mat[i, j] = score
        # Update missing teeth scores
        missing_teeth = np.where((centroids == np.array([0, 0])).all(axis=1))[0]
        inter_objects_score_mat[missing_teeth] = self.missing_score
        inter_objects_score_mat[:, missing_teeth] = self.missing_score
        # Update diagonal scores
        inter_objects_score_mat[np.arange(self.n_teeth), np.arange(self.n_teeth)] = self.diagonal_score
        return inter_objects_score_mat


class MultiObjectCentroidMapper:
    def __init__(self, n_teeth=32):
        self.object_centroid_mapper = ObjectCentroidMapper()
        self.n_teeth = n_teeth

    def __call__(self, boxes, labels):
        object_centroids = np.zeros((32, 2), dtype=np.float32)
        for box, label in zip(boxes, labels):
            object_centroids[label - 1] = self.object_centroid_mapper(box)
        return object_centroids


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


class AssignmentSolver:
    def __init__(self):
        pass

    def __call__(self, cost_matrix):
        """

        :param cost_matrix: 3D or 2D numpy array; in 3D the first dimension denotes the number of cost matrices passed on
        :return: labels for each tooth and new cost matrix where non-assigned jobs per worker are zero
        """
        if len(cost_matrix.shape) == 2:
            row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
            cost_matrix_processed = np.zeros(cost_matrix.shape)
            cost_matrix_processed[row_ind, col_ind] = 1
            return col_ind, cost_matrix_processed
        elif len(cost_matrix.shape) == 3:
            col_ind_acc = []
            cost_matrix_processed_acc = []
            for cost_m in cost_matrix:
                col_ind, cost_matrix_processed = self(cost_m)
                col_ind_acc.append(col_ind)
                cost_matrix_processed_acc.append(cost_matrix_processed)
            return np.array(col_ind_acc), np.array(cost_matrix_processed_acc)
        else:
            raise ValueError("The parameter cost_matrix should be either 2D or 3D.")
