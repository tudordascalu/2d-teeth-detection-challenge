import numpy as np
import torchvision.transforms.functional as F


class PadToSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img_tensor):
        padding = [0, 0, self.size[1] - img_tensor.size(2), self.size[0] - img_tensor.size(1)]
        return F.pad(img_tensor, padding, padding_mode='constant', fill=0)


class RandomObjectRemover:
    def __init__(self, p=0.5, max=5):
        """

        :param p: probability of removing any objects
        :param max: controls the maximum number of teeth that can be removed
        """
        self.p = p
        self.max = max

    def __call__(self, x):
        centroids, labels = x
        i_labels = np.arange(0, len(labels))

        # Check if we should remove any
        if np.random.rand() >= self.p:
            return centroids, labels

        # Remove up to "max" objects
        n = np.random.randint(1, self.max + 1)
        i_labels_remove = np.random.choice(i_labels, size=n, replace=False)
        centroids[i_labels_remove] = [0, 0]

        return centroids, labels


# TODO: consider performing swaps to neighboring objects
class RandomObjectSwapper:
    def __init__(self, p, max=2):
        """

        :param p: probability of swapping any objects
        :param max: controls the maximum number of teeth that can be swapped
        """
        self.p = p
        self.max = max

    def __call__(self, x):
        centroids, labels = x
        n_objects = len(labels)

        # Check if we should swap any
        if np.random.rand() >= self.p:
            return centroids, labels

        # Add missing instances to previously swapped list
        i_labels_swapped = np.where((centroids == np.array([0, 0])).all(axis=1))[0].tolist()

        # Perform up to "n" swaps
        n = min(np.random.randint(1, self.max + 1), int((n_objects + 1) / 2))
        i = 0
        while i < n:
            i_labels = np.arange(0, len(labels))
            i_labels_not_swapped = i_labels[~np.isin(i_labels, i_labels_swapped)]
            i_label_1, i_label_2 = np.random.choice(i_labels_not_swapped, size=2)

            # Swap labels
            centroids, labels = self._swap(centroids, labels, i_label_1, i_label_2)
            i_labels_swapped.extend([i_label_1, i_label_2])
            i += 1

        return centroids, labels

    @staticmethod
    def _swap(centroids, labels, i_label_1, i_label_2):
        """
        :param centroids: array of centroids of shape (n, 3)
        :param labels: array of labels of shape (n,), where each element corresponds to a centroid
        :return: centroids and labels, with swapped teeth
        """
        labels[i_label_1], labels[i_label_2] = labels[i_label_2], labels[i_label_1]
        centroids[[i_label_1, i_label_2]] = centroids[[i_label_2, i_label_1]]
        return centroids, labels


class RandomObjectShifter:
    def __init__(self, p, max_dist=10, max_count=5):
        """
        :param p: probability of shifting any objects
        :param max_dist: controls the maximum distance ("px") that an object can be shifted by in x, y directions
        :param max_count: controls the maximum number of teeth that can be shifted
        """
        self.p = p
        self.max_dist = max_dist
        self.max_count = max_count

    def __call__(self, x):
        centroids, labels = x

        # Check if we should shift any
        if np.random.rand() >= self.p:
            return centroids, labels

        # Leave missing instances alone
        i_labels = np.arange(0, len(labels))
        i_labels_missing = np.where((centroids == np.array([0, 0])).all(axis=1))[0].tolist()
        i_labels = i_labels[~np.isin(i_labels, i_labels_missing)]

        # Compute objects to be shifted
        n = min(np.random.randint(1, self.max_count + 1), len(i_labels) + 1)
        i_labels_shift = np.random.choice(i_labels, size=n, replace=False)

        # Compute displacement
        centroid_displacement = np.random.uniform(low=-self.max_dist, high=self.max_dist, size=(n, 2)).astype(
            np.float32)

        # Shift centroids
        centroids[i_labels_shift] += centroid_displacement

        return centroids, labels
