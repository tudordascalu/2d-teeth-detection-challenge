import numpy as np
import torch


class OneHotMultiLabel:
    def __init__(self):
        pass

    def __call__(self, batch_labels, num_classes):
        """
        This function accepts labels from a batch of samples.

        :param labels: list of label lists
        :param num_classes: number of classes in classification problem
        :return: list of shape (1, "num_classes") including at label indexes and 0 in the remaining positions
        """
        one_hot_labels = torch.zeros((len(batch_labels), num_classes), dtype=torch.float32)

        for i, labels in enumerate(batch_labels):
            for label in labels:
                one_hot_labels[i, label] = 1

        return one_hot_labels
