import numpy as np


class LabelEncoder:
    def __init__(self):
        # Set possible tooth labels considering that 0 is used for background and we have 1-8 teeth for each quadrant
        labels = np.concatenate(
            ([0], (np.arange(1, 9) + 10)[::-1], np.arange(1, 9), (np.arange(1, 9) + 30)[::-1],
             np.arange(1, 9) + 20)).tolist()
        self.fit(labels)

    def fit(self, labels):
        self.encoder = {label: i for i, label in enumerate(labels)}

    def transform(self, labels):
        return np.array([self.encoder[label] for label in labels])

    def inverse_transform(self, encoded_labels):
        decoder = {i: label for label, i in self.encoder.items()}
        return np.array([decoder[encoded_label] for encoded_label in encoded_labels])


class LabelProcessor:
    def __init__(self):
        pass

    def __call__(self, category_id_1, category_id_2):
        """

        :param category_id_1: int in [0, 3]
        :param category_id_2: int in [0, 7]
        :return: <category_id> * 10 + <category_id_2> + 1
        """
        return category_id_1 * 10 + category_id_2 + 1
