import unittest

import numpy as np
import torch

from src.data.panoramic_dataset import PanoramicDataset


class TestPanoramicDataset(unittest.TestCase):

    def setUp(self):
        # Initialize test dataset and image directory
        self.dataset = np.load("../../../data/processed/y_quadrant_enumeration_disease.npy", allow_pickle=True)
        self.image_dir = "../../../data/raw/training_data/quadrant_enumeration_disease/xrays"

    def test_getitem(self):
        # Create an instance of PanoramicDataset
        dataset = PanoramicDataset(self.dataset, self.image_dir)
        # Test __getitem__ with an index
        idx = 0
        image, targets = dataset[idx]
        # Assert the output types and shapes
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(targets, dict)
        self.assertEqual(targets["labels"].tolist(), [32, 24, 31, 19, 20, 21, 14,  7, 22, 23, 18, 17, 25])
        self.assertEqual(image.shape, (1, 1316, 2744))

    def test_len(self):
        # Create an instance of PanoramicDataset
        dataset = PanoramicDataset(self.dataset, self.image_dir)
        # Assert the expected length
        self.assertEqual(len(dataset), len(self.dataset))


if __name__ == '__main__':
    unittest.main()
