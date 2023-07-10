import unittest

import numpy as np
from torchvision.transforms import transforms

from src.data.tooth_segmentation_dataset import ToothSegmentationDataset
from src.utils.transforms import SquarePad


class TestToothSegmentationDataset(unittest.TestCase):
    def setUp(self):
        self.X = np.load(
            "../../assets/y_quadrant_enumeration_disease_with_healthy_samples_and_segmentation_unpacked_val.npy",
            allow_pickle=True)

    def test_len(self):
        dataset = ToothSegmentationDataset(dataset=self.X, data_dir="../../../data")
        self.assertEqual(len(dataset), len(self.X))

    def test_get_item_without_mask(self):
        # Load data
        dataset = ToothSegmentationDataset(dataset=self.X, data_dir="../../../data")
        sample = dataset[0]
        image, mask, label = sample["image"], sample["mask"], sample["label"]

        # Checks
        self.assertEqual(label, 4)
        self.assertTrue(image.min() >= 0 and image.max() <= 1)
        self.assertEquals(mask.unique().tolist(), [0, 1])
        self.assertEqual(image.shape, (3, 331, 85))
        self.assertEqual(mask.shape, (3, 331, 85))

    def test_get_item_with_mask(self):
        # Load data
        dataset = ToothSegmentationDataset(dataset=self.X, data_dir="../../../data")
        sample = dataset[1]
        image, mask, label = sample["image"], sample["mask"], sample["label"]

        # Checks
        self.assertEqual(label, 3)
        self.assertTrue(image.min() >= 0 and image.max() <= 1)
        self.assertEquals(mask.unique().tolist(), [0, 1])
        self.assertEqual(image.shape, (3, 276, 253))
        self.assertEqual(mask.shape, (3, 276, 253))
        self.assertEquals(mask.unique().tolist(), [0, 1])

    def test_get_item_cropping(self):
        # Load data
        transform_input = transforms.Compose([
            SquarePad(),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR)
        ])
        transform_target = transforms.Compose([
            SquarePad(),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.NEAREST)
        ])
        dataset = ToothSegmentationDataset(dataset=self.X,
                                           data_dir="../../../data",
                                           transform_input=transform_input,
                                           transform_target=transform_target)
        sample = dataset[1]
        image, mask, label = sample["image"], sample["mask"], sample["label"]

        # Checks
        self.assertEqual(label, 3)
        self.assertTrue(image.min() >= 0 and image.max() <= 1)
        self.assertEquals(mask.unique().tolist(), [0, 1])
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertEqual(mask.shape, (3, 224, 224))
        self.assertEquals(mask.unique().tolist(), [0, 1])


if __name__ == '__main__':
    unittest.main()
