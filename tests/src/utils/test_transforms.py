from unittest import mock

import torch
from src.utils.transforms import PadToSize, RandomObjectRemover, RandomObjectSwapper, RandomObjectShifter

import unittest
import numpy as np


class TestPadToSize(unittest.TestCase):
    def test_pad_to_size(self):
        pad_to_size = PadToSize((200, 300))

        # Test that an image smaller than target size is correctly padded
        img = torch.ones(3, 100, 100)
        padded_img = pad_to_size(img)
        assert padded_img.shape == (3, 200, 300)

        # Test that an image larger than target size is correctly padded
        img = torch.zeros(3, 250, 350)
        padded_img = pad_to_size(img)
        assert padded_img.shape == (3, 200, 300)

        # Test that an image equal to the target size is not altered
        img = torch.zeros(3, 200, 300)
        padded_img = pad_to_size(img)
        assert padded_img.shape == (3, 200, 300)

        # Test that padding is filled with zeros
        img = torch.ones(3, 100, 150)
        padded_img = pad_to_size(img)
        assert padded_img[0, 199, 299] == 0


class TestRandomObjectRemover(unittest.TestCase):

    def test_init(self):
        """
        Test that the RandomObjectRemover class initializes correctly
        """
        remover = RandomObjectRemover(p=0.7, max=10)
        self.assertEqual(remover.p, 0.7)
        self.assertEqual(remover.max, 10)

    @mock.patch('numpy.random.choice')
    @mock.patch('numpy.random.rand')
    def test_call_no_removal(self, mock_rand, mock_choice):
        """
        Test the call method
        """
        # Mock random values
        mock_rand.return_value = .6  # np.random.rand() always returns 0.0
        mock_choice.return_value = np.array([1])  # np.random.choice() always returns [0]

        # Initialize
        remover = RandomObjectRemover(p=.5, max=2)  # set probability to 0 so no objects are removed
        centroids = np.array([[1, 2], [3, 4], [3, 4], [3, 4], [3, 4]])
        labels = np.array([0, 1, 2, 3, 4])

        # Perform removal
        centroids_result, labels_result = remover((centroids, labels))

        # Assert
        np.testing.assert_array_equal(centroids_result, centroids)
        np.testing.assert_array_equal(labels_result, labels)

    @mock.patch('numpy.random.choice')
    @mock.patch('numpy.random.rand')
    def test_call_with_one_removal(self, mock_rand, mock_choice):
        """
        Test the call method
        """
        # Mock random values
        mock_rand.return_value = .4  # np.random.rand() always returns 0.0
        mock_choice.return_value = np.array([1])  # np.random.choice() always returns [0]

        # Initialize
        remover = RandomObjectRemover(p=.5, max=2)  # set probability to 0 so no objects are removed
        centroids = np.array([[1, 2], [3, 4], [3, 4], [3, 4], [3, 4]])
        labels = np.array([0, 1, 2, 3, 4])

        # Perform removal
        centroids_result, labels_result = remover((centroids, labels))

        # Assert
        np.testing.assert_array_equal(centroids_result, np.array([[1, 2], [0, 0], [3, 4], [3, 4], [3, 4]]))
        np.testing.assert_array_equal(labels_result, labels)

    @mock.patch('numpy.random.choice')
    @mock.patch('numpy.random.rand')
    def test_call_with_two_removal(self, mock_rand, mock_choice):
        """
        Test the call method
        """
        # Mock random values
        mock_rand.return_value = .4  # np.random.rand() always returns 0.0
        mock_choice.return_value = np.array([1, 3])  # np.random.choice() always returns [0]

        # Initialize
        remover = RandomObjectRemover(p=.5, max=2)  # set probability to 0 so no objects are removed
        centroids = np.array([[1, 2], [3, 4], [3, 4], [3, 4], [3, 4]])
        labels = np.array([0, 1, 2, 3, 4])

        # Perform removal
        centroids_result, labels_result = remover((centroids, labels))

        # Assert
        np.testing.assert_array_equal(centroids_result, np.array([[1, 2], [0, 0], [3, 4], [0, 0], [3, 4]]))
        np.testing.assert_array_equal(labels_result, labels)


class TestRandomObjectSwapper(unittest.TestCase):

    def test_init(self):
        """
        Test that the RandomObjectSwapper class initializes correctly
        """
        swapper = RandomObjectSwapper(p=0.7, max=10)
        self.assertEqual(swapper.p, 0.7)
        self.assertEqual(swapper.max, 10)

    @mock.patch('numpy.random.choice')
    @mock.patch('numpy.random.rand')
    def test_call_no_swap(self, mock_rand, mock_choice):
        """
        Test the call method with no swap
        """
        # Mock
        mock_rand.return_value = 1.0  # np.random.rand() always returns 1.0, so no swap will be performed

        # Initialize
        swapper = RandomObjectSwapper(p=0.5, max=10)
        centroids = np.array([[1, 2], [3, 4], [5, 6]])
        labels = np.array([0, 1, 2])

        # Swap
        centroids_result, labels_result = swapper((centroids, labels))

        # Assert
        np.testing.assert_array_equal(centroids_result, np.array([[1, 2], [3, 4], [5, 6]]))
        np.testing.assert_array_equal(labels_result, np.array([0, 1, 2]))

    @mock.patch('numpy.random.choice')
    @mock.patch('numpy.random.rand')
    @mock.patch('numpy.random.randint')
    def test_call_with_swap(self, mock_randint, mock_rand, mock_choice):
        """
        Test the call method with no swap
        """
        # Mock
        mock_rand.return_value = 0.4
        mock_randint.return_value = 2
        mock_choice.side_effect = [np.array([0, 1]), np.array(
            [2, 3])]  # np.random.choice() will return [0, 1] at first call and [1, 2] at second call

        # Initialize
        swapper = RandomObjectSwapper(p=0.5, max=2)
        centroids = np.array([[1, 2], [3, 4], [4, 5], [5, 6]])
        labels = np.array([0, 1, 2, 3])

        # Swap
        centroids_result, labels_result = swapper((centroids, labels))

        # Assert
        np.testing.assert_array_equal(centroids_result, np.array([[3, 4], [1, 2], [5, 6], [4, 5]]))
        np.testing.assert_array_equal(labels_result, np.array([1, 0, 3, 2]))

    @mock.patch('numpy.random.choice')
    @mock.patch('numpy.random.rand')
    @mock.patch('numpy.random.randint')
    def test_call_with_missing(self, mock_randint, mock_rand, mock_choice):
        """
        Test the call method with no swap
        """
        # Mock
        mock_rand.return_value = 0.4
        mock_randint.return_value = 1
        mock_choice.side_effect = [np.array([2, 3]), np.array(
            [0, 1])]  # np.random.choice() will return [0, 1] at first call and [1, 2] at second call

        # Initialize
        swapper = RandomObjectSwapper(p=0.5, max=2)
        centroids = np.array([[0, 0], [3, 4], [4, 5], [5, 6]])
        labels = np.array([0, 1, 2, 3])

        # Swap
        centroids_result, labels_result = swapper((centroids, labels))

        # Assert
        np.testing.assert_array_equal(centroids_result, np.array([[0, 0], [3, 4], [5, 6], [4, 5]]))
        np.testing.assert_array_equal(labels_result, np.array([0, 1, 3, 2]))


class TestRandomObjectShifter(unittest.TestCase):
    @mock.patch('numpy.random.uniform')
    @mock.patch('numpy.random.choice')
    @mock.patch('numpy.random.randint')
    @mock.patch('numpy.random.rand')
    def test_call_with_shift(self, mock_rand, mock_randint, mock_choice, mock_uniform):
        # Mock
        mock_rand.return_value = 0.0  # np.random.rand() always returns 0.0, so shifting will be performed
        mock_randint.return_value = 1  # only 1 item to be shifted
        mock_choice.return_value = np.array([1])  # index 1 will be shifted
        mock_uniform.return_value = np.array([[2.2, 2.2]])  # the displacement will be [2, 2]

        # Initialize
        shifter = RandomObjectShifter(p=0.5, max_dist=10, max_count=2)
        centroids = np.array([[1, 1], [3, 3], [5, 5]], dtype=np.float32)
        labels = np.array([0, 1, 2])

        # Shift
        centroids_result, labels_result = shifter((centroids, labels))

        # Assert that the second centroid has been shifted by [2, 2]
        expected_centroids = np.array([[1, 1], [5.2, 5.2], [5, 5]], dtype=np.float32)
        np.testing.assert_array_equal(centroids_result, expected_centroids)
        np.testing.assert_array_equal(labels_result, labels)

    @mock.patch('numpy.random.rand')
    def test_call_no_shift(self, mock_rand):
        mock_rand.return_value = 1.0  # np.random.rand() always returns 1.0, so no shift will be performed

        shifter = RandomObjectShifter(p=0.5, max_dist=10, max_count=2)
        centroids = np.array([[1, 1], [3, 3], [5, 5]])
        labels = np.array([0, 1, 2])

        centroids_result, labels_result = shifter((centroids, labels))

        # Assert that centroids and labels remain the same
        np.testing.assert_array_equal(centroids_result, centroids)
        np.testing.assert_array_equal(labels_result, labels)

    @mock.patch('numpy.random.uniform')
    @mock.patch('numpy.random.choice')
    @mock.patch('numpy.random.randint')
    @mock.patch('numpy.random.rand')
    def test_call_all_shifted(self, mock_rand, mock_randint, mock_choice, mock_uniform):
        mock_rand.return_value = 0.0  # np.random.rand() always returns 0.0, so shifting will be performed
        mock_randint.return_value = 3  # all items to be shifted
        mock_choice.return_value = np.array([0, 1, 2])  # all indexes will be shifted
        mock_uniform.return_value = np.array([[2, 2], [2, 2], [2, 2]],
                                             dtype=np.float32)  # the displacement will be [2, 2] for each

        shifter = RandomObjectShifter(p=0.5, max_dist=10, max_count=3)
        centroids = np.array([[1, 1], [3, 3], [5, 5]], dtype=np.float32)
        labels = np.array([0, 1, 2])

        centroids_result, labels_result = shifter((centroids, labels))

        # Assert that all centroids have been shifted by [2, 2]
        expected_centroids = np.array([[3, 3], [5, 5], [7, 7]], dtype=np.float32)
        np.testing.assert_array_equal(centroids_result, expected_centroids)
        np.testing.assert_array_equal(labels_result, labels)


if __name__ == "__main__":
    unittest.main()
