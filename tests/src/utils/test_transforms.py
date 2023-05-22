import torch

from src.utils.transforms import PadToSize
import matplotlib.pyplot as plt


class TestPadToSize:
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
