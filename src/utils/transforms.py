import torchvision.transforms.functional as F


class PadToSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img_tensor):
        padding = [0, 0, self.size[1] - img_tensor.size(2), self.size[0] - img_tensor.size(1)]
        return F.pad(img_tensor, padding, padding_mode='constant', fill=0)
