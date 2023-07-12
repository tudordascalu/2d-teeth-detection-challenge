import torch.nn.functional as F


class DiceScore:
    def __init__(self, smooth=1e-5, ignore_background=False):
        self.smooth = smooth
        self.ignore_background = ignore_background

    def __call__(self, prediction, target):
        """
        This function computes the dice score.
         - n represents the number of samples to average the loss over;
         - c represents the number of diseases;

        :param prediction: list of predicted masks for each disease of shape (n, c, w, h)
        :param target: one-hot encoded list of target masks for each disease of shape (n, c, w, h)
        :return: (2 * intersection + smooth) / (intersection + union)
        """
        # Apply softmax to prediction pixel across the "c" classes
        prediction = F.softmax(prediction, dim=1)

        # Ignore the background if the parameter is set to True
        if self.ignore_background:
            prediction = prediction[:, 1:, :, :]
            target = target[:, 1:, :, :]

        # Convert prediction and target tensors to contiguous
        prediction = prediction.contiguous()
        target = target.contiguous()

        # Compute dice score
        intersection = (prediction * target).sum(dim=2).sum(dim=2)
        dice_score = (2. * intersection + self.smooth) / (
                prediction.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)

        return dice_score.mean()


class DiceLoss:
    def __init__(self, smooth=1e-5, ignore_background=False):
        self.dice_score = DiceScore(smooth=smooth, ignore_background=ignore_background)

    def __call__(self, prediction, target):
        """
        This function computes the dice score.
         - n represents the number of samples to average the loss over;
         - c represents the number of diseases;

        :param prediction: list of predicted masks for each disease of shape (n, c, w, h)
        :param target: one-hot encoded list of target masks for each disease of shape (n, c, w, h)
        :return: 1 - dice_score
        """
        return 1 - self.dice_score(prediction, target)
