import torch
from torchvision.ops import nms


class UniqueClassNMSProcessor:
    def __init__(self, iou_threshold=.7):
        self.iou_threshold = iou_threshold

    def __call__(self, output):
        """
        Ensures that each class is assigned to at most one box and there are no two boxes with an overlap higher than "iou_threshold"

        :param output: dictionary including "boxes", "labels", "scores" keys that could be the output of object detector
        :return: output_processed
        """
        boxes = output["boxes"]
        labels = output["labels"]
        scores = output["scores"]

        # Perform NMS on boxes irrespective of their classes
        indices = nms(torch.tensor(boxes), torch.tensor(scores), self.iou_threshold)

        # Use these indices to get the intermediate boxes, labels, and scores:
        boxes_processed = boxes[indices]
        labels_processed = labels[indices]
        scores_processed = scores[indices]

        # Create a dictionary to store the highest scoring box for each class
        best_boxes = {}

        # Loop through all the boxes
        for box, label, score in zip(boxes_processed, labels_processed.tolist(), scores_processed):
            # If we haven't seen this class yet or this box has a higher score
            # than the previous highest scoring box, store this box
            if label not in best_boxes or score > best_boxes[label][1]:
                best_boxes[label] = (box, score)

        # Process output considering best boxes
        output_processed = dict(boxes=[], labels=[], scores=[])
        for label, (box, score) in best_boxes.items():
            output_processed["boxes"].append(box)
            output_processed["labels"].append(label)
            output_processed["scores"].append(score)

        # At this point, best_boxes should contain the highest scoring box for each class
        return output_processed
