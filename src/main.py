"""
This script encapsulate the entire detection pipeline.
"""
import json

import pandas as pd
import torch
from torch.nn.functional import softmax
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, InterpolationMode, CenterCrop, transforms
from tqdm import tqdm

from src.model.faster_rcnn.faster_rcnn import FasterRCNN
from src.model.resnet.resnet import ResNet
from src.model.vgg.vgg import Vgg
from src.utils.label_encoder import LabelEncoder
from src.utils.processors import UniqueClassNMSProcessor
from src.utils.transforms import SquarePad


class PanoramicProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load data
        self.dataset = pd.read_excel("data/raw/validation_data/validation_data.xlsx")

        # Load models
        self.faster_rcnn = FasterRCNN.load_from_checkpoint(
            f"checkpoints/faster_rcnn/version_3/checkpoints/epoch=epoch=88-val_loss=val_loss=0.81.ckpt")
        self.faster_rcnn.eval()
        self.vgg = Vgg.load_from_checkpoint(
            f"checkpoints/vgg/version_10/checkpoints/epoch=epoch=31-val_loss=val_f1=0.38.ckpt")
        self.vgg.eval()
        # Utilities
        self.encoder = LabelEncoder()
        self.transform = transforms.Compose([
            SquarePad(),
            transforms.Resize(224, interpolation=InterpolationMode.BILINEAR)
        ])
        self.unique_class_nms_processor = UniqueClassNMSProcessor(iou_threshold=.75)

    def __call__(self):
        # Process patients
        predictions = []
        for i, row in tqdm(self.dataset.iterrows(), total=len(self.dataset)):
            # Load data
            image_id = row["Image id"]
            image = self._read_image(f"data/raw/validation_data/quadrant_enumeration_disease/xrays/{row['File name']}")

            # Detect teeth
            with torch.no_grad():
                output_1 = self.faster_rcnn(image.unsqueeze(0).to(self.device))[0]
            output_1 = self.unique_class_nms_processor(output_1)

            boxes_1 = torch.stack(output_1["boxes"])
            labels_1 = output_1["labels"]
            scores_1 = torch.tensor(output_1["scores"])

            # Decode labels 1
            labels_1 = self.encoder.inverse_transform(labels_1)
            category_id_1_acc = torch.tensor([int(label / 10) for label in labels_1], dtype=torch.int32)
            category_id_2_acc = torch.tensor([int(label % 10) - 1 for label in labels_1], dtype=torch.int32)

            # Classify teeth
            labels_2 = []
            p_2 = []
            for box in boxes_1:
                # Crop image
                box_int = torch.tensor(box, dtype=torch.int64)
                image_crop = image.repeat(3, 1, 1)[:, box_int[1]:box_int[3], box_int[0]:box_int[2]]
                image_crop = self.transform(image_crop)
                # Get label
                output = softmax(self.vgg(image_crop.unsqueeze(0).to(self.device))[0])
                label_2 = output.argmax()
                p = output[label_2]
                labels_2.append(label_2)
                p_2.append(p)

            # Extract affected teeth
            labels_2 = torch.tensor(labels_2, dtype=torch.int64)
            p_2 = torch.tensor(p_2, dtype=torch.float32)
            indices = torch.where(labels_2 != 4)[0]
            boxes_2 = boxes_1[indices].to("cpu").tolist()
            scores_2 = scores_1[indices].to("cpu").tolist()
            category_id_1_acc = category_id_1_acc[indices].tolist()
            category_id_2_acc = category_id_2_acc[indices].tolist()
            category_id_3_acc = labels_2[indices].tolist()
            p_2 = p_2[indices].tolist()

            # Process boxes from x1y1x2y2 to xywh (COCO) format
            boxes_2 = [[box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in boxes_2]

            # Append predictions
            for box, score, category_id_1, category_id_2, category_id_3, p in zip(boxes_2, scores_2, category_id_1_acc,
                                                                                  category_id_2_acc, category_id_3_acc,
                                                                                  p_2):
                predictions.append(dict(bbox=box, category_id_1=category_id_1, category_id_2=category_id_2,
                                        category_id_3=category_id_3, score=score, image_id=image_id, p_category_id_3=p))
        with open("output/predictions.json", 'w') as f:
            json.dump(predictions, f)

    @staticmethod
    def _read_image(path):
        image = read_image(path)
        image = image[0].unsqueeze(0)
        image = image / 255.0
        return image


if __name__ == "__main__":
    panoramic_processor = PanoramicProcessor()
    panoramic_processor()
