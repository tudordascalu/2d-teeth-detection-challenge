"""
This script encapsulate the entire detection pipeline.
"""
import json

import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.transforms import InterpolationMode, transforms
from tqdm import tqdm

from src.model.faster_rcnn.faster_rcnn import FasterRCNN
from src.model.unet.unet import UNet
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
        self.unet = UNet.load_from_checkpoint(
            f"checkpoints/unet/version_63/checkpoints/epoch=epoch=35-val_loss=val_loss=0.20.ckpt")
        self.unet.eval()
        # self.vgg = Vgg.load_from_checkpoint(
        #     f"checkpoints/vgg/version_10/checkpoints/epoch=epoch=31-val_loss=val_f1=0.38.ckpt")
        # self.vgg.eval()

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
            for box in boxes_1:
                # Crop image
                box_int = torch.tensor(box, dtype=torch.int64)

                # image_crop = image.repeat(3, 1, 1)[:, box_int[1]:box_int[3], box_int[0]:box_int[2]]
                image_crop = image[:, box_int[1]:box_int[3], box_int[0]:box_int[2]]
                image_crop = self.transform(image_crop)

                # Get labels
                prediction = self.unet(image_crop.unsqueeze(0).to(self.device))
                labels = torch.where(F.sigmoid(prediction["classification"][0]) > .5)[0].detach().cpu().tolist()

                # Save labels
                labels_2.append(labels)

            # Extract affected teeth
            boxes_2 = []
            scores_2 = []
            category_id_1_2 = []
            category_id_2_2 = []
            category_id_3_2 = []
            for box, score, category_id_1, category_id_2, labels in zip(boxes_1,
                                                                        scores_1,
                                                                        category_id_1_acc,
                                                                        category_id_2_acc,
                                                                        labels_2):
                if 4 not in labels:
                    for category_id_3 in labels:
                        boxes_2.append(box.to("cpu").tolist())
                        category_id_1_2.append(category_id_1.item())
                        category_id_2_2.append(category_id_2.item())
                        category_id_3_2.append(category_id_3)
                        scores_2.append(score.item())

            # Process boxes from x1y1x2y2 to xywh (COCO) format
            boxes_2 = [[box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in boxes_2]

            # Append predictions
            for box, score, category_id_1, category_id_2, category_id_3 in zip(boxes_2, scores_2,
                                                                               category_id_1_2,
                                                                               category_id_2_2,
                                                                               category_id_3_2):
                predictions.append(dict(bbox=box, category_id_1=category_id_1, category_id_2=category_id_2,
                                        category_id_3=category_id_3, score=score, image_id=image_id))

        print("am ajuns aici")
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
