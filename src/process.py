"""
This script encapsulate the entire detection pipeline.
"""
import glob
import json
import SimpleITK as sitk

import torch
import yaml
from torchvision.transforms import InterpolationMode, transforms

from src.model.faster_rcnn.faster_rcnn import FasterRCNN
from src.model.unet.unet import UNet
from src.utils.label_encoder import LabelEncoder
from src.utils.processors import UniqueClassNMSProcessor
from src.utils.transforms import SquarePad

list_ids = [
    {"height": 1316, "width": 2892, "id": 1, "file_name": "val_15.png"},
    {"height": 1316, "width": 2942, "id": 2, "file_name": "val_38.png"},
    {"height": 1316, "width": 2987, "id": 3, "file_name": "val_33.png"},
    {"height": 1504, "width": 2872, "id": 4, "file_name": "val_30.png"},
    {"height": 1316, "width": 2970, "id": 5, "file_name": "val_5.png"},
    {"height": 1316, "width": 2860, "id": 6, "file_name": "val_21.png"},
    {"height": 1504, "width": 2804, "id": 7, "file_name": "val_39.png"},
    {"height": 1316, "width": 2883, "id": 8, "file_name": "val_46.png"},
    {"height": 1316, "width": 2967, "id": 9, "file_name": "val_20.png"},
    {"height": 1504, "width": 2872, "id": 10, "file_name": "val_3.png"},
    {"height": 1316, "width": 2954, "id": 11, "file_name": "val_29.png"},
    {"height": 976, "width": 1976, "id": 12, "file_name": "val_2.png"},
    {"height": 1316, "width": 2870, "id": 13, "file_name": "val_16.png"},
    {"height": 1316, "width": 3004, "id": 14, "file_name": "val_25.png"},
    {"height": 1316, "width": 2745, "id": 15, "file_name": "val_24.png"},
    {"height": 1504, "width": 2872, "id": 16, "file_name": "val_31.png"},
    {"height": 1316, "width": 2782, "id": 17, "file_name": "val_26.png"},
    {"height": 1316, "width": 2744, "id": 18, "file_name": "val_44.png"},
    {"height": 1504, "width": 2872, "id": 19, "file_name": "val_27.png"},
    {"height": 1504, "width": 2868, "id": 20, "file_name": "val_41.png"},
    {"height": 1316, "width": 3000, "id": 21, "file_name": "val_37.png"},
    {"height": 1316, "width": 2797, "id": 22, "file_name": "val_40.png"},
    {"height": 1316, "width": 2930, "id": 23, "file_name": "val_6.png"},
    {"height": 1316, "width": 3003, "id": 24, "file_name": "val_18.png"},
    {"height": 1316, "width": 2967, "id": 25, "file_name": "val_13.png"},
    {"height": 1316, "width": 2822, "id": 26, "file_name": "val_8.png"},
    {"height": 1316, "width": 2836, "id": 27, "file_name": "val_49.png"},
    {"height": 1316, "width": 2704, "id": 28, "file_name": "val_23.png"},
    {"height": 976, "width": 1976, "id": 29, "file_name": "val_1.png"},
    {"height": 1504, "width": 2872, "id": 30, "file_name": "val_43.png"},
    {"height": 1504, "width": 2872, "id": 31, "file_name": "val_28.png"},
    {"height": 1504, "width": 2872, "id": 32, "file_name": "val_19.png"},
    {"height": 1316, "width": 2728, "id": 33, "file_name": "val_14.png"},
    {"height": 1316, "width": 2747, "id": 34, "file_name": "val_32.png"},
    {"height": 976, "width": 1976, "id": 35, "file_name": "val_36.png"},
    {"height": 1316, "width": 2829, "id": 36, "file_name": "val_47.png"},
    {"height": 1316, "width": 2846, "id": 37, "file_name": "val_48.png"},
    {"height": 1536, "width": 3076, "id": 38, "file_name": "val_17.png"},
    {"height": 976, "width": 1976, "id": 39, "file_name": "val_42.png"},
    {"height": 1504, "width": 2884, "id": 40, "file_name": "val_45.png"},
    {"height": 1316, "width": 2741, "id": 41, "file_name": "val_9.png"},
    {"height": 1316, "width": 2794, "id": 42, "file_name": "val_4.png"},
    {"height": 1316, "width": 2959, "id": 43, "file_name": "val_34.png"},
    {"height": 1316, "width": 2874, "id": 44, "file_name": "val_10.png"},
    {"height": 1316, "width": 2978, "id": 45, "file_name": "val_35.png"},
    {"height": 1504, "width": 2884, "id": 46, "file_name": "val_11.png"},
    {"height": 1316, "width": 2794, "id": 47, "file_name": "val_12.png"},
    {"height": 1316, "width": 2959, "id": 48, "file_name": "val_7.png"},
    {"height": 1316, "width": 2912, "id": 49, "file_name": "val_22.png"},
    {"height": 1504, "width": 2872, "id": 50, "file_name": "val_0.png"},
]


class PanoramicProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load models
        with open("pretrained_models/faster_rcnn/version_3/hparams.yaml") as f:
            faster_rcnn_config = yaml.safe_load(f)["config"]
        faster_rcnn_config["pretrained"] = False
        self.faster_rcnn = FasterRCNN.load_from_checkpoint(
            f"pretrained_models/faster_rcnn/version_3/checkpoints/epoch=epoch=88-val_loss=val_loss=0.81.ckpt",
            config=faster_rcnn_config,
            map_location=self.device)
        self.faster_rcnn.eval()
        self.unet = UNet.load_from_checkpoint(
            f"pretrained_models/unet/version_66/checkpoints/epoch=epoch=20-val_loss=val_loss=0.23.ckpt",
            map_location=self.device)
        self.unet_threshold = torch.tensor([.7, .4, .8, .8, .1]).to(self.device)
        self.unet.eval()

        # Utilities
        self.encoder = LabelEncoder()
        self.transform = transforms.Compose([
            SquarePad(),
            transforms.Resize(224, interpolation=InterpolationMode.BILINEAR)
        ])
        self.unique_class_nms_processor = UniqueClassNMSProcessor(iou_threshold=.75)

    def __call__(self):
        # Load image array
        print("Loading image array..")
        file_path = glob.glob('/input/images/panoramic-dental-xrays/*.mha')[0]
        image_array = sitk.ReadImage(file_path)
        image_array = sitk.GetArrayFromImage(image_array)
        image_array = torch.from_numpy(image_array).type(torch.float32)

        # Predict affected teeth
        predictions = dict(name="Regions of interest",
                           type="Multiple 2D bounding boxes",
                           boxes=[],
                           version=dict(major=1, minor=0))
        for i in range(image_array.shape[2]):
            # Select image
            print("Selecting image..")
            image = image_array[:, :, i, 0].unsqueeze(0).type(torch.float32) / 255.0
            image_name = f"val_{i}.png"
            for input_image in list_ids:
                if input_image["file_name"] == image_name:
                    image_id = input_image["id"]
                    height = input_image["height"]
                    width = input_image["width"]

            # Detect teeth
            print("Detecting 1..")
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
            print("Detecting 2..")
            labels_2 = []
            for box in boxes_1:
                # Crop image
                box_int = torch.tensor(box, dtype=torch.int64)
                image_crop = image[:, box_int[1]:box_int[3], box_int[0]:box_int[2]]
                image_crop = self.transform(image_crop)

                # Get labels
                prediction = self.unet(image_crop.unsqueeze(0).to(self.device))
                prediction_probabilities = torch.sigmoid(prediction["classification"][0])
                labels = torch.where(prediction_probabilities >= self.unet_threshold)[
                    0].detach().cpu().tolist()

                # Save labels
                labels_2.append(labels)

            # Extract affected teeth
            print("Saving predictions..")
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

            # Append predictions
            for box, score, category_id_1, category_id_2, category_id_3 in zip(boxes_2, scores_2,
                                                                               category_id_1_2,
                                                                               category_id_2_2,
                                                                               category_id_3_2):
                predictions["boxes"].append(dict(
                    name=f"{category_id_1} - {category_id_2} - {category_id_3}",
                    corners=self.get_corners(box, image_id),
                    probability=score
                ))

        with open("/output/abnormal-teeth-detection.json", "w") as f:
            json.dump(predictions, f)

        print("Inference completed. Results saved to", "/output/abnormal-teeth-detection.json")

    @staticmethod
    def get_corners(box, id):
        """
        Convert x1y1x2y2 into all 4 corner coordinates
        :param box: x1y1x2y2 formatted box
        :return: list of box corners
        """
        x1, y1, x2, y2 = box

        return [[x1, y1, id], [x1, y2, id], [x2, y1, id], [x2, y2, id]]


if __name__ == "__main__":
    panoramic_processor = PanoramicProcessor()
    panoramic_processor()
