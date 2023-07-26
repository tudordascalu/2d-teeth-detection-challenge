import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import yaml

from src.model.faster_rcnn.faster_rcnn import FasterRCNN

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

if __name__ == "__main__":
    # image_acc = []
    # for i in range(3):
    #     # Read image
    #     image = plt.imread(f"data/raw/validation_data/quadrant_enumeration_disease/xrays/val_{i}.png")
    #
    #     # Pad images (max height 1536 max width 3076)
    #     image = np.pad(image, ((0, 1536 - image.shape[0]), (0, 3076 - image.shape[1]), (0, 0)))
    #
    #     # Store images
    #     image_acc.append(image)
    #
    # # Convert the NumPy array to a SimpleITK Image
    # image_acc = np.array(image_acc)
    # image_acc = image_acc.transpose(1, 2, 0, 3)
    # image_acc = sitk.GetImageFromArray(image_acc)
    #
    # # Save as .mha
    # sitk.WriteImage(image_acc, "input/images/panoramic-dental-xrays/input.mha")
    with open("pretrained_models/faster_rcnn/version_3/hparams.yaml") as f:
        config = yaml.safe_load(f)["config"]
    config["pretrained"] = False
    model = FasterRCNN.load_from_checkpoint(
        "pretrained_models/faster_rcnn/version_3/checkpoints/epoch=epoch=88-val_loss=val_loss=0.81.ckpt", config=config)