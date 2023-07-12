import numpy as np
import torch
import yaml
from torchvision.transforms import transforms
from tqdm import tqdm

from src.data.tooth_segmentation_dataset import ToothSegmentationDataset
from src.model.unet.unet import UNet
from src.utils.transforms import SquarePad
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open("./src/model/unet/scripts/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Find out whether gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    X = np.load(f"data/final/y_quadrant_enumeration_disease_with_healthy_samples_and_segmentation_unpacked_train.npy",
                allow_pickle=True)

    transform_input = transforms.Compose([
        SquarePad(),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR)
    ])

    transform_target = transforms.Compose([
        SquarePad(),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.NEAREST)  # Resize to 256 on the smaller edge
    ])

    dataset = ToothSegmentationDataset(dataset=X,
                                       data_dir=config["data_dir"],
                                       transform_input=transform_input,
                                       transform_target=transform_target)

    # Load model
    model = UNet.load_from_checkpoint(
        f"checkpoints/unet/version_23/checkpoints/epoch=epoch=316-val_dice_score=train_loss=0.01.ckpt")
    model.eval()

    # Predict for each sample
    for sample in tqdm(dataset, total=len(dataset)):
        input, mask, label = sample["image"], sample["mask"], sample["label"]

        if label != 4:
            with torch.no_grad():
                prediction = model(input.unsqueeze(0)).squeeze().argmax(0)
            plt.imshow(input[0], cmap="gray")
            plt.imshow(prediction, alpha=.5)
            plt.show()
            plt.close()
            print("da")
