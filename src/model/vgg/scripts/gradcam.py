import numpy as np
import torch
import yaml
from captum.attr import LayerGradCam
from torch.nn.functional import softmax
from torchvision.transforms import transforms, InterpolationMode
from tqdm import tqdm

from src.data.tooth_dataset import ToothDataset
from src.model.vgg.vgg import Vgg
from src.utils.transforms import SquarePad
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open("./src/model/vgg/scripts/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Find out whether gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    y_test = np.load(f"data/final/y_quadrant_enumeration_disease_unpacked_train.npy", allow_pickle=True)

    # Transforms
    transform = transforms.Compose([
        SquarePad(),
        transforms.Resize(224, interpolation=InterpolationMode.BILINEAR)  # Resize to 256 on the smaller edge
    ])

    dataset = ToothDataset(y_test, image_dir=f"{config['image_dir']}/{config['data_type']}/xrays", transform=transform)

    # Load model
    model = Vgg.load_from_checkpoint(f"checkpoints/vgg/version_16/checkpoints/epoch=epoch=43-val_loss=val_f1=0.62.ckpt")
    model.eval()

    for sample in tqdm(dataset, total=len(dataset)):
        # Extract data
        image = sample["image"]
        label = sample["label"]

        # Predict
        grad_cam = LayerGradCam(model, model.model.features[29])
        output = model(image.unsqueeze(0))
        confidence = torch.max(softmax(output, dim=-1))
        output = torch.argmax(output)

        # Grad-CAM analysis
        cam = grad_cam.attribute(image.unsqueeze(0), target=label)
        cam = torch.nn.functional.interpolate(cam, size=(image.shape[1], image.shape[2]), mode='bilinear',
                                              align_corners=False)
        cam = torch.nn.functional.relu(cam)

        # Save activation
        fig, ax = plt.subplots()
        ax.imshow(image[0], cmap="gray")
        ax.imshow(cam.squeeze().detach().numpy(), alpha=.5)
        plt.title(f"Predicted label: {output}; True label: {label}")
        if output == label and label == 2:
            fig.savefig(f"visualizations/gradcam/vgg/train_{sample['file_name']}")
