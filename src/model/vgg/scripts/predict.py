import json

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from src.data.tooth_dataset import ToothDataset
from src.model.vgg.vgg import Vgg
from src.utils.transforms import SquarePad

if __name__ == "__main__":
    with open("./src/model/unet/scripts/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Find out whether gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = Vgg.load_from_checkpoint(
        f"checkpoints/vgg/version_29/checkpoints/epoch=epoch=14-val_loss=val_f1=0.72.ckpt",
        map_location=device
    )
    model.eval()

    # Load data
    with open("data/final/train_quadrant_enumeration_disease_healthy_unpacked_test.json") as f:
        X = json.load(f)

    if model.hparams.config["n_classes"] > 1:
        # Remove healthy samples
        X = list(filter(lambda x: int(x["annotation"]["category_id_3"]) != 4, X))
    else:
        X = list(filter(lambda x: "score" not in x["annotation"] or x["annotation"]["score"] >= .9, X))
        for x in X:
            if x["annotation"]["category_id_3"] == 4:
                x["annotation"]["category_id_3"] = 0
            else:
                x["annotation"]["category_id_3"] = 1

    transform = transforms.Compose([
        SquarePad(),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR)
    ])
    to_pil = transforms.ToPILImage()

    # Define dataset
    dataset_args = dict(image_dir=f"data/raw/training_data/quadrant_enumeration_disease/xrays",
                        n_classes=model.hparams.config["n_classes"])
    dataset = ToothDataset(X, transform=transform, **dataset_args)

    # Define loaders
    loader_args = dict(batch_size=config["batch_size"], num_workers=0, pin_memory=True)
    loader_train = DataLoader(dataset, **loader_args)

    # Predict for each sample
    prediction_list = []
    for sample in tqdm(dataset, total=len(dataset)):
        input = sample["image"]
        file_name = sample["file_name"]
        label = sample["label"]

        with torch.no_grad():
            prediction = model(input.unsqueeze(0).to(device))[0]
        prediction_list.append(dict(
            file_name=file_name,
            label=label.cpu().tolist(),
            prediction=prediction.detach().cpu().tolist()))

    with open(f"output/vgg_binary_predict.json", "w") as f:
        json.dump(prediction_list, f)
