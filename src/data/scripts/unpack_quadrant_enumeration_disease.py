import json

if __name__ == "__main__":
    # Load data
    with open("data/processed/train_quadrant_enumeration_disease_healthy.json") as f:
        data = json.load(f)

    # Unpack data
    data_unpacked = []
    for sample in data:
        for annotation in sample["annotations"]:
            data_unpacked.append(dict(file_name=sample["file_name"], annotation=annotation))

    # Save data
    with open("data/processed/train_quadrant_enumeration_disease_healthy_unpacked.json", "w") as f:
        json.dump(data_unpacked, f, indent=4)
