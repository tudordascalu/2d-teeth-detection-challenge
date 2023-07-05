import json

if __name__ == "__main__":
    # Read predictions
    with open("output/predictions.json") as f:
        predictions = json.load(f)

    # Filter predictions with confiedence score lower than 0.5
    predictions_filtered = list(filter(lambda x: float(x["p_category_id_3"]) >= 0.6, predictions))

    # Save predictions
    with open("output/predictions_filtered.json", "w") as f:
        json.dump(predictions_filtered, f)
