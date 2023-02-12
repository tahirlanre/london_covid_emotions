import argparse
import os
from typing import List

import torch
from transformers import BertForSequenceClassification, BertTokenizer
import pandas as pd
import numpy as np


def load_model(model_path: str) -> BertForSequenceClassification:
    model = BertForSequenceClassification.from_pretrained(model_path)
    return model


def tokenize(batch: List[str], tokenizer: BertTokenizer) -> dict:
    return tokenizer(
        batch,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )


def predict(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    dataset: pd.DataFrame,
    batch_size: int,
) -> List[int]:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)
    model.eval()

    batches = np.array_split(dataset, batch_size)
    predictions = []

    for batch in batches:
        with torch.no_grad():
            inputs = tokenize(batch["text"].tolist(), tokenizer)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            predictions.extend(logits.argmax(-1).cpu().numpy().tolist())

    return predictions


def main(args):
    model_path = os.path.join("./saved_models", args.emotion)
    model = load_model(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    dataset = pd.read_csv(args.data_path)

    predictions = predict(model, tokenizer, dataset, args.batch_size)

    # save predictions to csv
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, f"{args.emotion}.csv")
    dataset[args.emotion] = predictions
    dataset[["id", args.emotion]].to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to data to predict on")
    parser.add_argument("--output_dir", type=str, help="Directory to save predictions")
    parser.add_argument(
        "--emotion",
        type=str,
        choices=["annoyed", "anxious", "empathetic", "sad"],
        help="Emotion to predict",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for prediction"
    )

    args = parser.parse_args()
    main(args)
