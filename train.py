import pickle
from typing import Iterable, Mapping, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import re


def load_jsonl_files(file_paths: Iterable[str]) -> dict[str, list[Mapping[str, Any]]]:
    datasets = dict()
    for file_path in file_paths:
        data = []
        dataset_type = re.search(r"data/(.*).jsonl", file_path).group(1)
        with open(file_path, "r") as file:
            for line in file:
                data.append(json.loads(line))
        datasets[dataset_type] = data
    return datasets


def process_dataset(
    dataset: Iterable[Mapping[str, Any]], model=SentenceTransformer("all-MiniLM-L6-v2")
) -> tuple[Any, list[str]]:
    encoded = []
    labels = []

    for article in tqdm(dataset):
        paragraphs = [article["headline"] + " " + article["short_description"]]
        embeddings = model.encode(paragraphs)
        mean_pooled = embeddings.mean(axis=0)
        encoded.append(mean_pooled)
        if "category" in article:
            labels.append(article["category"])
    return np.array(encoded), labels


def main() -> None:
    file_paths = ["data/dev.jsonl", "data/test.jsonl", "data/train.jsonl"]
    dataset = load_jsonl_files(file_paths)["train"]

    x_train, y_train = process_dataset(dataset)
    x_train = np.array(x_train)

    logreg = LogisticRegression(
        max_iter=1000,
    )
    logreg.fit(x_train, y_train)
    pickle.dump(logreg, open("logreg.pkl", "wb"))
    loaded_model = pickle.load(open("logreg.pkl", "rb"))

    print(classification_report(y_train, loaded_model.predict(x_train)))


if __name__ == "__main__":
    main()
