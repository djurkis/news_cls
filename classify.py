import pickle
import numpy as np
from sklearn.metrics import classification_report

from train import load_jsonl_files, process_dataset
import json

model = pickle.load(open("logreg.pkl", "rb"))
file_paths = [
    "data/test.jsonl",
]
eval_dataset = load_jsonl_files(file_paths)["test"]

x_test, y_test = process_dataset(eval_dataset)

preds = model.predict(np.array(x_test))


with open("test_w_preds.jsonl", "w") as file:
    for article, pred in zip(eval_dataset, preds):
        article["category"] = pred
        file.write(json.dumps(article) + "\n")

print(classification_report(y_test, preds))
