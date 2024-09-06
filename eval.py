import pickle
import numpy as np
from sklearn.metrics import classification_report
from train import load_jsonl_files, process_dataset


model = pickle.load(open("logreg.pkl", "rb"))

file_paths = [
    "data/dev.jsonl",
]
eval_dataset = load_jsonl_files(file_paths)["dev"]

x_val, y_val = process_dataset(eval_dataset)

preds = model.predict(np.array(x_val))
print(classification_report(y_val, preds))
