import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
 
import numpy as np
import pandas as pd
 
from inference import load_results
from probing import (build_feature_matrix,train_val_split,train_probe,evaluate_probe,)
 
data_dir = os.path.join(os.path.dirname(__file__), "../data")
results = load_results(os.path.join(data_dir, "inference_results.pkl"))
 
split_path = os.path.join(data_dir, "train_val_split.npz")
if os.path.exists(split_path):
    split = np.load(split_path)
    train_indices = split["train_indices"].tolist()
    val_indices = split["val_indices"].tolist()
else:
    train_indices, val_indices = train_val_split(results, 0.3, 0)
    np.savez(split_path,train_indices=np.array(train_indices),val_indices=np.array(val_indices),)
 
train_results = [results[i] for i in train_indices]
val_results = [results[i] for i in val_indices]
 
records = []
for layer in range(len(results[0].hidden_states)):
    X_train, y_train = build_feature_matrix(train_results, layer, "mean")
    X_val, y_val = build_feature_matrix(val_results, layer, "mean")
    probe = train_probe(X_train, y_train)
    metrics = evaluate_probe(probe, X_val, y_val)
    records.append({"layer": layer, **metrics})
 
pd.DataFrame(records).to_csv(os.path.join(data_dir, "probe_results.csv"), index=False)
 