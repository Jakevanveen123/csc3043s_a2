import sys, os, pickle
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
 
import numpy as np
import pandas as pd
 
from inference import load_results
from probing import (build_feature_matrix,train_val_split,train_probe,evaluate_probe,)
 
data_dir = os.path.join(os.path.dirname(__file__), "../data")
results_path = os.path.join(data_dir, "inference_results.pkl")

def load_layer(path, layer):
    results = []
    with open(path, "rb") as f:
        while True:
            try:
                r = pickle.load(f)
            except EOFError:
                break
            r.hidden_states = {layer: r.hidden_states[layer]}
            results.append(r)
    return results

with open(results_path, "rb") as f:
    num_layers = len(pickle.load(f).hidden_states)
 
split_path = os.path.join(data_dir, "train_val_split.npz")
if os.path.exists(split_path):
    split = np.load(split_path)
    train_indices = split["train_indices"].tolist()
    val_indices = split["val_indices"].tolist()
else:
    train_indices, val_indices = train_val_split(load_layer(results_path, 0), 0.3, 0)
    np.savez(split_path,train_indices=np.array(train_indices),val_indices=np.array(val_indices),)
 
 
records = []
for layer in range(num_layers):
    results = load_layer(results_path, layer)
    train_results = [results[i] for i in train_indices]
    val_results = [results[i] for i in val_indices]
    X_train, y_train = build_feature_matrix(train_results, layer, "mean")
    X_val, y_val = build_feature_matrix(val_results, layer, "mean")
    probe = train_probe(X_train, y_train)
    metrics = evaluate_probe(probe, X_val, y_val)
    records.append({"layer": layer, **metrics})
 
pd.DataFrame(records).to_csv(os.path.join(data_dir, "probe_results.csv"), index=False)
 