import sys, os, pickle
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
 
import numpy as np
import pandas as pd
 
from probing import build_feature_matrix, train_probe, evaluate_probe
from sklearn.metrics import roc_auc_score
from generalization import cross_category_split, compare_probe_to_baseline
 
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

probe_results = pd.read_csv(os.path.join(data_dir, "probe_results.csv"))
best_row = probe_results.loc[probe_results["auroc"].idxmax()]
best_layer = int(best_row["layer"])

results = load_layer(results_path, best_layer)
 
split = np.load(os.path.join(data_dir, "train_val_split.npz"))
train_indices = split["train_indices"].tolist()
val_indices = split["val_indices"].tolist()
 
cross_train, cross_test = cross_category_split(results,["present", "absent_random"],["absent_adversarial"],val_indices,)
cross_train_results = [results[i] for i in cross_train]
cross_test_results = [results[i] for i in cross_test]
 
X_train, y_train = build_feature_matrix(cross_train_results, best_layer, "mean")
X_test, y_test = build_feature_matrix(cross_test_results, best_layer, "mean")
cross_probe = train_probe(X_train, y_train)
cross_metrics = evaluate_probe(cross_probe, X_test, y_test)
 
pd.DataFrame([{"split": "cross_category", **cross_metrics},{"split": "within_distribution","accuracy": best_row["accuracy"],"auroc": best_row["auroc"],},]).to_csv(
    os.path.join(data_dir, "cross_category_results.csv"), index=False)
 
train_results = [results[i] for i in train_indices]
val_results = [results[i] for i in val_indices]
 
X_train, y_train = build_feature_matrix(train_results, best_layer, "mean")
best_probe = train_probe(X_train, y_train)
 
X_val, y_val = build_feature_matrix(val_results, best_layer, "mean")
probe_predictions = best_probe.predict(X_val)
 
metadata = []
confidence_values = []
for r in val_results:
    if r.parsed_answer is not None:
        metadata.append({"image_id": r.image_id,"category": r.category,"question_type": r.question_type,"confidence": r.confidence,})
        confidence_values.append(r.confidence)
 
confidence_values = np.array(confidence_values)
confidence_predictions = (confidence_values < 0.5).astype(int)
 
disagreements = compare_probe_to_baseline(probe_predictions, confidence_predictions, y_val, metadata)
disagreements.to_csv(os.path.join(data_dir, "disagreements.csv"), index=False)
print(roc_auc_score(y_val, 1 - confidence_values))