import pandas as pd

def cross_category_split(results: list["InferenceResult"],train_types: list[str],test_types: list[str],val_indices: list[int],) -> tuple[list[int], list[int]]:
    val_set = set(val_indices)

    train_indices = []
    for i in range(len(results)):
        if i not in val_set and results[i].question_type in train_types:
            train_indices.append(i)
 
    test_indices = []
    for i in val_indices:
        if results[i].question_type in test_types:
            test_indices.append(i)
 
    return train_indices, test_indices

def compare_probe_to_baseline(probe_predictions: "np.ndarray",confidence_predictions: "np.ndarray",y_true: "np.ndarray",metadata: list[dict],) -> "pandas.DataFrame":
    rows = []
    for i in range(len(y_true)):
        if probe_predictions[i] != confidence_predictions[i]:
            rows.append({
                "image_id": metadata[i]["image_id"],
                "category": metadata[i]["category"],
                "question_type": metadata[i]["question_type"],
                "ground_truth": y_true[i],
                "confidence": metadata[i]["confidence"],
                "probe_prediction": probe_predictions[i],
            })
 
    return pd.DataFrame(rows)