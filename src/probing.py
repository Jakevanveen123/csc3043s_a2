import numpy as np
 
 
def pool_layer(hidden_states: "np.ndarray", strategy: str) -> "np.ndarray":
    if strategy == "mean":
        return hidden_states.mean(axis=0)
    elif strategy == "last":
        return hidden_states[-1]
    elif strategy == "max":
        return hidden_states.max(axis=0)

 
 
def build_feature_matrix(results: list["InferenceResult"], layer: int, strategy: str) -> tuple["np.ndarray", "np.ndarray"]:
    features = []
    labels = []
    for result in results:
        if result.parsed_answer is None:
            continue
        features.append(
            pool_layer(result.hidden_states[layer], strategy)
        )
        labels.append(
            0 if result.parsed_answer == result.ground_truth else 1
        )
    X = np.stack(features, axis=0)
    y = np.array(labels, dtype=int)
    return X, y