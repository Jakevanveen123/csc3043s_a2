import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
 
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

def train_val_split(results: list["InferenceResult"], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = np.arange(len(results))
    question_types = []
    for result in results:
        question_types.append(result.question_type)
    question_types = np.array(question_types)
 
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=question_types,
    )
    return train_indices.tolist(), val_indices.tolist()


def train_probe(X_train: "np.ndarray", y_train: "np.ndarray") -> "LogisticRegression":
    probe = LogisticRegression(max_iter=1000)
    probe.fit(X_train, y_train)
    return probe

def evaluate_probe(probe, X_val: "np.ndarray", y_val: "np.ndarray") -> dict:
    y_pred = probe.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
 
    if len(np.unique(y_val)) < 2:
        auroc = float("nan")
    else:
        y_proba = probe.predict_proba(X_val)[:, 1]
        auroc = roc_auc_score(y_val, y_proba)
 
    return {"accuracy": accuracy, "auroc": auroc}