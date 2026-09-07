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