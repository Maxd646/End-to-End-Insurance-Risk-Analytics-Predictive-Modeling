# src/insurance_analytics/models/decission_tree.py

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

def train_decision_tree(X_train, y_train, task="regression", max_depth=None):
    """
    Train a decision tree for classification or regression.
    """
    try:
        if task == "classification":
            model = DecisionTreeClassifier(max_depth=max_depth)
        else:
            model = DecisionTreeRegressor(max_depth=max_depth)

        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"[ERROR] Decision tree training failed: {e}")
        raise
