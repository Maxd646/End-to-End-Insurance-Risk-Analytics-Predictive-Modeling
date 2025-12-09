# src/insurance_analytics/models/random_forest.py

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def train_random_forest(X_train, y_train, task="regression", n_estimators=200):
    """
    Train a random forest model.
    """
    try:
        if task == "classification":
            model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1)

        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"[ERROR] Random forest training failed: {e}")
        raise
