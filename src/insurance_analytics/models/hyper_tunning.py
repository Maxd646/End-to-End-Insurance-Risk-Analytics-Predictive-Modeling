# src/insurance_analytics/models/hyper_tunning.py

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

def tune_random_forest(X_train, y_train):
    """
    RandomizedSearchCV hyperparameter tuning for Random Forest.
    """
    try:
        params = {
            "n_estimators": [100, 300, 500],
            "max_depth": [None, 10, 20, 40],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

        model = RandomForestRegressor()

        search = RandomizedSearchCV(
            model,
            params,
            n_iter=10,
            cv=3,
            verbose=1,
            n_jobs=-1,
            random_state=42,
        )

        search.fit(X_train, y_train)
        return search.best_estimator_
    except Exception as e:
        print(f"[ERROR] Random forest tuning failed: {e}")
        raise
