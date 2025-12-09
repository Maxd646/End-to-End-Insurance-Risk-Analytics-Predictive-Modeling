# src/insurance_analytics/models/xgboost_model.py

from xgboost import XGBRegressor, XGBClassifier

def train_xgb_regressor(X_train, y_train):
    """
    Train an XGBoost regression model.
    """
    try:
        model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"[ERROR] XGBRegressor training failed: {e}")
        raise


def train_xgb_classifier(X_train, y_train):
    """
    Train an XGBoost classifier.
    """
    try:
        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"[ERROR] XGBClassifier training failed: {e}")
        raise
