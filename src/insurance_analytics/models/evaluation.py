# src/insurance_analytics/models/evaluation.py

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def eval_regression(model, X_test, y_test):
    """
    Evaluate regression models with RMSE, R2 and MAE.
    """
    try:
        preds = model.predict(X_test)

        return {
            "RMSE": mean_squared_error(y_test, preds, squared=False),
            "R2": r2_score(y_test, preds),
            "MAE": mean_absolute_error(y_test, preds),
        }
    except Exception as e:
        print(f"[ERROR] Regression evaluation failed: {e}")
        raise


def eval_classification(model, X_test, y_test):
    """
    Evaluate classification models using accuracy, precision, recall and F1.
    """
    try:
        preds = model.predict(X_test)

        try:
            prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, prob)
        except:
            auc = None

        return {
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1": f1_score(y_test, preds, zero_division=0),
            "ROC-AUC": auc,
        }
    except Exception as e:
        print(f"[ERROR] Classification evaluation failed: {e}")
        raise
