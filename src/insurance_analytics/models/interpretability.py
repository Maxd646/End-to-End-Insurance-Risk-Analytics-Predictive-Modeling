# src/insurance_analytics/models/interpretability.py

import shap

def shap_summary(model, X):
    """
    Display SHAP summary plot.
    """
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
        shap.summary_plot(values, X)
    except Exception as e:
        print(f"[ERROR] SHAP summary failed: {e}")
        raise


def shap_waterfall(model, X, index=0):
    """
    Display SHAP waterfall for one prediction.
    """
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
        shap.plots.waterfall(values[index])
    except Exception as e:
        print(f"[ERROR] SHAP waterfall failed: {e}")
        raise
