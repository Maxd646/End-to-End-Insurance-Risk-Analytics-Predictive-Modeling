# src/insurance_analytics/viz/plot_model.py

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot top N feature importances for tree-based models.
    """
    try:
        if not hasattr(model, "feature_importances_"):
            print("Model does not support feature importance.")
            return

        imp = model.feature_importances_
        idx = np.argsort(imp)[-top_n:]

        plt.figure(figsize=(10, 6))
        plt.barh(np.array(feature_names)[idx], imp[idx])
        plt.title("Top Feature Importances")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[ERROR] Feature importance plot failed: {e}")
        raise
