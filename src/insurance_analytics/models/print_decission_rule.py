# src/insurance_analytics/models/print_decission_rule.py

from sklearn.tree import export_text

def print_tree_rules(model, feature_names):
    """
    Print readable text rules from a decision tree.
    """
    try:
        if not hasattr(model, "tree_"):
            print("Model does not support decision tree rules.")
            return

        rules = export_text(model, feature_names=list(feature_names))
        print(rules)
    except Exception as e:
        print(f"[ERROR] Printing tree rules failed: {e}")
        raise
