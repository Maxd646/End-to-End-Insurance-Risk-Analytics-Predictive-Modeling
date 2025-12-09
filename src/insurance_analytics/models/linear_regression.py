# src/insurance_analytics/models/linear_regression.py

from sklearn.linear_model import LinearRegression

def train_linear_regression(X_train, y_train):
    """
    Train a simple linear regression model.
    """
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"[ERROR] Linear regression training failed: {e}")
        raise
