# src/insurance_analytics/models/data_split.py

from sklearn.model_selection import train_test_split

def split_train_test(df, target, test_size=0.2, stratify=False, random_state=42):
    """
    Safely split a dataset into train/test.
    """
    try:
        X = df.drop(columns=[target])
        y = df[target]

        strat_val = y if stratify else None

        return train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=strat_val,
            random_state=random_state,
        )
    except Exception as e:
        print(f"[ERROR] Data split failed: {e}")
        raise
