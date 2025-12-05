def build_features(df):
    df["VehicleAge"] = 2025 - df["RegistrationYear"]
    df["ClaimOccurred"] = (df["TotalClaims"] > 0).astype(int)
    return df
