from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score

def train_model(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    model = RandomForestRegressor()
    model.fit(X_train,y_train)

    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test,preds,squared=False)
    r2 = r2_score(y_test,preds)

    return model,rmse,r2
