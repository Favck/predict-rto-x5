#MAPE: 0.1346914211251842
#R2: 0.8606215617496467
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error
import joblib

data = pd.read_csv("second_stage/data/train_2.csv", index_col=0)

df = data.copy()
def coder(df):
    cat_cols = df.select_dtypes(include="str").columns

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

coder(df)
train = df[df["Год"] < 2025]
test = df[df["Год"] == 2025]

X_train = train.drop(columns=["РТО"])
Y_train = train["РТО"]

X_test = test.drop(columns=["РТО"])
# X_test.to_csv("X_test.csv")
Y_test = test["РТО"]



model = RandomForestRegressor(random_state=42)

model.fit(X_train, Y_train)

preds = model.predict(X_test)
print("MAPE:", mean_absolute_percentage_error(Y_test, preds))
print("R2:", model.score(X_test, Y_test))
joblib.dump(model, "BaselineModel.pkl")


