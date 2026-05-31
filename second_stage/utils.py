from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def coder(df):
    cat_cols = df.select_dtypes(include="str").columns

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

def feature_plot(model, X_train, cat=False):

    if cat:
        imp = model.get_feature_importance()
    else:
        imp = model.feature_importances_

    plt.barh(X_train.columns, imp)
    plt.show()

def predict_save(model, data, filename="test.csv", norm=False):
    pr = model.predict(data)
    if norm:
        test = pd.DataFrame({"new_id":data["new_id"].to_list(), "rto": np.expm1(pr)})
    else:
        test = pd.DataFrame({"new_id":data["new_id"].to_list(), "rto": pr})
    test.set_index("new_id", inplace=True)
    test.to_csv(f"second_stage/data/{filename}")


def split_train_test(df):
    X_test = df.groupby("new_id").tail(2)
    X_train = df.drop(X_test.index)
    Y_test = X_test["РТО"]
    Y_train = X_train["РТО"]
    X_test, X_train = X_test.drop(columns=["РТО"]), X_train.drop(columns=["РТО"])
    return X_train, Y_train, X_test,Y_test