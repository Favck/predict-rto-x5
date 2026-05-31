import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from catboost import CatBoostRegressor
import pandas as pd
from utils import split_train_test
from sklearn.metrics import mean_absolute_percentage_error
import joblib

data = pd.read_csv("second_stage/data/EditedData_final.csv", index_col=0)

cat_features = list(data.select_dtypes(include="str").columns)

model = CatBoostRegressor(
                iterations=5000,
                has_time=True,
                random_seed=42,
                cat_features=cat_features,
                learning_rate=0.01,
                l2_leaf_reg=10,
                rsm=0.6,
                random_strength=3,
                early_stopping_rounds=50
                          )

X_train, Y_train, X_test, Y_test = split_train_test(data)




model.fit(X_train, Y_train, eval_set=(X_test, Y_test))
pred = model.predict(X_test)
print("R2:", model.score(X_test,Y_test))
print("MAPE:", mean_absolute_percentage_error(Y_test, pred))

model.save_model("model_35.cbm")

#R2: 0.9691136330912609
#MAPE: 0.002741023002383745