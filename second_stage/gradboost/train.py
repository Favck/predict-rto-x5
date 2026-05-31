import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import coder, split_train_test
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import joblib




df = pd.read_csv("second_stage/data/EditedData_final.csv", index_col=0)

coder(df)


X_train, Y_train, X_test, Y_test = split_train_test(df)

model = GradientBoostingRegressor(
    random_state=42,
    learning_rate=0.01,
    n_estimators=500
)

model.fit(X_train, Y_train)


joblib.dump(model, "gradintboost.pkl")

print("R2:", model.score(X_test, Y_test)) 
print("MAPE:", mean_absolute_percentage_error(Y_test, model.predict(X_test)))
# R2: 0.9577545441351356
# MAPE: 0.003402266775103677