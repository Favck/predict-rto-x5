import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
from utils import coder, feature_plot, predict_save, split_train_test
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

data = pd.read_csv("second_stage/data/March_data_final.csv", index_col=0)

coder(data)
_,_, X_test, Y_test = split_train_test(data)

model = joblib.load("second_stage/gradboost/gradientboost.pkl")

print("R2:", model.score(X_test, Y_test))
print("MAPE:", mean_absolute_percentage_error(Y_test, model.predict(X_test)))

feature_plot(model, X_test)

predict_save(model,data, filename="testGrad.csv", norm=True)

