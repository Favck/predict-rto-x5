import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from catboost import CatBoostRegressor
import pandas as pd
import joblib 
import numpy as np
from utils import predict_save, feature_plot, split_train_test, coder

data = pd.read_csv("second_stage/data/March_data_final.csv", index_col=0)
model = CatBoostRegressor()
model.load_model(f"model_35.cbm")
pred_data = data
pr = model.predict(pred_data)
    
test = pd.DataFrame({"new_id":data["new_id"].to_list(), "rto": np.expm1(pr)})
test.set_index("new_id", inplace=True)
test.to_csv(f"second_stage/data/final.csv")

feature_plot(model, data, cat=True)
predict_save(model, data, filename="final.csv", norm=True)