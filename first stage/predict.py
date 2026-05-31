from catboost import CatBoostRegressor
import pandas as pd
import numpy as np

data = pd.read_csv("data_normalize.csv", index_col=0)

model = CatBoostRegressor()

model.load_model("model3.cbm")

predict = model.predict(data)

kk = pd.DataFrame({"new_id":data["new_id"].astype(int), "rto":np.expm1(predict)})

kk.to_csv("final3.csv")
