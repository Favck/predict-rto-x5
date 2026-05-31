import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
import numpy as np

data = pd.read_csv("train.csv")


data["РТО"] = np.log1p(data["РТО"])

for i in range(1, 7):
  data[f"lag{i}"] = data.groupby("new_id")["РТО"].shift(i)


data['trend_1_3'] = data['lag1'] - data['lag3']   # чуть шире


data['mean3'] = (
    data.groupby('new_id')['РТО']
      .shift(1)
      .rolling(3)
      .mean()
)


data['month_sin'] = np.sin(2*np.pi*data['Месяц']/12)
data['month_cos'] = np.cos(2*np.pi*data['Месяц']/12)



print(data.head(11))
data.to_csv("data_normalize.csv")

data["Численность населения"] = np.log1p(data["Численность населения"])

print(data.describe())
print(data.columns)
X_train = data.drop(columns=["РТО"])
Y_train = data["РТО"]

cat_features = ["Дата открытия, категориальный", "Торговая площадь, категориальный", "Населенный пункт", "Регион", ]

model = CatBoostRegressor(loss_function="MAE", 
                          l2_leaf_reg=5, 
                          n_estimators=5000,
                          depth=7,
                          learning_rate=0.03, 
                          thread_count=3,
                          random_seed=42)

model.fit(X_train, Y_train, cat_features=cat_features, plot=True)

model.save_model("model3.cbm")

