import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import joblib



data = pd.read_csv("second_stage/data/train_2.csv", index_col=0)
data['new_id'] = data['new_id'].astype(str)
norm = ["Численность населения", "Количество домохозяйств", "РТО"]

for i in norm:
    data[i] = np.log1p(data[i])

for num in [1, 2,3,6,12]:
    data[f"RTO_lag{num}"] = (data.groupby("new_id")["РТО"].shift(num))

data["month_sin"] = np.sin(2 * np.pi * data["Месяц"] / 12)
data["month_cos"] = np.cos(2 * np.pi * data["Месяц"] / 12)

data["RTO_mean3"] = data.groupby("new_id")["РТО"].transform(lambda x: x.shift(1).rolling(3).mean())
data["RTO_mean6"] = data.groupby("new_id")["РТО"].transform(lambda x: x.shift(1).rolling(6).mean())
data["trend"] = data["RTO_lag1"] - data["RTO_lag2"]
data["trend6"] = data["RTO_lag1"] - data["RTO_lag6"]
data["ema3"] = 0.6*data["RTO_lag1"] + 0.3*data["RTO_lag2"] + 0.1*data["RTO_lag3"]
data["rto_std3"] = data[["RTO_lag1","RTO_lag2","RTO_lag3"]].std(axis=1)

data.fillna(0, inplace=True)
data.to_csv("EditedData_final.csv")