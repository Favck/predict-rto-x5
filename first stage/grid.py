from sklearn.model_selection import ParameterGrid

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import numpy as np

data = pd.read_csv("data.csv", index_col=0)

param_grid = {
    "n_estimators":[1500],
    "bootstrap_type":[None],
    'l2_leaf_reg': [5],
    "depth":[8, 10]

}
k = data[data["Месяц"] < 10]
X_train = data[(data["Месяц"] < 10)].drop(columns=["РТО"])
Y_train =  k["РТО"]


t = data[data["Месяц"] == 10]
X_val = t.drop(columns=["РТО"])
Y_val = t["РТО"]


best_score = 10**6
best_params = None
best_model = None

grid = ParameterGrid(param_grid)
cat_features = ["Дата открытия, категориальный", "Торговая площадь, категориальный", "Населенный пункт", "Регион", ]

for params in grid:
    print("Test ", params)

    model = CatBoostRegressor(
        **params,
        cat_features=cat_features,
        learning_rate=0.1,
        thread_count=3,
        loss_function="MAE",

        eval_metric="MAPE",
        verbose=False
    )

    model.fit(
        X_train,
        Y_train,
        eval_set=(X_val, Y_val),
        use_best_model=True
    )

    preds = model.predict(X_val)
    score = mean_absolute_percentage_error(Y_val, preds)

    print(f"MAPE: {score}")
    if score < best_score:
        best_score = score
        best_params = params
        best_model = model

print("Best score:", best_score)
print("Best params:", best_params)

# сохраняем лучшую модель
best_model.save_model("best_catboost.cbm")
print("Model saved!")