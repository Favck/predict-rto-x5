import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import split_train_test
from catboost import CatBoostRegressor, EFeaturesSelectionAlgorithm, Pool
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import joblib
import numpy as np

data = pd.read_csv("second_stage/data/EditedData_final.csv", index_col=0)

cat_features = list(data.select_dtypes(include="str").columns)

X_train, Y_train, X_test, Y_test = split_train_test(data)
feature_names = X_train.columns.tolist()
feature_indices = list(range(1, len(feature_names)))
for i in [18, 22, 14, len(feature_names)]:

    model = CatBoostRegressor(
                iterations=5000,
                has_time=True,
                random_seed=42,
                cat_features=cat_features,
                learning_rate=0.01,
                l2_leaf_reg=10,
                rsm=0.6,
                random_strength=3,
                early_stopping_rounds=50,
                verbose=100
                          )




    train_pool = Pool(X_train,Y_train, feature_names=feature_names,cat_features=cat_features)
    test_pool = Pool(X_test, Y_test, feature_names=feature_names, cat_features=cat_features)
    

    summary = model.select_features(
                train_pool,
                eval_set=test_pool,
                features_for_select=feature_indices,
                algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
                num_features_to_select=i,
                steps=4,
    )

    joblib.dump(summary, f"summary{i}.pkl")

    selected = ["new_id"] + summary['selected_features_names']
    X_train_sel = X_train[selected]
    X_test_sel = X_test[selected]
    cat_sel = [f for f in cat_features if f in X_train_sel.columns]


    model_final = CatBoostRegressor(
                iterations=5000,
                random_seed=42,
                has_time=True,
                cat_features=cat_sel,
                learning_rate=0.01,
                l2_leaf_reg=10,
                rsm=0.6,
                random_strength=3,
                early_stopping_rounds=50,
                verbose=100
                          )
    model_final.fit(X_train_sel,Y_train, eval_set=(X_test_sel, Y_test))
    model_final.save_model(f"model_{i}.cbm")

    preds = model_final.predict(X_test_sel)

    preds_real = np.expm1(preds)
    Y_test_real = np.expm1(Y_test)
    
    print(f"\n=== i={i} ===")
    print(f"MAPE: {mean_absolute_percentage_error(Y_test, preds):.6f}")

    



