import pandas as pd
import numpy as np

data = pd.read_csv("data_normalize.csv", index_col=0)
test = pd.read_csv("test.csv")  # new_id, rto

base = data.drop_duplicates(subset="new_id").set_index("new_id")
rto_by_month = data.set_index(["new_id", "Месяц"])["РТО"]

month_sin_11 = np.sin(2 * np.pi * 11 / 12)
month_cos_11 = np.cos(2 * np.pi * 11 / 12)

rows = []

for _, row in test.iterrows():
    nid = row["new_id"]
    rto_11 = row["rto"]
    
    train_row = base.loc[nid]
    rto = {m: rto_by_month.loc[(nid, m)] for m in range(1, 11)}
    
    lag1, lag2, lag3 = rto[10], rto[9], rto[8]
    lag4, lag5, lag6 = rto[7], rto[6], rto[5]
    
    trend_1_3 = lag1 - lag3  # month10 - month8
    mean3 = np.mean([lag1, lag2, lag3])
    
    static_cols = [c for c in train_row.index if c not in [
        "Месяц", "РТО",
        "lag1","lag2","lag3","lag4","lag5","lag6",
        "trend_1_3", "mean3",
        "month_sin","month_cos"
    ]]
    
    new_row = {
        "new_id": nid,
        "Месяц": 11,
        **{c: train_row[c] for c in static_cols},
        "РТО": rto_11,
        "lag1": lag1, "lag2": lag2, "lag3": lag3,
        "lag4": lag4, "lag5": lag5, "lag6": lag6,
        "trend_1_3": trend_1_3,
        "mean3": mean3,
        "month_sin": month_sin_11,
        "month_cos": month_cos_11,
    }
    rows.append(new_row)

cols = [c for c in data.columns if c != "Unnamed: 0"]
result = pd.DataFrame(rows, columns=cols)
print(result)
result = result.drop(columns=["РТО"])
result.to_csv("test_data_norm.csv")
