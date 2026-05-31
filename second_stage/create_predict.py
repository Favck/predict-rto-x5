import pandas as pd
import numpy as np

df = pd.read_csv("second_stage/data/train_2.csv", index_col=0)
# те же нормализации что в train
for col in ["Численность населения", "Количество домохозяйств", "РТО"]:
    df[col] = np.log1p(df[col])

static_cols = [
    "Среднее количество промо товаров в чеке",
    "Среднее количество товаров в чеке",
    "Среднее количество отмен",
    "Рабочие часы в день",
    "Дата открытия, категориальный",
    "Торговая площадь, категориальный",
    "Населенный пункт",
    "Регион",
    "Численность населения",
    "Количество домохозяйств",
    "Трафик пеший, в час",
    "Трафик авто, в час",
    "Маркетплейсы, доставки, постаматы (100 м)",
    "Медицинские уч. и аптеки (300 м)",
    "Школы (300 м)",
    "Остановки (300 м)",
    "Продуктовые магазины (500 м)",
    "Пятерочки (500 м)",
    "Количество касс",
    "Флаг алкогольной лицензии",
]

TARGET_YEAR  = 2025
TARGET_MONTH = 3

def shift_month(year, month, lag):
    total = year * 12 + (month - 1) - lag
    return total // 12, total % 12 + 1

all_data = []

for new_id in df["new_id"].unique():
    shop_df = df[df["new_id"] == new_id].copy()
    shop_df = shop_df.sort_values(["Год", "Месяц"])

    shop_2025 = shop_df[shop_df["Год"] == TARGET_YEAR]
    if len(shop_2025) == 0:
        continue

    def get_rto(year, month):
        row = shop_df[(shop_df["Год"] == year) & (shop_df["Месяц"] == month)]
        if len(row) == 0:
            return 0.0
        return row["РТО"].iloc[0]

    rto_lag1  = get_rto(*shift_month(TARGET_YEAR, TARGET_MONTH, 1))
    rto_lag2  = get_rto(*shift_month(TARGET_YEAR, TARGET_MONTH, 2))
    rto_lag3  = get_rto(*shift_month(TARGET_YEAR, TARGET_MONTH, 3))
    rto_lag6  = get_rto(*shift_month(TARGET_YEAR, TARGET_MONTH, 6))
    rto_lag12 = get_rto(*shift_month(TARGET_YEAR, TARGET_MONTH, 12))

    lags_3 = [get_rto(*shift_month(TARGET_YEAR, TARGET_MONTH, i)) for i in range(1, 4)]
    lags_6 = [get_rto(*shift_month(TARGET_YEAR, TARGET_MONTH, i)) for i in range(1, 7)]

    non_zero_3 = [v for v in lags_3 if v != 0.0]
    non_zero_6 = [v for v in lags_6 if v != 0.0]

    rto_mean3 = sum(non_zero_3) / len(non_zero_3) if non_zero_3 else 0.0
    rto_mean6 = sum(non_zero_6) / len(non_zero_6) if non_zero_6 else 0.0

    data = {
        "new_id": str(new_id),
        "Год": TARGET_YEAR,
        "Месяц": TARGET_MONTH,
        **{col: shop_2025[col].iloc[-1] for col in static_cols},
        "RTO_lag1":  rto_lag1,
        "RTO_lag2":  rto_lag2,
        "RTO_lag3":  rto_lag3,
        "RTO_lag6":  rto_lag6,
        "RTO_lag12": rto_lag12,
        "month_sin": np.sin(2 * np.pi * TARGET_MONTH / 12),
        "month_cos": np.cos(2 * np.pi * TARGET_MONTH / 12),
        "RTO_mean3": rto_mean3,
        "RTO_mean6": rto_mean6,
        "trend":     rto_lag1 - rto_lag2,
        "trend6":    rto_lag1 - rto_lag6,
        "ema3":      0.6*rto_lag1 + 0.3*rto_lag2 + 0.1*rto_lag3,
        "rto_std3":  np.std([rto_lag1, rto_lag2, rto_lag3]),
    }

    all_data.append(data)

result_df = pd.DataFrame(all_data)
result_df.to_csv("March_data_final.csv")
print(result_df)