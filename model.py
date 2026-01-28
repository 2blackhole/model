import pandas as pd
import numpy as np
import glob 
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

path_sensors_pattern = 'data/*/**/*месяц*.xlsx' 

path_failures_list = [
    'data/30/ТНС30 2023.xlsx', 
    'data/30/ТНС30 2024.xlsx', 
    'data/30/ТНС30 2025.xlsx',
    'data/16/ТНС16 2023.xlsx', 
    'data/16/ТНС16 2024.xlsx', 
    'data/16/ТНС16 2025.xlsx'
]

all_files = glob.glob(path_sensors_pattern, recursive=True)
df_list = []

for filename in all_files:
    try:
        temp = pd.read_excel(filename, skiprows=6, engine='openpyxl')
        
        temp = temp.iloc[:, [0, 1, 2, 4, 6, 8, 9, 10]]
        temp.columns = ['datetime', 'p1', 't1', 'v1', 'p2', 't2', 'v2', 'q_heat']
        temp['datetime'] = pd.to_datetime(temp['datetime'], dayfirst=True, errors='coerce')
        temp = temp.dropna(subset=['datetime'])
        
        df_list.append(temp)
    except Exception as e:
        print(f"Ошибка чтения {filename}: {e}")
        continue

sensors = pd.concat(df_list, ignore_index=True)

sensors = sensors.sort_values('datetime').reset_index(drop=True)

for col in ['p1', 't1', 'v1', 'p2', 't2', 'v2', 'q_heat']:
    sensors[col] = pd.to_numeric(sensors[col], errors='coerce')

sensors = sensors.dropna()
sensors['date_only'] = sensors['datetime'].dt.date

print(f"Сенсоры загружены. Строк: {len(sensors)}")

print("Загрузка аварий...")
fail_list = []
for f_path in path_failures_list:
    try:
        if f_path.endswith('.csv'):
            temp_fail = pd.read_csv(f_path)
        else:
            temp_fail = pd.read_excel(f_path, engine='openpyxl')
        
        fail_list.append(temp_fail)
    except Exception as e:
        print(f"Ошибка аварий {f_path}: {e}")

failures = pd.concat(fail_list, ignore_index=True)
failures = failures.dropna(subset=['Дата обнаружения'])
failures['date_fail'] = pd.to_datetime(failures['Дата обнаружения'], dayfirst=True, errors='coerce').dt.date
failures = failures.dropna(subset=['date_fail'])

print(f"Аварий загружено: {len(failures)}")


sensors['target'] = 0
LOOK_AHEAD = 2

for f_date in failures['date_fail']:
    start_danger = f_date - pd.Timedelta(days=LOOK_AHEAD)
    mask = (sensors['date_only'] >= start_danger) & (sensors['date_only'] <= f_date)
    sensors.loc[mask, 'target'] = 1

sensors['delta_p'] = sensors['p1'] - sensors['p2']
sensors['delta_t'] = sensors['t1'] - sensors['t2']

ROLLING = 3
sensors['p1_mean_3d'] = sensors['p1'].rolling(window=ROLLING).mean()
sensors['p1_std_3d'] = sensors['p1'].rolling(window=ROLLING).std()
sensors['q_heat_mean_3d'] = sensors['q_heat'].rolling(window=ROLLING).mean()
sensors['v_diff'] = sensors['v1'] - sensors['v2']

df_final = sensors.dropna().drop(columns=['date_only'])

WINDOW_SIZE = 4
features_to_lag = ['delta_p', 'v_diff', 'delta_t', 'q_heat_mean_3d', 'p1_mean_3d', 'p1_std_3d']

for col in features_to_lag:
    for window in range(1, WINDOW_SIZE + 1):
        df_final[f'{col}_lag_{window}'] = df_final[col].shift(window)

df_final = df_final.dropna()

print(df_final.filter(like='p1').head())
print("Размер итогового датасета:", df_final.shape)
print("Баланс классов:", df_final['target'].value_counts())

X = df_final[features_to_lag]
y = df_final['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

ratio = float(np.sum(y == 0)) / np.sum(y == 1)
model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.4,
    max_depth=7,
    scale_pos_weight=ratio,
    eval_metric='auc',
    early_stopping_rounds=50,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

probs = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, probs)
print(f"\nROC-AUC Score: {auc:.4f}")

y_pred = (probs > 0.7).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

xgb.plot_importance(model, max_num_features=10)
plt.title("Топ признаков (XGBoost)")
plt.show()