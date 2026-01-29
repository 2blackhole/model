import pandas as pd
import numpy as np
import glob
import xgboost as xgb
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt

# --- НАСТРОЙКИ ---
path_sensors_pattern = 'data/30/**/*сутки*.xlsx'

path_failures_list = [
    'data/30/ТНС30 2023.xlsx',
    'data/30/ТНС30 2024.xlsx',
    'data/30/ТНС30 2025.xlsx'
]

# --- 1. ЗАГРУЗКА СЕНСОРОВ ---
print(f"Поиск файлов по маске: {path_sensors_pattern}")
all_files = glob.glob(path_sensors_pattern, recursive=True)
print(f"Найдено файлов: {len(all_files)}")

df_list = []
count = 0

for filename in all_files:
    try:
        temp = pd.read_excel(filename, skiprows=6, engine='openpyxl')
        
        # Выбираем колонки
        temp = temp.iloc[:, [0, 1, 2, 4, 6, 8, 9, 10]]
        
        # Исправление времени: Превращаем дату в строку перед парсингом
        temp.iloc[:, 0] = temp.iloc[:, 0].astype(str)
        
        temp.columns = ['datetime', 'p1', 't1', 'v1', 'p2', 't2', 'v2', 'q_heat']
        
        # Парсим дату
        temp['datetime'] = pd.to_datetime(temp['datetime'], dayfirst=True, errors='coerce')
        temp = temp.dropna(subset=['datetime'])
        
        df_list.append(temp)
        
        count += 1
        if count % 200 == 0:
            print(f"Обработано {count} файлов...")
            
    except Exception as e:
        print(f"Ошибка чтения {filename}: {e}")
        continue

if not df_list:
    raise ValueError("Не удалось прочитать ни одного файла!")

print("Склейка таблиц...")
sensors = pd.concat(df_list, ignore_index=True)
sensors = sensors.sort_values('datetime').reset_index(drop=True)

for col in ['p1', 't1', 'v1', 'p2', 't2', 'v2', 'q_heat']:
    sensors[col] = pd.to_numeric(sensors[col], errors='coerce')

sensors = sensors.dropna()
sensors['date_only'] = sensors['datetime'].dt.date
print(f"Сенсоры загружены. Всего строк: {len(sensors)}")

# --- 2. ЗАГРУЗКА АВАРИЙ ---
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

if fail_list:
    failures = pd.concat(fail_list, ignore_index=True)
    failures = failures.dropna(subset=['Дата обнаружения'])
    failures['date_fail'] = pd.to_datetime(failures['Дата обнаружения'], dayfirst=True, errors='coerce').dt.date
    failures = failures.dropna(subset=['date_fail'])
    print(f"Аварий загружено: {len(failures)}")
else:
    failures = pd.DataFrame(columns=['date_fail'])

# --- 3. ПОДГОТОВКА ДАННЫХ ---
sensors['target'] = 0
LOOK_AHEAD = 2

if not failures.empty:
    for f_date in failures['date_fail']:
        start_danger = f_date - pd.Timedelta(days=LOOK_AHEAD)
        mask = (sensors['date_only'] >= start_danger) & (sensors['date_only'] <= f_date)
        sensors.loc[mask, 'target'] = 1

sensors['delta_p'] = sensors['p1'] - sensors['p2']
sensors['delta_t'] = sensors['t1'] - sensors['t2']
sensors['v_diff'] = sensors['v1'] - sensors['v2']

ROLLING_WINDOW = 24 
sensors['p1_mean_24h'] = sensors['p1'].rolling(window=ROLLING_WINDOW).mean()
sensors['p1_std_24h'] = sensors['p1'].rolling(window=ROLLING_WINDOW).std()
sensors['q_heat_mean_24h'] = sensors['q_heat'].rolling(window=ROLLING_WINDOW).mean()

df_final = sensors.dropna().drop(columns=['date_only'])

LAG_WINDOW = 6
features_to_lag = ['delta_p', 'v_diff', 'delta_t', 'q_heat_mean_24h', 'p1_mean_24h', 'p1_std_24h']

for col in features_to_lag:
    for lag in range(1, LAG_WINDOW + 1):
        df_final[f'{col}_lag_{lag}h'] = df_final[col].shift(lag)

df_final = df_final.dropna()

print("Добавляем сезонность...")
df_final['month'] = df_final['datetime'].dt.month
df_final['hour'] = df_final['datetime'].dt.hour
df_final['is_heating_season'] = df_final['month'].isin([10, 11, 12, 1, 2, 3, 4]).astype(int)

cols_to_drop = ['datetime', 'target']
feature_cols = [c for c in df_final.columns if c not in cols_to_drop]

# --- 4. ВАЛИДАЦИЯ ---
print("\nРазделение по времени (Train: Прошлое, Test: Будущее)...")
df_final = df_final.sort_values('datetime')
split_idx = int(len(df_final) * 0.80)

X = df_final[feature_cols]
y = df_final['target']

X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

print(f"Обучение: {df_final.iloc[0]['datetime']} -> {df_final.iloc[split_idx]['datetime']}")
print(f"Тест:     {df_final.iloc[split_idx]['datetime']} -> {df_final.iloc[-1]['datetime']}")

# --- 5. ОБУЧЕНИЕ (СБАЛАНСИРОВАННЫЙ РЕЖИМ) ---

# Считаем естественный дисбаланс
natural_ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

# Уменьшаем множитель с 5.0 до 2.0.
# Это всё ещё "наказывает" за пропуски, но не загоняет все вероятности в потолок.
PUNISHMENT_FACTOR = 2.0
final_weight = natural_ratio * PUNISHMENT_FACTOR

print(f"\n Баланс классов: 1 к {natural_ratio:.1f}")
print(f"Вес (scale_pos_weight): {final_weight:.1f}")

model = xgb.XGBClassifier(
    n_estimators=1500,        # Больше деревьев для точности
    learning_rate=0.01,       # Медленнее учимся = лучше распределение вероятностей
    max_depth=8,              # <-- Увеличили глубину (было 6), чтобы видеть детали
    gamma=0.2,                # Немного регуляризации, чтобы не шуметь
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=final_weight,
    eval_metric='auc',
    early_stopping_rounds=200,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

# --- ДОБАВЛЕНИЕ: ГИСТОГРАММА ВЕРОЯТНОСТЕЙ ---
# Это покажет вам, как распределились риски (должен быть "холм" или разброс от 0 до 1)
probs_check = model.predict_proba(X_test)[:, 1]
print(f"\nПроверка распределения риска:")
print(f"Min: {probs_check.min():.4f}")
print(f"Max: {probs_check.max():.4f}")
print(f"Mean: {probs_check.mean():.4f}")

plt.figure(figsize=(10, 4))
plt.hist(probs_check, bins=50, color='blue', alpha=0.7)
plt.title("Гистограмма распределения Risk Score (должна быть широкой!)")
plt.xlabel("Вероятность аварии")
plt.ylabel("Количество случаев")
plt.savefig('risk_distribution.png')
print("Гистограмма сохранена в 'risk_distribution.png'")

# --- 6. АНАЛИЗ РЕЗУЛЬТАТОВ (АВТО-ПОРОГ) ---
probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)
print(f"\nROC-AUC Score: {auc:.4f}")

# --- АВТОМАТИЧЕСКИЙ ПОДБОР ПОРОГА ---
TARGET_RECALL = 0.50  # Цель: ловить 75% аварий
print(f"\nПодбираем порог для Recall >= {TARGET_RECALL*100}%...")

precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

# Ищем индекс, где Recall >= 0.75
valid_idxs = np.where(recalls[:-1] >= TARGET_RECALL)[0]
if len(valid_idxs) > 0:
    optimal_threshold = thresholds[valid_idxs[-1]]
else:
    optimal_threshold = 0.5 

print(f"Порог найден: {optimal_threshold:.4f}")

# Применяем порог
y_pred = (probs > optimal_threshold).astype(int)

print("\nОтчет по классификации:")
print(classification_report(y_test, y_pred))

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("\nИТОГИ (в штуках):")
print(f"Поймано аварий (TP): {tp}")
print(f"Пропущено (FN):      {fn}")
print(f"Ложных тревог (FP):  {fp}")
print(f"Тишина (TN):         {tn}")

# --- 7. СОХРАНЕНИЕ ---
model.save_model("xgboost_final.json")
with open("model_features.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

# Сохраняем Excel с рисками
df_final['risk_score'] = model.predict_proba(X)[:, 1]

# Оставляем понятные колонки
report_cols = ['datetime', 'risk_score', 'target', 'p1', 't1', 'v_diff'] 
final_report_cols = [c for c in report_cols if c in df_final.columns]

df_final[final_report_cols].to_excel('final_risk_report.xlsx', index=False)

print("\nМодель сохранена в 'xgboost_final.json'")
print("Полный отчет в 'final_risk_report.xlsx'")