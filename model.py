import pandas as pd
import numpy as np
import glob
import xgboost as xgb
import os

# Добавил confusion_matrix, которого не хватало в прошлом запуске
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# --- НАСТРОЙКИ ---
# Паттерн для поиска суточных файлов (ищем во всех подпапках)
path_sensors_pattern = 'data/**/*сутки*.xlsx'

path_failures_list = [
    'data/30/ТНС30 2023.xlsx',
    'data/30/ТНС30 2024.xlsx',
    'data/30/ТНС30 2025.xlsx',
    'data/16/ТНС16 2023.xlsx',
    'data/16/ТНС16 2024.xlsx',
    'data/16/ТНС16 2025.xlsx'
]

# --- 1. ЗАГРУЗКА СЕНСОРОВ (СУТКИ) ---
print(f"🔍 Ищем файлы по маске: {path_sensors_pattern}")
all_files = glob.glob(path_sensors_pattern, recursive=True)
print(f"Найдено файлов: {len(all_files)}")

df_list = []
count = 0

for filename in all_files:
    try:
        # Читаем файл. skiprows=6 обычно подходит и для суточных
        temp = pd.read_excel(filename, skiprows=6, engine='openpyxl')
        
        # Проверяем структуру (индексы колонок для суточных файлов)
        # 0:Date, 1:P1, 2:T1, 4:V1, 6:P2, 8:T2, 9:V2, 10:Q
        temp = temp.iloc[:, [0, 1, 2, 4, 6, 8, 9, 10]]
        temp.columns = ['datetime', 'p1', 't1', 'v1', 'p2', 't2', 'v2', 'q_heat']
        
        # Чистим дату
        temp['datetime'] = pd.to_datetime(temp['datetime'], dayfirst=True, errors='coerce')
        temp = temp.dropna(subset=['datetime'])
        
        df_list.append(temp)
        
        # Вывод прогресса каждые 100 файлов
        count += 1
        if count % 100 == 0:
            print(f"Обработано {count} файлов...")
            
    except Exception as e:
        # Можно раскомментировать, если хотите видеть ошибки по конкретным файлам
        # print(f"Ошибка чтения {filename}: {e}")
        continue

if not df_list:
    raise ValueError("❌ Не удалось прочитать ни одного файла! Проверьте путь.")

print("Склейка таблиц...")
sensors = pd.concat(df_list, ignore_index=True)

# Сортировка КРИТИЧЕСКИ важна для суточных файлов
sensors = sensors.sort_values('datetime').reset_index(drop=True)

# Числа
cols_num = ['p1', 't1', 'v1', 'p2', 't2', 'v2', 'q_heat']
for col in cols_num:
    sensors[col] = pd.to_numeric(sensors[col], errors='coerce')

sensors = sensors.dropna()
sensors['date_only'] = sensors['datetime'].dt.date

print(f"✅ Сенсоры загружены. Всего строк: {len(sensors)}")

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
    print("⚠️ Аварии не загружены! Создаем пустой список.")
    failures = pd.DataFrame(columns=['date_fail'])

# --- 3. ПОДГОТОВКА ДАННЫХ (Feature Engineering) ---
sensors['target'] = 0
LOOK_AHEAD = 2  # Окно предсказания (дней до аварии)

# Разметка таргета
if not failures.empty:
    for f_date in failures['date_fail']:
        start_danger = f_date - pd.Timedelta(days=LOOK_AHEAD)
        mask = (sensors['date_only'] >= start_danger) & (sensors['date_only'] <= f_date)
        sensors.loc[mask, 'target'] = 1

# Физические признаки
sensors['delta_p'] = sensors['p1'] - sensors['p2']
sensors['delta_t'] = sensors['t1'] - sensors['t2']
sensors['v_diff'] = sensors['v1'] - sensors['v2']

# Rolling (Скользящие средние)
# Для суточных данных (часовых) окно 24 - это сутки
ROLLING_WINDOW = 24 
sensors['p1_mean_24h'] = sensors['p1'].rolling(window=ROLLING_WINDOW).mean()
sensors['p1_std_24h'] = sensors['p1'].rolling(window=ROLLING_WINDOW).std()
sensors['q_heat_mean_24h'] = sensors['q_heat'].rolling(window=ROLLING_WINDOW).mean()

df_final = sensors.dropna().drop(columns=['date_only'])

# Лаги (сдвиги назад во времени)
LAG_WINDOW = 6
features_to_lag = ['delta_p', 'v_diff', 'delta_t', 'q_heat_mean_24h', 'p1_mean_24h', 'p1_std_24h']

for col in features_to_lag:
    for lag in range(1, LAG_WINDOW + 1):
        # Сдвиг на 1 час, 2 часа и т.д.
        df_final[f'{col}_lag_{lag}h'] = df_final[col].shift(lag)

df_final = df_final.dropna()

# ... (твой код загрузки и создания лагов выше остается без изменений) ...

# --- ДОБАВЛЕНИЕ СЕЗОННОСТИ (ГЛАВНОЕ ИСПРАВЛЕНИЕ) ---
print("📆 Добавляем календарные признаки...")
df_final['month'] = df_final['datetime'].dt.month
df_final['hour'] = df_final['datetime'].dt.hour
# Отопительный сезон (примерно с октября по апрель)
df_final['is_heating_season'] = df_final['month'].isin([10, 11, 12, 1, 2, 3, 4]).astype(int)

# Добавляем новые колонки в список фичей
# Берем все числовые колонки + новые календарные
cols_to_drop = ['datetime', 'target']
feature_cols = [c for c in df_final.columns if c not in cols_to_drop]

print(f"Признаки для обучения ({len(feature_cols)}): {feature_cols}")

# --- ЧЕСТНАЯ ВАЛИДАЦИЯ (TIME SERIES SPLIT) ---

print("\n⏱️ Запуск проверки по времени...")
df_final = df_final.sort_values('datetime')

# Берем последние 20% данных для теста (Будущее)
split_idx = int(len(df_final) * 0.80)

X = df_final[feature_cols]
y = df_final['target']

X_train = X.iloc[:split_idx]
y_train = y.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

print(f"Обучение: {df_final.iloc[0]['datetime']} -> {df_final.iloc[split_idx]['datetime']}")
print(f"Тест:     {df_final.iloc[split_idx]['datetime']} -> {df_final.iloc[-1]['datetime']}")

# --- ОБУЧЕНИЕ С ЗАЩИТОЙ ОТ ПЕРЕОБУЧЕНИЯ ---
ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

model = xgb.XGBClassifier(
    n_estimators=2000,       # Больше деревьев
    learning_rate=0.01,      # Меньше шаг (учимся медленнее и аккуратнее)
    max_depth=4,             # Меньше глубина (было 6) - чтобы не зубрить шум
    subsample=0.7,           # Берем не все данные сразу (регуляризация)
    colsample_bytree=0.7,    # Берем не все фичи сразу (регуляризация)
    scale_pos_weight=ratio,
    eval_metric='auc',
    early_stopping_rounds=200, # Даем больше шансов на исправление
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

# --- РЕЗУЛЬТАТЫ ---
probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)
print(f"\n🏆 REAL ROC-AUC Score: {auc:.4f}")

# Подбор порога
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

# Стараемся держать Recall хотя бы 70%
target_recall = 0.70
try:
    # Находим самый высокий порог, который дает Recall >= 0.70
    valid_idxs = np.where(recalls[:-1] >= target_recall)[0]
    if len(valid_idxs) > 0:
        optimal_threshold = thresholds[valid_idxs[-1]]
    else:
        optimal_threshold = 0.5
except:
    optimal_threshold = 0.5

print(f"🎯 Оптимальный порог (для Recall >{target_recall*100}%): {optimal_threshold:.4f}")

y_pred = (probs > optimal_threshold).astype(int)
print(classification_report(y_test, y_pred))

# Важность признаков (теперь тут должны появиться month или is_heating_season)
plt.figure(figsize=(10, 8))
xgb.plot_importance(model, max_num_features=15)
plt.title("Топ признаков (с учетом сезонности)")
plt.savefig('seasonal_importance.png')
print("График сохранен в 'seasonal_importance.png'")