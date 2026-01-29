import xgboost as xgb
import pandas as pd
import argparse
import os
import json

def main():
    model = xgb.XGBClassifier()
    model.load_model('xgboost_final.json')

    parser = argparse.ArgumentParser(description='Загрузка файла для обработки')
    
    parser.add_argument('--file', 
                       type=str, 
                       required=True, 
                       help='Путь к файлу для загрузки')
    
    args = parser.parse_args()
    if not os.path.exists(args.file):
        print(f"Ошибка: Файл '{args.file}' не найден")
        return


    sensors = pd.read_json(args.file)
    sensors['datetime'] = pd.to_datetime(sensors['datetime'], dayfirst=True, errors='coerce')
    sensors = sensors.dropna(subset=['datetime'])

    for col in ['p1', 't1', 'v1', 'p2', 't2', 'v2', 'q_heat']:
        sensors[col] = pd.to_numeric(sensors[col], errors='coerce')

    sensors = sensors.dropna()
    sensors['date_only'] = sensors['datetime'].dt.date

    sensors['delta_p'] = sensors['p1'] - sensors['p2']
    sensors['delta_t'] = sensors['t1'] - sensors['t2']
    sensors['v_diff'] = sensors['v1'] - sensors['v2']
    ROLLING_WINDOW = 24
    sensors['p1_mean_24h'] = sensors['p1'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    sensors['p1_std_24h'] = sensors['p1'].rolling(window=ROLLING_WINDOW, min_periods=1).std()
    sensors['q_heat_mean_24h'] = sensors['q_heat'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    df_final = sensors.dropna().drop(columns=['date_only'])
    LAG_WINDOW = 6
    features_to_lag = ['delta_p', 'v_diff', 'delta_t', 'q_heat_mean_24h', 'p1_mean_24h', 'p1_std_24h']
    for col in features_to_lag:
        for lag in range(1, LAG_WINDOW + 1):
            df_final[f'{col}_lag_{lag}h'] = df_final[col].shift(lag)
    df_final = df_final.iloc[LAG_WINDOW:].reset_index(drop=True)
    df_final['month'] = df_final['datetime'].dt.month
    df_final['hour'] = df_final['datetime'].dt.hour
    df_final['is_heating_season'] = df_final['month'].isin([10, 11, 12, 1, 2, 3, 4]).astype(int)
    cols_to_drop = ['datetime']   
    feature_cols = [c for c in df_final.columns if c not in cols_to_drop]
    df_final = df_final.sort_values('datetime')
    df_final = df_final[feature_cols]
    
    res = model.predict_proba(df_final)[:, 1][-1]
    
    with open('out.json', 'w', encoding='utf-8') as f:
        json.dump(res.tolist(), f, 
                ensure_ascii=False, 
                indent=2,            
                sort_keys=False)


if __name__ == "__main__":
    main()