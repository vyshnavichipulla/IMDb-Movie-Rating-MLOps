from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/raw/imdb.csv'
PROC = ROOT / 'data/processed'
MODELS = ROOT / 'models'
FEATURES = ['Genre', 'Year', 'Runtime (Minutes)', 'Votes']
NUMERIC = ['Year', 'Runtime (Minutes)', 'Votes']
CATEGORICAL = ['Genre']
TARGET = 'Rating'

def main():
    PROC.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        raise FileNotFoundError(f'IMDb dataset not found: {DATA}')
    df = pd.read_csv(DATA)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f'Missing required columns: {missing}')
    df = df[required].copy()
    for col in NUMERIC + [TARGET]: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=[TARGET])
    df['Genre'] = df['Genre'].fillna('Unknown').astype(str)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    numeric_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    categorical_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    prep = ColumnTransformer([('num', numeric_pipe, NUMERIC), ('cat', categorical_pipe, CATEGORICAL)])
    model = Pipeline([('preprocess', prep), ('regressor', RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2, n_jobs=-1))])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {'MAE': float(mean_absolute_error(y_test, pred)), 'RMSE': float(np.sqrt(mean_squared_error(y_test, pred))), 'R2': float(r2_score(y_test, pred)), 'train_rows': int(len(X_train)), 'test_rows': int(len(X_test))}
    joblib.dump(model, MODELS / 'imdb_rating_model.joblib')
    X_train.to_csv(PROC / 'X_train.csv', index=False); X_test.to_csv(PROC / 'X_test.csv', index=False)
    y_train.to_frame(TARGET).to_csv(PROC / 'y_train.csv', index=False); y_test.to_frame(TARGET).to_csv(PROC / 'y_test.csv', index=False)
    X_test.to_csv(PROC / 'baseline_features.csv', index=False)
    (ROOT / 'outputs/monitoring').mkdir(parents=True, exist_ok=True)
    (ROOT / 'outputs/monitoring/baseline_metrics.json').write_text(json.dumps(metrics, indent=2))
    print('Baseline model trained successfully.')
    print(json.dumps(metrics, indent=2))
    print('Saved: models/imdb_rating_model.joblib and data/processed/')

if __name__ == '__main__': main()
