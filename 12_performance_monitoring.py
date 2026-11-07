from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
ROOT=Path(__file__).resolve().parent

def main():
    model_p=ROOT/'models/imdb_rating_model.joblib'; x_p=ROOT/'data/processed/X_test.csv'; y_p=ROOT/'data/processed/y_test.csv'
    if not all(p.exists() for p in [model_p,x_p,y_p]): raise FileNotFoundError('Run 12_preprocess_train.py first.')
    model=joblib.load(model_p); X=pd.read_csv(x_p); y=pd.read_csv(y_p)['Rating']; pred=model.predict(X)
    metrics={'MAE':float(mean_absolute_error(y,pred)),'RMSE':float(np.sqrt(mean_squared_error(y,pred))),'R2':float(r2_score(y,pred)),'evaluated_rows':int(len(y))}
    out=ROOT/'outputs/monitoring/performance_metrics.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(metrics,indent=2))
    print('Baseline model performance:'); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
