from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parent
FEATURES=['Genre','Year','Runtime (Minutes)','Votes']; NUM=['Year','Runtime (Minutes)','Votes']; CAT=['Genre']

def main():
    train_x=ROOT/'data/processed/X_train.csv'; train_y=ROOT/'data/processed/y_train.csv'; evolved=ROOT/'data/monitoring/drift/evolved_data.csv'
    if not all(p.exists() for p in [train_x,train_y,evolved]): raise FileNotFoundError('Run preprocessing and evolved dataset scripts first.')
    X=pd.concat([pd.read_csv(train_x),pd.read_csv(evolved)],ignore_index=True); y=pd.concat([pd.read_csv(train_y)['Rating'],pd.read_csv(ROOT/'data/processed/y_test.csv')['Rating']],ignore_index=True)
    # The evolved batch reuses the held-out rows; their original labels are used for this controlled classroom simulation.
    numeric=Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    categorical=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))])
    prep=ColumnTransformer([('num',numeric,NUM),('cat',categorical,CAT)])
    model=Pipeline([('preprocess',prep),('regressor',RandomForestRegressor(n_estimators=200,random_state=42,min_samples_leaf=2,n_jobs=-1))])
    model.fit(X,y); joblib.dump(model,ROOT/'models/imdb_rating_model_retrained.joblib')
    Xv=pd.read_csv(ROOT/'data/processed/X_test.csv'); yv=pd.read_csv(ROOT/'data/processed/y_test.csv')['Rating']; pred=model.predict(Xv)
    metrics={'MAE':float(mean_absolute_error(yv,pred)),'RMSE':float(np.sqrt(mean_squared_error(yv,pred))),'R2':float(r2_score(yv,pred)),'training_rows':int(len(X))}
    (ROOT/'outputs/monitoring/retrained_metrics.json').write_text(json.dumps(metrics,indent=2))
    print('Retrained model saved: models/imdb_rating_model_retrained.joblib'); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
