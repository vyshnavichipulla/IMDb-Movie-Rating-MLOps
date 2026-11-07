from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent

def main():
    src = ROOT / 'data/processed/baseline_features.csv'
    if not src.exists(): raise FileNotFoundError('Run 12_preprocess_train.py first.')
    df = pd.read_csv(src)
    rng = np.random.default_rng(17)
    # Simulated monitoring shift: newer release years, longer runtimes and higher vote counts.
    df['Year'] = (df['Year'].fillna(df['Year'].median()) + rng.integers(2, 7, len(df))).clip(2000, 2035)
    df['Runtime (Minutes)'] = (df['Runtime (Minutes)'].fillna(df['Runtime (Minutes)'].median()) + rng.integers(5, 26, len(df))).clip(60, 240)
    df['Votes'] = (df['Votes'].fillna(df['Votes'].median()) * rng.uniform(1.15, 1.8, len(df))).astype(int)
    genres = df['Genre'].fillna('Unknown').astype(str).tolist()
    for i in range(0, len(genres), 4): genres[i] = 'Drama,Thriller'
    df['Genre'] = genres
    out = ROOT / 'data/monitoring/drift/evolved_data.csv'; out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f'Simulated evolved batch saved: {out.relative_to(ROOT)} ({len(df)} rows)')
    print('Note: this is simulated drift data, not newly collected real-world data.')
if __name__ == '__main__': main()
