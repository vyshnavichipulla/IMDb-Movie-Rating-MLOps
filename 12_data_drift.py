from pathlib import Path
import json
import pandas as pd
from scipy.stats import ks_2samp
ROOT = Path(__file__).resolve().parent
NUMERIC = ['Year', 'Runtime (Minutes)', 'Votes']
CATEGORICAL = ['Genre']

def main():
    base_p = ROOT / 'data/processed/baseline_features.csv'; new_p = ROOT / 'data/monitoring/drift/evolved_data.csv'
    if not base_p.exists() or not new_p.exists(): raise FileNotFoundError('Run preprocessing and evolved-dataset scripts first.')
    base, new = pd.read_csv(base_p), pd.read_csv(new_p)
    rows=[]
    for c in NUMERIC:
        a=pd.to_numeric(base[c], errors='coerce').dropna(); b=pd.to_numeric(new[c], errors='coerce').dropna()
        stat,p=ks_2samp(a,b)
        rows.append({'feature':c,'test':'KS','statistic':float(stat),'p_value':float(p),'drift_detected':bool(p < 0.05)})
    for c in CATEGORICAL:
        pa=base[c].fillna('Unknown').astype(str).value_counts(normalize=True)
        pb=new[c].fillna('Unknown').astype(str).value_counts(normalize=True)
        cats=pa.index.union(pb.index); distance=float(0.5*(pa.reindex(cats,fill_value=0)-pb.reindex(cats,fill_value=0)).abs().sum())
        rows.append({'feature':c,'test':'total_variation_distance','statistic':distance,'p_value':None,'drift_detected':bool(distance > 0.10)})
    out=pd.DataFrame(rows); dest=ROOT/'outputs/drift/data_drift_report.csv'; dest.parent.mkdir(parents=True,exist_ok=True); out.to_csv(dest,index=False)
    summary={'drift_detected':bool(out['drift_detected'].any()),'features_with_drift':out.loc[out.drift_detected,'feature'].tolist(),'results':rows}
    (ROOT/'outputs/drift/data_drift_summary.json').write_text(json.dumps(summary,indent=2))
    print(out.to_string(index=False)); print(f'\nOverall drift detected: {summary["drift_detected"]}')
if __name__ == '__main__': main()
