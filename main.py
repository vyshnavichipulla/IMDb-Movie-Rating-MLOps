"""Lab 13: end-to-end IMDb MLOps lifecycle runner."""
from pathlib import Path
import json, subprocess, sys
ROOT=Path(__file__).resolve().parent
SCRIPTS=['12_preprocess_train.py','12_create_evolved_dataset.py','12_data_drift.py','12_performance_monitoring.py','12_retrain_model.py']
def main():
    report={'project':'IMDb Movie Rating Prediction','lab':'13 - End-to-End MLOps Lifecycle','steps':[]}
    for script in SCRIPTS:
        print(f'\n===== Running {script} =====',flush=True)
        result=subprocess.run([sys.executable,str(ROOT/script)],cwd=ROOT)
        report['steps'].append({'script':script,'status':'success' if result.returncode==0 else 'failed','return_code':result.returncode})
        if result.returncode != 0: break
    report['overall_status']='success' if all(s['status']=='success' for s in report['steps']) and len(report['steps'])==len(SCRIPTS) else 'failed'
    out=ROOT/'outputs/lifecycle_report.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2))
    print('\n===== LAB 13 LIFECYCLE REPORT ====='); print(json.dumps(report,indent=2))
    if report['overall_status']!='success': raise SystemExit(1)
if __name__=='__main__': main()
