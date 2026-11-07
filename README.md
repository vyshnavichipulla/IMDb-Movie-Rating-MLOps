# IMDb Movie Rating Prediction – MLOps Labs 12 & 13

This project adapts the Churn Labs 12/13 monitoring and lifecycle ideas to the supplied IMDb movie dataset.

## Prediction task
- Target: `Rating` (regression)
- Features: `Genre`, `Year`, `Runtime (Minutes)`, `Votes`
- Excluded: Rank, Title, Description, Director, Actors (not used in this first baseline)
- Metrics: MAE, RMSE, R²

## Open and run in VS Code (Windows PowerShell)
1. Extract this ZIP.
2. In VS Code choose **File → Open Folder** and select `IMDb_MLOps_Labs12_13`.
3. Open Terminal → New Terminal. Ensure terminal is at the project root (the folder containing `main.py`).
4. Create and activate a virtual environment:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If `py` is unavailable, use `python -m venv .venv`.
5. Install packages:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. Run Lab 12 scripts in order:
   ```powershell
   python 12_preprocess_train.py
   python 12_create_evolved_dataset.py
   python 12_data_drift.py
   python 12_performance_monitoring.py
   python 12_retrain_model.py
   ```
7. Run Lab 13 end-to-end workflow:
   ```powershell
   python main.py
   ```

## Lab 12 files
- `12_preprocess_train.py`: validates, preprocesses, trains baseline model and saves holdout data.
- `12_create_evolved_dataset.py`: creates a reproducible simulated monitoring batch from IMDb rows.
- `12_data_drift.py`: compares baseline and evolved features (KS test for numeric, categorical distribution distance).
- `12_performance_monitoring.py`: evaluates baseline predictions on the held-out test set.
- `12_retrain_model.py`: retrains a fresh model using the original training data plus the simulated evolved batch, then saves it.

## Lab 13
`main.py` runs the full lifecycle in order: preprocessing/training, evolved data creation, drift analysis, performance monitoring, retraining, and a JSON lifecycle report.

## Notes
- The evolved batch is simulated for a classroom drift-monitoring demonstration; it is not newly collected real-world IMDb data.
- The dataset has only 1,000 rows. Results are educational and should not be treated as a production-quality movie-rating predictor.
- Run commands from the project root. Do not run individual scripts from inside `data/`.
