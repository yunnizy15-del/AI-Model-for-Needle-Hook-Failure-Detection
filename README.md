# Needle Hook Failure Classifier

This project trains a binary classifier from friction-coefficient CSV files:

- `valid/*.csv` -> label `valid` (normal)
- `invalid/*.csv` -> label `invalid` (failed)

Each CSV is treated as one sample. The model extracts time-series features from `mu_true` and trains a random forest classifier.

## 1) Train

```powershell
python train_model.py
```

Outputs:

- `model/needle_hook_model.joblib`
- `model/metrics.json`

Optional arguments:

```powershell
python train_model.py --valid-dir valid --invalid-dir invalid --test-size 0.2 --n-estimators 500
```

## 2) Predict a single CSV

```powershell
python predict_model.py --input valid\000001.csv
```

## 3) Predict a folder

```powershell
python predict_model.py --input invalid --output-csv model\invalid_predictions.csv
```

## Notes

- Required columns in CSV: `t_s`, `mu_true`
- If a few files have shorter length, feature extraction still works.

## 4) GUI (for thesis charts)

```powershell
python gui_app.py
```

If GUI startup reports missing `Tcl/Tk`, reinstall Python with `tkinter` support.

GUI features:

- Train model with configurable parameters
- Predict one CSV or a folder
- Auto export common paper-ready charts
- Preview exported charts in the GUI

Training charts:

- Class distribution
- Confusion matrix
- ROC curve
- PR curve
- Top feature importance
- Probability distribution on test set
- OOB error curve (training-process chart, optional)

Prediction charts:

- Predicted class counts
- Probability histogram
- Top-risk samples bar chart
- `mu_mean` vs `mu_std` scatter
- Risk ranking curve
- Top-risk signal curve (`t_s` vs `mu_true`)
