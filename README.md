# Boston Housing Price Predictor

A Streamlit app that estimates a neighborhood's median home value from three features
(rooms, lower-income share, student-teacher ratio), with a likely price range and
model-quality charts.

## Run it locally

```bash
pip install -r requirements.txt
python train_model.py      # trains the models, creates model.joblib and metrics.json
streamlit run app.py       # opens http://localhost:8501
```

## Files

| File | Purpose |
|---|---|
| `train_model.py` | Compares Linear Regression, Decision Tree, Random Forest and Gradient Boosting with cross-validation, evaluates the best on a held-out test set, saves the model |
| `app.py` | The Streamlit app (3 tabs: estimate, market context, model quality) |
| `housing.csv` | The dataset |
| `model.joblib`, `metrics.json` | Output of `train_model.py`, read by the app |
| `.streamlit/config.toml` | Theme |

## Deploy for free (Streamlit Community Cloud)

1. Push this folder to a GitHub repository (keep `model.joblib` and `metrics.json` in it).
2. Go to share.streamlit.io, sign in with GitHub, choose the repo and `app.py`, click Deploy.

## Notes

- The estimate range is an 80% interval taken from cross-validated errors.
- A warning appears if inputs fall outside the range the model was trained on.
