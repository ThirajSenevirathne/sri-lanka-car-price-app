# Sri Lanka Used Car Price Prediction

IT5514 – Applied Machine Learning · MSc in Artificial Intelligence · University of Moratuwa
Author: TTD Senevirathne (269194b)

This project predicts the price of a used car in Sri Lanka. It uses real car ads from ikman.lk.
The project compares five machine learning models, explains them with SHAP, LIME and Partial Dependence Plots, and has a web app.

**Web app:** https://269194b-sri-lanka-car-price-app.streamlit.app/

---

## Files in this repository

| File | What it is |
|---|---|
| `*.ipynb` | The notebook: data cleaning, EDA, preprocessing, training, tuning, evaluation and explainability (with all outputs) |
| `app.py` | The Streamlit web app (3 tabs: Predict and explain, Compare cars, Model insight) |
| `global_insight.json` | Global feature importance and PDP data for the Model insight tab (from 2,000 test cars) |
| `model.pkl` | The best model (scikit-learn pipeline: preprocessing + XGBoost) |
| `app_options.json` | The dropdown lists for the app (brands, models and other categories) |
| `requirements.txt` | The Python libraries and versions for the app |

---

## Dataset

- **Name:** Sri Lanka Vehicle Ads Dataset (shared by Ivantha on Kaggle)
- **Link:** https://www.kaggle.com/datasets/ivantha/sri-lanka-vehicle-ads-dataset
- **Source:** public ads from ikman.lk, September 2022 – April 2023
- **Raw size:** 258,888 ads, 26 columns (all vehicle types)
- **After cleaning:** 38,462 car ads, 11 features + 1 target

The dataset file (`vehicles.csv`, about 206 MB) is too large for GitHub. Please download it from the Kaggle link above.

**Features:** brand, model, car_age, mileage_km, engine_cc, condition, transmission, fuel_type, body_type, district, seller_type
**Target:** price_lkr (trained as log(price))

---

## Method (short)

1. **Cleaning:** kept only cars for sale, unpacked the car details, removed wrong values with domain rules, removed duplicate ads.
2. **Preprocessing:** 80/20 train/test split, one car kept aside for inference, pipeline with SimpleImputer + StandardScaler (numeric) and OneHotEncoder (categorical).
3. **Models:** Linear Regression (baseline), Random Forest, **Extra Trees (new algorithm)**, XGBoost, CatBoost.
4. **Tuning:** RandomizedSearchCV, 10 combinations, 3-fold cross-validation on the train set only.
5. **Explainability:** SHAP (beeswarm + waterfall), LIME, Partial Dependence Plots.

---

## Main results (test set, 7,693 cars)

| Model | RMSE (Rs) | MAE (Rs) | R² |
|---|---|---|---|
| **XGBoost (hand-set) – best** | 2,325,176 | 799,064 | 0.927 |
| Extra Trees (new algorithm) | 2,482,167 | 800,050 | 0.916 |
| Linear Regression (baseline) | 3,016,239 | 1,073,861 | 0.877 |

---

## How to run the app on your computer

Use **Python 3.12 or 3.13**.

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## How to run the notebook

The notebook was run on Kaggle (CPU). Add the dataset above as an input, set `DATA_DIR` in the first code cell to the dataset folder, then run all cells. The full run can take more than 1 hour, because of the tuning step.

---

## Note

The prices are from the time of the vehicle import ban (2022–23). Today's prices may be different. The app should be used only as a guide.
