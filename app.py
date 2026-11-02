"""
Sri Lankan Used Car Price Predictor (Streamlit app)

Uses model.pkl (the best model from the notebook: preprocessing + XGBoost).
The model predicts log(price), so results are converted back with np.expm1.
Explanations use SHAP (same method as the notebook).
"""
import json
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

AD_YEAR = 2023   # latest ad year in the data (car_age = AD_YEAR - year of manufacture)
MAX_COMPARE = 4  # max cars in the comparison

FEATURES = ["brand", "model", "car_age", "mileage_km", "engine_cc", "condition",
            "transmission", "fuel_type", "body_type", "district", "seller_type"]

st.set_page_config(page_title="Sri Lanka Used Car Price Predictor", page_icon="🚗", layout="wide")


# ---------- Load files ----------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_json(path):
    with open(path) as f:
        return json.load(f)


model = load_model()
options = load_json("app_options.json")
insight = load_json("global_insight.json")


@st.cache_resource
def load_explainer():
    est = model.named_steps["model"] if isinstance(model, Pipeline) else model
    return shap.TreeExplainer(est)


# ---------- Helper functions ----------
def explain(car):
    """SHAP values for one car: one value per original feature, in log(price)."""
    explainer = load_explainer()
    if isinstance(model, Pipeline):
        pre = model.named_steps["preprocess"]
        est = model.named_steps["model"]
        X_pre = pre.transform(car)
        if not isinstance(est, XGBRegressor) and hasattr(X_pre, "toarray"):
            X_pre = X_pre.toarray()     # tree models from scikit-learn need dense data
        values = explainer.shap_values(X_pre)
        values = values.toarray() if hasattr(values, "toarray") else np.asarray(values)

        # Add up the one-hot columns of each original feature
        names = pre.get_feature_names_out()
        grouped = np.zeros(len(FEATURES))
        for j, f in enumerate(FEATURES):
            cols = np.array([n == "num__" + f or n.startswith("cat__" + f + "_") for n in names])
            grouped[j] = values[0, cols].sum()
    else:
        grouped = np.asarray(explainer.shap_values(car))[0]   # CatBoost

    base_value = float(np.ravel(explainer.expected_value)[0])
    return grouped, base_value


def pct(v):
    """SHAP value in log(price) -> approximate % effect on price."""
    return (np.exp(v) - 1) * 100


def predict_rs(df):
    return np.expm1(model.predict(df[FEATURES]))


def what_if(car, feature, grid):
    """Predicted price when only one feature changes (ICE curve)."""
    rows = pd.concat([car[FEATURES]] * len(grid), ignore_index=True)
    rows[feature] = grid
    return predict_rs(rows)


# ---------- Page ----------
st.title("🚗 Sri Lanka Used Car Price Predictor")
st.info(
    "Prices are based on ikman.lk car ads from **Sept 2022 – Apr 2023** "
    "(during the vehicle import ban). Today's prices may be different. "
    "Use the price only as a guide."
)

if "compare" not in st.session_state:
    st.session_state.compare = []

tab_predict, tab_compare, tab_global = st.tabs(
    ["🔮 Predict and explain", "📊 Compare cars", "🌍 Model insight"])

# ===================== Tab 1: Predict and explain =====================
with tab_predict:
    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Brand", options["brands"], index=options["brands"].index("Toyota"))
        car_model = st.selectbox("Model", options["models_by_brand"][brand])
        year = st.number_input("Year of manufacture", min_value=1990, max_value=2024, value=2015, step=1)
        mileage_km = st.number_input("Mileage (km)", min_value=0, max_value=500_000, value=80_000, step=1_000)
        engine_cc = st.number_input("Engine capacity (cc)", min_value=600, max_value=6000, value=1500, step=50)
    with col2:
        condition = st.selectbox("Condition", options["condition"])
        transmission = st.selectbox("Transmission", options["transmission"])
        fuel_type = st.selectbox("Fuel type", options["fuel_type"])
        body_type = st.selectbox("Body type", options["body_type"])
        district = st.selectbox("District", options["district"])
        seller_type = st.selectbox("Seller type", options["seller_type"])

    inputs = {
        "brand": brand, "model": car_model, "car_age": max(AD_YEAR - int(year), 0),
        "mileage_km": int(mileage_km), "engine_cc": int(engine_cc), "condition": condition,
        "transmission": transmission, "fuel_type": fuel_type, "body_type": body_type,
        "district": district, "seller_type": seller_type,
    }

    if st.button("Predict price", type="primary"):
        car = pd.DataFrame([inputs])[FEATURES]
        values, base_value = explain(car)
        st.session_state.current = {
            "inputs": inputs, "year": int(year),
            "price": float(predict_rs(car)[0]),
            "values": values.tolist(), "base": base_value,
        }

    cur = st.session_state.get("current")
    if cur is None:
        st.caption("Enter the car details and click **Predict price**.")
    elif cur["inputs"] != inputs:
        st.warning("The inputs changed. Click **Predict price** again to update the result.")
    else:
        car = pd.DataFrame([cur["inputs"]])[FEATURES]
        values = np.array(cur["values"])
        label = f'{cur["year"]} {cur["inputs"]["brand"]} {cur["inputs"]["model"]}'

        st.success(f'Estimated price: **Rs {cur["price"]:,.0f}**')
        st.caption(f'Car age used by the model: {cur["inputs"]["car_age"]} years ({AD_YEAR} − {cur["year"]})')

        # ---------- Why this price? (SHAP waterfall) ----------
        st.subheader("Why this price?")
        left, right = st.columns([3, 2])
        with left:
            car_exp = shap.Explanation(values=values, base_values=cur["base"],
                                       data=car.iloc[0].values, feature_names=FEATURES)
            shap.plots.waterfall(car_exp, max_display=11, show=False)
            st.pyplot(plt.gcf(), clear_figure=True)
            st.caption("Red bars push the price up. Blue bars push it down. The chart starts at the "
                       f"average prediction E[f(X)] (about Rs {np.expm1(cur['base']):,.0f}) and ends at this "
                       "car's prediction f(x). Values are in log(price).")
        with right:
            top = int(np.argmax(np.abs(values)))
            direction = "up" if values[top] > 0 else "down"
            st.write(f"**Biggest effect:** {FEATURES[top]} = {car[FEATURES[top]][0]} → "
                     f"price {direction} about {abs(pct(values[top])):.0f}%.")

            # ---------- Effect table (% effect) ----------
            table = pd.DataFrame({
                "Feature": FEATURES,
                "Value": [str(v) for v in car.iloc[0].values],
                "SHAP (log)": values,
                "Effect on price (%)": pct(values),
            })
            table = table.reindex(table["SHAP (log)"].abs().sort_values(ascending=False).index)
            st.dataframe(table.style.format({"SHAP (log)": "{:+.3f}", "Effect on price (%)": "{:+.1f}%"}),
                         hide_index=True, width="stretch")
            st.caption("Effect (%) = (e^SHAP − 1) × 100. The model works in log(price), "
                       "so each feature multiplies the price by e^SHAP.")

        # ---------- What-if curves (ICE) ----------
        st.subheader("What if? (change one feature, keep the others)")
        # Car age range of most cars of this brand and model in the data (5th-95th percentile)
        lo, hi = options["age_range_by_model"].get(f'{cur["inputs"]["brand"]}|{cur["inputs"]["model"]}', [0, AD_YEAR - 1990])
        lo, hi = min(lo, cur["inputs"]["car_age"]), max(hi, cur["inputs"]["car_age"])
        ages = np.arange(lo, hi + 1)
        miles = np.arange(0, 300_001, 10_000)
        price_age = what_if(car, "car_age", ages) / 1e6
        price_mile = what_if(car, "mileage_km", miles) / 1e6

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(ages, price_age, marker="o", markersize=3)
        axes[0].scatter([cur["inputs"]["car_age"]], [cur["price"] / 1e6], color="red", zorder=3, s=60, label="This car")
        axes[0].set_xlabel("Car age (years)")
        axes[0].set_ylabel("Predicted price (Rs. million)")
        axes[0].set_title("Price vs car age")
        axes[0].set_ylim(bottom=0)
        axes[0].legend()
        axes[1].plot(miles / 1000, price_mile, marker="o", markersize=3)
        axes[1].scatter([cur["inputs"]["mileage_km"] / 1000], [cur["price"] / 1e6], color="red", zorder=3, s=60, label="This car")
        axes[1].set_xlabel("Mileage (thousand km)")
        axes[1].set_ylabel("Predicted price (Rs. million)")
        axes[1].set_title("Price vs mileage")
        axes[1].set_ylim(bottom=0)
        axes[1].legend()
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.caption(f"These are ICE curves (Individual Conditional Expectation): the same car, with only one "
                   f"feature changed. The red dot is this car. The car age range ({lo}–{hi} years) covers most "
                   f"cars of this model in the data (5th to 95th percentile), so the curve does not go where there is little data. The y-axis starts at 0, so the size of the change is not "
                   f"exaggerated. Small steps in the line come from the tree splits.")

        # ---------- Add to comparison ----------
        saved = [c["label"] for c in st.session_state.compare]
        if st.button("➕ Add this car to the comparison"):
            if label in saved:
                st.info("This car is already in the comparison.")
            elif len(saved) >= MAX_COMPARE:
                st.warning(f"The comparison can have up to {MAX_COMPARE} cars. Clear it in the Compare cars tab.")
            else:
                st.session_state.compare.append({"label": label, "price": cur["price"],
                                                 "values": cur["values"], "inputs": cur["inputs"]})
                st.success(f"Added: {label}. Open the **Compare cars** tab.")

# ===================== Tab 2: Compare cars =====================
with tab_compare:
    cars = st.session_state.compare
    if len(cars) < 2:
        st.info("Add at least 2 cars from the **Predict and explain** tab to compare them.")
    else:
        st.subheader("Predicted prices")
        summary = pd.DataFrame([{
            "Car": c["label"],
            "Predicted price (Rs)": c["price"],
            "Biggest effect": FEATURES[int(np.argmax(np.abs(c["values"])))],
        } for c in cars])
        st.dataframe(summary.style.format({"Predicted price (Rs)": "{:,.0f}"}), hide_index=True,
                     width="stretch")

        fig, ax = plt.subplots(figsize=(10, 0.6 + 0.6 * len(cars)))
        ax.barh([c["label"] for c in cars], [c["price"] / 1e6 for c in cars], color="#2c7fb8")
        for i, c in enumerate(cars):
            ax.text(c["price"] / 1e6, i, f'  {c["price"] / 1e6:.2f}M', va="center")
        ax.set_xlabel("Predicted price (Rs. million)")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        st.subheader("Why are the prices different?")
        effects = pd.DataFrame({c["label"]: pct(np.array(c["values"])) for c in cars}, index=FEATURES)
        order = effects.abs().max(axis=1).sort_values(ascending=True).index
        effects = effects.loc[order]

        fig, ax = plt.subplots(figsize=(10, 6))
        n = len(cars)
        h = 0.8 / n
        y = np.arange(len(effects))
        for k, col in enumerate(effects.columns):
            ax.barh(y + k * h - 0.4 + h / 2, effects[col], height=h, label=col)
        ax.set_yticks(y)
        ax.set_yticklabels(effects.index)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Effect on price (%)  —  right = up, left = down")
        ax.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.caption("Each bar is one feature's effect for one car (from SHAP). The same model can give "
                   "different reasons for different cars.")

        st.dataframe(effects.iloc[::-1].style.format("{:+.1f}%"), width="stretch")

    if cars and st.button("🗑️ Clear the comparison"):
        st.session_state.compare = []
        st.rerun()

# ===================== Tab 3: Model insight (global) =====================
with tab_global:
    st.subheader("What did the model learn overall?")
    st.caption(f'Based on {insight["n_cars"]:,} random test cars (the same cars as in the notebook). '
               "Global view = the whole model. The other tabs show the local view = one car.")

    imp = pd.Series(insight["mean_abs_shap"]).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(imp.index, imp.values, color="#2c7fb8")
    for i, v in enumerate(imp.values):
        ax.text(v, i, f"  {v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Mean |SHAP value| (log price)")
    ax.set_title("Global feature importance")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.caption("Mean |SHAP| = the average size of a feature's effect, up or down. "
               "Car age, engine size and transmission matter most. Condition matters least, "
               "because about 99% of the ads are used cars.")

    st.subheader("Partial dependence (average effect)")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    labels = {"car_age": "Car age (years)", "engine_cc": "Engine capacity (cc)", "mileage_km": "Mileage (km)"}
    for ax, f in zip(axes, ["car_age", "engine_cc", "mileage_km"]):
        p = insight["pdp"][f]
        ax.plot(p["grid"], np.expm1(p["avg_log_price"]) / 1e6)
        ax.set_xlabel(labels[f])
        ax.set_ylabel("Average predicted price (Rs. million)")
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.caption("A PDP is the average of many ICE curves. It shows the general trend for all cars: "
               "older → cheaper, bigger engine → more expensive, mileage → small effect.")

st.caption("Model: XGBoost (best model from the notebook). "
           "Scope: cars from 1990–2024, Rs 0.5M – 100M, petrol / diesel / hybrid.")
