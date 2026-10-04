"""
Boston Housing Price Predictor  -  Streamlit app
Run:  streamlit run app.py
(first run  python train_model.py  to create model.joblib and metrics.json)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Boston Housing Price Predictor",
                   page_icon="🏠", layout="wide")

BASE = Path(__file__).parent
BLUE, AMBER, GREY = "#1F4E79", "#E08A00", "#8A97A6"

LABELS = {
    "RM": "Average rooms per home",
    "LSTAT": "Lower-income residents (%)",
    "PTRATIO": "Students per teacher",
}

# ----------------------------------------------------------------------------
# Cached loaders: the model and data are read once, not on every click
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(BASE / "model.joblib")


@st.cache_data
def load_data():
    return pd.read_csv(BASE / "housing.csv")


@st.cache_data
def load_metrics():
    return json.loads((BASE / "metrics.json").read_text())


if not (BASE / "model.joblib").exists():
    st.error("model.joblib was not found. Run `python train_model.py` first, "
             "then restart the app.")
    st.stop()

model, data, metrics = load_model(), load_data(), load_metrics()
ranges = metrics["feature_ranges"]
# slider bounds: whole numbers just outside the data range
bounds = {k: (float(np.floor(v[0])), float(np.ceil(v[1]))) for k, v in ranges.items()}

# ----------------------------------------------------------------------------
# Sidebar inputs
# ----------------------------------------------------------------------------
EXAMPLES = {   # the three clients from the project brief
    "Client 1": (5.0, 17.0, 15.0),
    "Client 2": (4.0, 32.0, 22.0),
    "Client 3": (8.0, 3.0, 12.0),
}

for key, default in zip(("RM", "LSTAT", "PTRATIO"), (6.0, 12.0, 18.0)):
    st.session_state.setdefault(key, default)


def load_example(name):
    rm, lstat, ptr = EXAMPLES[name]
    st.session_state.update(RM=rm, LSTAT=lstat, PTRATIO=ptr)


with st.sidebar:
    st.header("Describe the home")
    rm = st.slider(LABELS["RM"], *bounds["RM"],
                   key="RM", step=0.1,
                   help="Average number of rooms among homes in the neighborhood.")
    lstat = st.slider(LABELS["LSTAT"], *bounds["LSTAT"], key="LSTAT", step=0.5,
                      help="Share of residents considered lower-income.")
    ptratio = st.slider(LABELS["PTRATIO"], *bounds["PTRATIO"], key="PTRATIO", step=0.5,
                        help="Students per teacher in nearby schools.")

    st.divider()
    st.caption("Load an example from the project brief")
    cols = st.columns(3)
    for col, name in zip(cols, EXAMPLES):
        col.button(name, on_click=load_example, args=(name,))

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
user = pd.DataFrame([[rm, lstat, ptratio]], columns=["RM", "LSTAT", "PTRATIO"])
price = float(model.predict(user)[0])
low = max(price + metrics["interval"]["low"], 0)
high = price + metrics["interval"]["high"]
median_price = float(data["MEDV"].median())
percentile = float((data["MEDV"] < price).mean() * 100)
diff_pct = (price - median_price) / median_price * 100

st.title("Boston Housing Price Predictor")
st.write(f"Estimates a neighborhood's median home value from three features. "
         f"Model: **{metrics['best_model']}**, tested on homes it never saw during training.")

tab_predict, tab_explore, tab_model = st.tabs(
    ["Your estimate", "Where it sits in the market", "How good is the model?"])

# ----------------------------------------------------------------------------
# Tab 1: the estimate
# ----------------------------------------------------------------------------
with tab_predict:
    c1, c2, c3 = st.columns([1.3, 1, 1])
    c1.metric("Estimated price", f"${price:,.0f}")
    c2.metric("Vs. market median", f"{diff_pct:+.0f}%",
              help=f"Median price in the data is ${median_price:,.0f}.")
    c3.metric("Higher than", f"{percentile:.0f}% of homes")

    st.write(f"**Likely range (80% of the time):** ${low:,.0f} to ${high:,.0f}")

    # price distribution with the estimate marked
    fig = px.histogram(data, x="MEDV", nbins=30, color_discrete_sequence=[GREY])
    fig.add_vline(x=price, line_color=AMBER, line_width=3,
                  annotation_text="Estimate", annotation_font_color=AMBER)
    fig.add_vrect(x0=low, x1=high, fillcolor=AMBER, opacity=0.12, line_width=0)
    fig.update_layout(height=320, bargap=0.05, showlegend=False,
                      margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="Home price ($)", yaxis_title="Neighborhoods")
    st.plotly_chart(fig)

    # inputs outside the training range -> be honest about it
    out = [LABELS[c] for c, v in zip(("RM", "LSTAT", "PTRATIO"), (rm, lstat, ptratio))
           if not ranges[c][0] <= v <= ranges[c][1]]
    if out:
        st.warning("Outside the range the model was trained on: " + ", ".join(out))

# ----------------------------------------------------------------------------
# Tab 2: feature scatter plots with the user's input marked
# ----------------------------------------------------------------------------
with tab_explore:
    st.write("Each dot is a neighborhood. The orange line is your input.")
    for feat, val, title in (("RM", rm, "Rooms vs. price"),
                             ("LSTAT", lstat, "Lower-income share vs. price"),
                             ("PTRATIO", ptratio, "Student-teacher ratio vs. price")):
        corr = data[feat].corr(data["MEDV"])
        fig = px.scatter(data, x=feat, y="MEDV", opacity=0.55,
                         color_discrete_sequence=[BLUE],
                         labels={feat: LABELS[feat], "MEDV": "Price ($)"})
        fig.add_vline(x=val, line_color=AMBER, line_width=3, line_dash="dash",
                      annotation_text="Your input", annotation_font_color=AMBER)
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0))
        left, right = st.columns([3, 1])
        with left:
            st.subheader(title)
            st.plotly_chart(fig)
        with right:
            direction = "rises" if corr > 0 else "falls"
            st.metric("Correlation with price", f"{corr:+.2f}")
            st.caption(f"Price generally {direction} as this value goes up.")

# ----------------------------------------------------------------------------
# Tab 3: model quality
# ----------------------------------------------------------------------------
with tab_model:
    t = metrics["test"]
    m1, m2, m3 = st.columns(3)
    m1.metric("R² on unseen data", f"{t['r2']:.2f}",
              help="1.0 is perfect; 0 is no better than always guessing the average.")
    m2.metric("Typical error (MAE)", f"${t['mae']:,.0f}")
    m3.metric("RMSE", f"${t['rmse']:,.0f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Models compared")
        comp = pd.DataFrame(metrics["comparison"]).sort_values("cv_r2_mean")
        fig = px.bar(comp, x="cv_r2_mean", y="model", orientation="h",
                     error_x="cv_r2_std", range_x=[0, 1],
                     color_discrete_sequence=[BLUE],
                     labels={"cv_r2_mean": "Cross-validated R²", "model": ""})
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig)
        st.caption("Scores come from 10 shuffled train/validation splits of the "
                   "training data. The best one is used above.")
    with right:
        st.subheader("What drives the price")
        imp = pd.DataFrame({"feature": [LABELS[k] for k in metrics["importance"]],
                            "importance": list(metrics["importance"].values())}
                           ).sort_values("importance")
        fig = px.bar(imp, x="importance", y="feature", orientation="h",
                     color_discrete_sequence=[BLUE], labels={"feature": ""})
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig)

    st.subheader("Predicted vs. actual (test set)")
    tp = pd.DataFrame(metrics["test_points"])
    fig = px.scatter(tp, x="actual", y="predicted", opacity=0.7,
                     color_discrete_sequence=[BLUE],
                     labels={"actual": "Actual price ($)", "predicted": "Predicted price ($)"})
    lim = [tp[["actual", "predicted"]].min().min(), tp[["actual", "predicted"]].max().max()]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Perfect prediction",
                             line=dict(color=AMBER, dash="dash")))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig)
    st.caption("Dots close to the dashed line are accurate predictions.")

st.caption("Data: Boston housing dataset (489 neighborhoods). Estimates are "
           "statistical and not a formal appraisal.")
