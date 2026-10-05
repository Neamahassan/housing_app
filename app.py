import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="Boston Housing Price Predictor",
                   page_icon="🏠", layout="wide")

BASE = Path(__file__).parent


TEXT = "#2F2A25"
MUTED = "#7A7167"
SAGE = "#5B7F6B"        # primary
SAGE_SOFT = "#9DB5A6"   # scatter points
SAND = "#DDD3C3"        # histogram bars
OCHRE = "#B7791F"       # "your input" / estimate
GRID = "#E7DFD2"
PANEL = "#F3EDE3"
BORDER = "#E4DACB"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap');

html, body, .stApp, .stMarkdown, button, input, label, p, span, div {{
    font-family: 'DM Sans', system-ui, sans-serif;
}}
h1, h2, h3, [data-testid="stMetricValue"], .hero-price {{
    font-family: 'Newsreader', Georgia, serif !important;
    letter-spacing: -0.01em;
}}
h1 {{ font-weight: 500 !important; font-size: 2.6rem !important; }}
h2, h3 {{ font-weight: 500 !important; color: {TEXT}; }}

.block-container {{ padding-top: 2.6rem; max-width: 1180px; }}
footer {{ visibility: hidden; }}

.subtitle {{ color: {MUTED}; font-size: 1.05rem; margin: -0.4rem 0 1.4rem 0; max-width: 62ch; }}

/* hero estimate panel */
.hero {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 14px;
    padding: 1.6rem 1.8rem;
}}
.hero-label {{ color: {MUTED}; font-size: 0.95rem; }}
.hero-price {{ font-size: 3.6rem; font-weight: 500; color: {TEXT}; line-height: 1.1; margin: 0.15rem 0 0.5rem 0; }}
.hero-range {{ color: {TEXT}; font-size: 1rem; }}
.hero-range b {{ font-weight: 600; }}
.hero-note {{ color: {MUTED}; font-size: 0.85rem; margin-top: 0.3rem; }}

/* metrics */
[data-testid="stMetric"] {{
    background: transparent; border-left: 3px solid {BORDER}; padding: 0.2rem 0 0.2rem 1rem;
}}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.92rem; }}
[data-testid="stMetricValue"] {{ font-weight: 500; font-size: 2rem; }}

/* tabs */
button[data-baseweb="tab"] p {{ font-size: 1rem; font-weight: 500; }}

/* sidebar */
section[data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}
section[data-testid="stSidebar"] h2 {{ font-size: 1.5rem; }}
.stButton button {{
    border: 1px solid {BORDER}; background: #FBF8F3; color: {TEXT};
}}
.stButton button {{ width: 100%; white-space: nowrap; }}
.stButton button:hover {{ border-color: {SAGE}; color: {SAGE}; }}
</style>
""", unsafe_allow_html=True)

LABELS = {
    "RM": "Average rooms per home",
    "LSTAT": "Lower-income residents (%)",
    "PTRATIO": "Students per teacher",
}


def style(fig, height=340, price_axis=None):
    """Apply one consistent look to every chart."""
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color=TEXT, size=13),
        hoverlabel=dict(bgcolor="#FBF8F3", font_color=TEXT, bordercolor=BORDER),
        legend=dict(orientation="h", y=1.1, x=0),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID, title_font_color=MUTED)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID, title_font_color=MUTED)
    if price_axis:
        getattr(fig, f"update_{price_axis}axes")(tickprefix="$", tickformat="~s")
    return fig



@st.cache_data
def load_data():
    return pd.read_csv(BASE / "housing.csv")


@st.cache_data
def load_metrics():
    return json.loads((BASE / "metrics.json").read_text())


@st.cache_resource
def load_model():
    """Load the saved model. If the file is missing or was saved with a different
    library version (common on cloud hosts), retrain the same model on the spot.
    Training takes about a second, so the app always works."""
    try:
        return joblib.load(BASE / "model.joblib")
    except Exception:
        df = load_data()
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42
        ).fit(df[["RM", "LSTAT", "PTRATIO"]], df["MEDV"])


model, data, metrics = load_model(), load_data(), load_metrics()
ranges = metrics["feature_ranges"]
bounds = {k: (float(np.floor(v[0])), float(np.ceil(v[1]))) for k, v in ranges.items()}


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
    rm = st.slider(LABELS["RM"], *bounds["RM"], key="RM", step=0.1,
                   help="Average number of rooms among homes in the neighborhood.")
    lstat = st.slider(LABELS["LSTAT"], *bounds["LSTAT"], key="LSTAT", step=0.5,
                      help="Share of residents considered lower-income.")
    ptratio = st.slider(LABELS["PTRATIO"], *bounds["PTRATIO"], key="PTRATIO", step=0.5,
                        help="Students per teacher in nearby schools.")

    st.divider()
    st.caption("Load an example from the project brief")
    for name in EXAMPLES:
        st.button(name, on_click=load_example, args=(name,))

user = pd.DataFrame([[rm, lstat, ptratio]], columns=["RM", "LSTAT", "PTRATIO"])
price = float(model.predict(user)[0])
low = max(price + metrics["interval"]["low"], 0)
high = price + metrics["interval"]["high"]
median_price = float(data["MEDV"].median())
percentile = float((data["MEDV"] < price).mean() * 100)
diff_pct = (price - median_price) / median_price * 100

st.title("Boston Housing Price Predictor")
st.markdown(
    f'<p class="subtitle">Estimates a neighborhood\'s median home value from three '
    f'features. Model: <b>{metrics["best_model"]}</b>, checked on homes it never '
    f'saw during training.</p>', unsafe_allow_html=True)

tab_predict, tab_explore, tab_model = st.tabs(
    ["Your estimate", "Where it sits in the market", "How good is the model?"])


with tab_predict:
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        # &#36; is a dollar sign; it avoids Streamlit reading $...$ as math
        st.markdown(f"""
<div class="hero">
  <div class="hero-label">Estimated price</div>
  <div class="hero-price">&#36;{price:,.0f}</div>
  <div class="hero-range">Likely range: <b>&#36;{low:,.0f}</b> to <b>&#36;{high:,.0f}</b></div>
  <div class="hero-note">The real price falls inside this range about 80% of the time.</div>
</div>""", unsafe_allow_html=True)
    with right:
        st.metric("Compared with the market median", f"{diff_pct:+.0f}%",
                  help=f"Median price in the data is \\${median_price:,.0f}.")
        st.metric("Priced above", f"{percentile:.0f}% of neighborhoods")

    st.subheader("Where this estimate falls")
    fig = px.histogram(data, x="MEDV", nbins=30, color_discrete_sequence=[SAND])
    fig.add_vrect(x0=low, x1=high, fillcolor=OCHRE, opacity=0.10, line_width=0)
    fig.add_vline(x=price, line_color=OCHRE, line_width=3,
                  annotation_text="Estimate", annotation_font_color=OCHRE)
    fig.update_layout(bargap=0.06, showlegend=False,
                      xaxis_title="Home price", yaxis_title="Neighborhoods")
    st.plotly_chart(style(fig, 320, "x"))

    # inputs outside the training range -> be honest about it
    out = [LABELS[c] for c, v in zip(("RM", "LSTAT", "PTRATIO"), (rm, lstat, ptratio))
           if not ranges[c][0] <= v <= ranges[c][1]]
    if out:
        st.warning("Outside the range the model was trained on: " + ", ".join(out)
                   + ". Treat this estimate with extra caution.")


with tab_explore:
    st.write("Each dot is a neighborhood. The dashed line marks your input.")
    for feat, val, title in (("RM", rm, "Rooms and price"),
                             ("LSTAT", lstat, "Lower-income share and price"),
                             ("PTRATIO", ptratio, "Student-teacher ratio and price")):
        corr = data[feat].corr(data["MEDV"])
        fig = px.scatter(data, x=feat, y="MEDV", opacity=0.6,
                         color_discrete_sequence=[SAGE_SOFT],
                         labels={feat: LABELS[feat], "MEDV": "Price"})
        fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color=SAGE)))
        fig.add_vline(x=val, line_color=OCHRE, line_width=3, line_dash="dash",
                      annotation_text="Your input", annotation_font_color=OCHRE)
        c_left, c_right = st.columns([3, 1], gap="large")
        with c_left:
            st.subheader(title)
            st.plotly_chart(style(fig, 330, "y"))
        with c_right:
            direction = "rises" if corr > 0 else "falls"
            st.metric("Correlation with price", f"{corr:+.2f}")
            st.caption(f"Price generally {direction} as this value goes up.")


with tab_model:
    t = metrics["test"]
    m1, m2, m3 = st.columns(3)
    m1.metric("R² on unseen data", f"{t['r2']:.2f}",
              help="1.0 is perfect; 0 is no better than always guessing the average.")
    m2.metric("Typical error (MAE)", f"${t['mae']:,.0f}")
    m3.metric("RMSE", f"${t['rmse']:,.0f}")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Models compared")
        comp = pd.DataFrame(metrics["comparison"]).sort_values("cv_r2_mean")
        comp["chosen"] = np.where(comp["model"] == metrics["best_model"], "Selected", "Other")
        fig = px.bar(comp, x="cv_r2_mean", y="model", orientation="h", error_x="cv_r2_std",
                     range_x=[0, 1], color="chosen",
                     color_discrete_map={"Selected": SAGE, "Other": SAND},
                     labels={"cv_r2_mean": "Cross-validated R²", "model": ""})
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 320))
        st.caption("Scores come from 10 shuffled train/validation splits of the "
                   "training data. The green bar is the model used above.")
    with c2:
        st.subheader("What drives the price")
        imp = pd.DataFrame({"feature": [LABELS[k] for k in metrics["importance"]],
                            "importance": list(metrics["importance"].values())}
                           ).sort_values("importance")
        fig = px.bar(imp, x="importance", y="feature", orientation="h",
                     color_discrete_sequence=[SAGE], labels={"feature": "", "importance": "Share of influence"})
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(style(fig, 320))
        st.caption("How much the model relies on each input.")

    st.subheader("Predicted and actual prices (test set)")
    tp = pd.DataFrame(metrics["test_points"])
    fig = px.scatter(tp, x="actual", y="predicted", opacity=0.7,
                     color_discrete_sequence=[SAGE_SOFT],
                     labels={"actual": "Actual price", "predicted": "Predicted price"})
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color=SAGE)))
    lim = [tp[["actual", "predicted"]].min().min(), tp[["actual", "predicted"]].max().max()]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Perfect prediction",
                             line=dict(color=OCHRE, dash="dash")))
    fig.update_layout(showlegend=False)
    st.plotly_chart(style(fig, 380, "x"))
    st.caption("Dots close to the dashed line are accurate predictions.")

st.caption("Data: Boston housing dataset (489 neighborhoods). Estimates are "
           "statistical and not a formal appraisal.")
