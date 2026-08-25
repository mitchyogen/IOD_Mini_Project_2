import streamlit as st
import os
import joblib
import pandas as pd
from datetime import datetime

import sklearn
import xgboost
import numpy as np

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_PATH = "hdb_resale_price_model.joblib"

st.set_page_config(
    page_title="HDB Resale House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

# --------------------------------------------------------------------------
# Cached loader
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the trained model. Cached so it only loads once per session."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def predict(bundle, row: dict) -> float:
    df = pd.DataFrame([row])
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    X_prepared = preprocessor.transform(df)
    pred = model.predict(X_prepared)[0]
    return float(pred)


# --------------------------------------------------------------------------
# App layout
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="HDB Resale Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

model = load_model()

st.markdown(
    """
    <style>
        .stApp {background: linear-gradient(180deg, #F5F9FF 0%, #FFFFFF 45%, #F8FAFC 100%);}
        .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
        .hero {padding: 2rem 2.2rem; border-radius: 24px; background: linear-gradient(135deg, #0B3C6F 0%, #1769AA 55%, #43A5D9 100%); color: white; box-shadow: 0 16px 40px rgba(11,60,111,.18); margin-bottom: 1.5rem;}
        .hero h1 {margin: 0; font-size: 2.35rem; line-height: 1.1; font-weight: 800;}
        .hero p {margin-top: .75rem; margin-bottom: 0; max-width: 760px; font-size: 1rem; color: rgba(255,255,255,.88);}
        .section-label {color: #0B3C6F; font-size: .82rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; margin-bottom: .2rem;}
        .section-title {color: #12263A; font-size: 1.35rem; font-weight: 800; margin-bottom: .25rem;}
        .section-copy {color: #64748B; font-size: .93rem; margin-bottom: 1rem;}
        .prediction-card {border-radius: 22px; padding: 1.6rem; background: linear-gradient(135deg, #EAF5FF 0%, #F7FBFF 100%); border: 1px solid #CDE7FA; text-align: center; min-height: 190px; display: flex; flex-direction: column; justify-content: center;}
        .prediction-label {color: #557087; font-size: .85rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em;}
        .prediction-value {color: #0B3C6F; font-size: 2.6rem; font-weight: 900; margin: .35rem 0;}
        .prediction-note {color: #6B7F93; font-size: .88rem;}
        div[data-testid="stMetric"] {background: #FFF; border: 1px solid #E7EEF7; padding: .9rem 1rem; border-radius: 16px;}
        div[data-testid="stForm"] {background: rgba(255,255,255,.88); border: 1px solid #E7EEF7; border-radius: 22px; padding: 1.3rem; box-shadow: 0 10px 30px rgba(15,45,75,.05);}
        .footer-note {margin-top: 2rem; text-align: center; color: #8292A5; font-size: .82rem;}
        .stSelectbox > label {color: #000000;}
        .stNumberInput > label {color: #000000;}
        .stElementContainer {color: #000000;}
        [data-testid="stVerticalBlock"] {color: #000000;}
        [data-testid="stMarkdownContainer"] {color: orange;}
        [data-testid="stMarkdownContainer"] > p {color: #000000;}
    </style>
    """,
    unsafe_allow_html=True,
)

TOWNS = [
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH",
    "BUKIT PANJANG", "BUKIT TIMAH", "CENTRAL AREA", "CHOA CHU KANG",
    "CLEMENTI", "GEYLANG", "HOUGANG", "JURONG EAST", "JURONG WEST",
    "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "PUNGGOL",
    "QUEENSTOWN", "SEMBAWANG", "SENGKANG", "SERANGOON", "TAMPINES",
    "TOA PAYOH", "WOODLANDS", "YISHUN"
]

FLAT_TYPES = [
    "1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM",
    "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"
]

FLAT_MODELS = [
    "Improved", "New Generation", "Model A", "Standard", "Simplified",
    "Apartment", "Maisonette", "Premium Apartment", "Model A2", "DBSS",
    "Type S1", "Type S2", "Adjoined flat", "Terrace", "Multi Generation",
    "Premium Apartment Loft", "2-room", "3Gen"
]

CURRENT_YEAR = datetime.now().year

st.markdown(
    """
    <div class="hero">
        <h1>🏠 HDB Resale Price Predictor</h1>
        <p>Estimate the resale value of an HDB flat using its location, property characteristics, storey, remaining lease and transaction timing.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
left, right = st.columns([1.65, 1], gap="large")

with left:
    st.markdown('<div class="section-label">Property details</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Tell us about the flat</div>', unsafe_allow_html=True)

    with st.form("hdb_prediction_form"):
        c1, c2 = st.columns(2)

        with c1:
            town = st.selectbox("Town", TOWNS, index=TOWNS.index("BISHAN"))
            flat_type = st.selectbox("Flat type", FLAT_TYPES, index=FLAT_TYPES.index("4 ROOM"))
            flat_model = st.selectbox("Flat model", FLAT_MODELS, index=FLAT_MODELS.index("Model A"))
            floor_area_sqm = st.number_input("Floor area (sqm)", min_value=31, max_value=200, value=93, step=1)
            
        with c2:
            storey_midpoint = st.number_input("Approximate storey", min_value=1, max_value=60, value=11, step=1, help="Use the midpoint of the HDB storey band, e.g. 10–12 → 11.")
            remaining_lease_years = st.number_input("Remaining lease (years)", min_value=0.0, max_value=99.0, value=71.0, step=0.1)
            transaction_year = st.selectbox("Transaction year", list(range(2017, CURRENT_YEAR + 2)), index=len(list(range(2017, CURRENT_YEAR + 2))) - 1)
            transaction_month = st.selectbox("Transaction month", list(range(1, 13)), index=max(datetime.now().month - 1, 0), format_func=lambda x: datetime(2000, x, 1).strftime("%B"))

        st.write("")
        submitted = st.form_submit_button("✨ Predict Resale Price", use_container_width=True, type="primary")

if submitted:
    try:
        if model is None:
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. "
                "Place the trained .joblib file in the same folder as app.py."
            )

        input_values = {
            "town": town,
            "flat_type": flat_type,
            "flat_model": flat_model,
            "floor_area_sqm": floor_area_sqm,
            "storey_midpoint": storey_midpoint,
            "remaining_lease_years": remaining_lease_years,
            "transaction_year": transaction_year,
            "transaction_month": transaction_month,
        }

        price = predict(model, input_values)
        st.session_state["predicted_price"] = price
        st.session_state.pop("prediction_error", None)

    except Exception as e:
        st.session_state["prediction_error"] = str(e)


with right:
    st.markdown('<div class="section-label">Estimate</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Predicted resale price</div>', unsafe_allow_html=True)

    if "prediction_error" in st.session_state:
        st.error(f"Couldn't generate a prediction: {st.session_state['prediction_error']}")
    elif "predicted_price" in st.session_state:
        st.success(
            f"### Estimated Price: S${st.session_state['predicted_price']:,.2f}"
        )
    else:
        st.markdown(
            """
            <div class="prediction-card">
                <div class="prediction-label">Estimated Resale Price</div>
                <div class="prediction-value">S$ —</div>
                <div class="prediction-note">
                    Complete the property details and run the prediction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    with st.container(border=True):
        st.markdown("#### Prediction summary")
        st.caption("A quick review of the user inputs will appear here.")
        s1, s2 = st.columns(2)
        with s1:
            st.write("**Town**")
            st.write(town)
            st.write("**Flat type**")
            st.write(flat_type)
            st.write("**Floor area**")
            st.write(f"{floor_area_sqm:.0f} sqm")
        with s2:
            st.write("**Storey**")
            st.write(f"{storey_midpoint:.0f}")
            st.write("**Remaining lease**")
            st.write(f"{remaining_lease_years:.1f} years")
            st.write("**Transaction period**")
            st.write(f"{datetime(2000, transaction_month, 1).strftime('%B')} {transaction_year}")

st.write("")
with st.container(border=True):
    st.write(
        """
        This interface is designed for an HDB resale-price regression model trained on historical resale transactions. The final prediction should be treated as an analytical estimate rather than an official valuation.

        The model connection and prediction logic are intentionally not included in this layout-only version.
        """
    )

st.markdown('<div class="footer-note">HDB Resale Price Predictor · Streamlit interface</div>', unsafe_allow_html=True)
