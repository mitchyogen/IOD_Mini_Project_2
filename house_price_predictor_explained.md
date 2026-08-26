# HDB Resale House Price Predictor — Code Walkthrough

This is a **Streamlit web app** that loads a trained machine learning model and lets a user enter house details (area, bedrooms, bathrooms) to get a predicted price. Below, the code is broken into separate blocks, each with a plain-English explanation, so you can follow the logic step by step.

---

## Block 1 — Imports and Setup

```python
import os
import joblib
import pandas as pd
import streamlit as st
```

**What's happening:**

- `os` — checks whether the model file actually exists on disk before trying to load it.
- `joblib` — used to load a machine learning model that was previously _saved_ (trained) elsewhere, usually in a Jupyter notebook.
- `pandas` — turns user input into a table (DataFrame), because most ML models expect tabular input, not a plain dictionary.
- `streamlit` — the library that turns this Python script into an interactive web app (buttons, sliders, forms, etc.) without writing any HTML/CSS/JS.

---

## Block 2 — Configuration

```python
MODEL_PATH = "house_price_linear_regression.joblib"

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
)
```

**What's happening:**

- `MODEL_PATH` is a constant — the filename of the saved model. Keeping it as a variable at the top means you only need to change it in one place if the filename changes.
- `st.set_page_config()` must be the **first Streamlit command** in the script. It sets the browser tab title, the tab icon (a house emoji 🏠), and centers the page layout instead of using the full width.

> 💡 **Student tip:** If you move `st.set_page_config()` below any other `st.` command, Streamlit will throw an error. Order matters here.

---

## Block 3 — Loading the Model (with Caching)

```python
@st.cache_resource
def load_model():
    """Load the trained model. Cached so it only loads once per session."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)
```

**What's happening:**

- `@st.cache_resource` is a **decorator**. It tells Streamlit: "Run this function once, then remember the result." Without it, Streamlit would reload the model file from disk _every single time_ the user interacts with the app (e.g., every click) — slow and wasteful.
- Inside the function: first it checks if the file exists using `os.path.exists()`. If not, it returns `None` instead of crashing.
- If the file _does_ exist, `joblib.load()` reads the saved model back into memory so it can be used for predictions.

> 💡 **Student tip:** Caching is a core Streamlit concept. `@st.cache_resource` is for things like ML models or database connections — objects you want to create once and reuse.

---

## Block 4 — The Prediction Function

```python
def predict(model, row: dict) -> float:
    df = pd.DataFrame([row])
    pred = model.predict(df)[0]
    return float(pred)
```

**What's happening:**

- This function takes the trained `model` and a `row` (a dictionary like `{"area_sqft": 1800, "bedrooms": 3, "bathrooms": 2}`).
- `pd.DataFrame([row])` wraps that single dictionary in a list, then converts it into a one-row table. Models trained with scikit-learn expect this table format, not a raw dictionary.
- `model.predict(df)` returns an **array** of predictions (even for one row, it's still a list-like object) — that's why `[0]` is used to grab just the first (and only) value.
- `float(pred)` converts the result from a NumPy number type into a plain Python float, which is safer for display and formatting later.

---

## Block 5 — Page Title and Model Loading Check

```python
st.title("House Price Predictor")
st.caption("Predict your House Price.")

model = load_model()

if model is None:
    st.error(
        f"Model file not found. Expected `{MODEL_PATH}` in the same folder "
        "as this script.\n\n"
        "Run the training notebook first (it saves the model with "
        "`joblib.dump(model, \"house_price_linear_regression.joblib\")`), "
        "then place that file next to `main.py`."
    )
    st.stop()
```

**What's happening:**

- `st.title()` and `st.caption()` display the big heading and the small gray subtitle at the top of the page.
- `model = load_model()` actually calls the cached function from Block 3.
- The `if model is None:` check is a **guard clause** — if the model file wasn't found, it shows a friendly red error box explaining exactly what to do, then calls `st.stop()`.
- `st.stop()` immediately halts execution of the rest of the script. This prevents the app from crashing later with a confusing error when it tries to use a model that doesn't exist.

> 💡 **Student tip:** This is good defensive programming — instead of letting the app crash with a cryptic Python traceback, it fails gracefully with instructions for the user.

---

## Block 6 — The Input Form

```python
st.markdown(
    """
    <div class="hero">
        <h1>🏠 HDB Resale Price Predictor</h1>
        <p>Estimate the resale value of an HDB flat using its location, property characteristics, storey, remaining lease and transaction timing.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Model target", "Resale Price")
with m2:
    st.metric("Prediction type", "Regression")
with m3:
    st.metric("Model status", "Ready to connect")

st.write("")
left, right = st.columns([1.65, 1], gap="large")

with left:
    st.markdown('<div class="section-label">Property details</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Tell us about the flat</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Enter the same information used by the trained model.</div>', unsafe_allow_html=True)

    with st.form("hdb_prediction_form"):
        c1, c2 = st.columns(2)

        with c1:
            town = st.selectbox("Town", TOWNS, index=TOWNS.index("BISHAN"))
            flat_type = st.selectbox("Flat type", FLAT_TYPES, index=FLAT_TYPES.index("4 ROOM"))
            flat_model = st.selectbox("Flat model", FLAT_MODELS, index=FLAT_MODELS.index("Model A"))
            floor_area_sqm = st.number_input("Floor area (sqm)", min_value=20.0, max_value=400.0, value=93.0, step=1.0)

        with c2:
            storey_midpoint = st.number_input("Approximate storey", min_value=1.0, max_value=60.0, value=11.0, step=1.0, help="Use the midpoint of the HDB storey band, e.g. 10–12 → 11.")
            remaining_lease_years = st.number_input("Remaining lease (years)", min_value=0.0, max_value=99.0, value=71.0, step=0.1)
            transaction_year = st.selectbox("Transaction year", list(range(2017, CURRENT_YEAR + 2)), index=len(list(range(2017, CURRENT_YEAR + 2))) - 1)
            transaction_month = st.selectbox("Transaction month", list(range(1, 13)), index=max(datetime.now().month - 1, 0), format_func=lambda x: datetime(2000, x, 1).strftime("%B"))

        st.write("")
        submitted = st.form_submit_button("✨ Predict Resale Price", use_container_width=True, type="primary")
```

**What's happening:**

- `st.form(...)` groups multiple inputs together so the app **doesn't rerun on every single input change** — it only reruns once, when the user clicks the submit button. This is more efficient and avoids a jumpy user experience.
- `st.columns(3)` splits the form into three side-by-side columns for a cleaner layout.
- `st.selectbox()` creates a drop down selection of values — good for categorical values and reduces risk of incorrect values.
- `st.number_input()` creates a numeric field with min/max bounds and a step size — good for continuous values like square footage.
- `st.form_submit_button()` is the button that, when clicked, sets `submitted = True` and triggers the rest of the code to run.

---

## Block 7 — Running the Prediction

```python
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
    st.markdown('<div class="section-copy">Your model output will appear here after prediction logic is connected.</div>', unsafe_allow_html=True)

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

```

**What's happening:**

- `if submitted:` — this block only runs _after_ the user clicks the "Predict Price" button.
- `input_values` packages the three form inputs into a dictionary, matching the format the `predict()` function expects.
- `predict(model, input_values)` calls Block 4's function to get the estimated price.
- `st.success(...)` shows the result in a green success box, formatted as currency with commas and two decimal places (`:,.2f`).
- The `try/except` block catches any unexpected errors (e.g., a mismatched column name or a corrupted model) and shows a readable red error message instead of crashing the whole app.

---

## Big-Picture Summary

| Step | What it does                                |
| ---- | ------------------------------------------- |
| 1–2  | Import libraries, configure the page        |
| 3    | Load the ML model once (cached)             |
| 4    | Define how to turn inputs into a prediction |
| 5    | Show the title and confirm the model loaded |
| 6    | Collect user input via a form               |
| 7    | Run the prediction and display the result   |

**Core flow:** User fills form → clicks submit → inputs become a DataFrame → model predicts → result is displayed.

This structure (config → load resources → define helpers → build UI → handle submission) is a common pattern for small Streamlit ML apps, so it's worth recognizing and reusing.


## Assumptions

**HDB Loan Restrictions**: HDB loans require a flat to have a minimum of 20 years of remaining lease.

**Floor Area (sqm)**: The lowest floor area ever recorded in an official HDB resale transaction is 31 square metres (sqm) (approximately 334 square feet).
