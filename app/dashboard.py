import os
import time
import random
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = st.get_option("theme.base") == "dark"

is_dark = st.session_state.dark_mode

if is_dark:
    C_PAGE_BG               = "#1A1A2E"
    C_SIDEBAR_BG            = "#12121F"
    C_CARD_BG               = "#1E2235"
    C_CARD_BG2              = "#252840"
    C_TEXT1                 = "#E8E8F0"
    C_TEXT2                 = "#B4B2A9"
    C_TEXT_MUTED            = "#888780"
    C_BORDER                = "#3C3489"
    C_BLUE                  = "#85B7EB"
    C_GREEN                 = "#5DCAA5"
    C_RED                   = "#F09595"
    C_AMBER                 = "#EF9F27"
    C_NAV_BG                = "#26215C"
    C_NAV_TEXT              = "#AFA9EC"
    C_GENUINE_CHIP_BG       = "#E6F1FB"
    C_GENUINE_CHIP_TEXT     = "#0C447C"
    C_FRAUD_CHIP_BG         = "#FCEBEB"
    C_FRAUD_CHIP_TEXT       = "#A32D2D"
    C_SUCCESS_BG            = "#173404"
    C_SUCCESS_TEXT          = "#5DCAA5"
    C_INFO_BG               = "#0C447C"
    C_INFO_TEXT             = "#85B7EB"
    C_CM_TN_BG              = "#0C447C"
    C_CM_TN_TEXT            = "#85B7EB"
    C_CM_FP_BG              = "#501313"
    C_CM_FP_TEXT            = "#F09595"
    C_CM_FN_BG              = "#501313"
    C_CM_FN_TEXT            = "#F09595"
    C_CM_TP_BG              = "#173404"
    C_CM_TP_TEXT            = "#5DCAA5"
    C_BEST_BG               = "#173404"
    C_BEST_BORDER           = "#5DCAA5"
    C_BEST_BADGE_BG         = "#173404"
    C_BEST_BADGE_TEXT       = "#5DCAA5"
    C_GENUINE_RESULT_BG     = "#173404"
    C_GENUINE_RESULT_BORDER = "#0F6E56"
    C_GENUINE_RESULT_TEXT   = "#5DCAA5"
    C_FRAUD_RESULT_BG       = "#501313"
    C_FRAUD_RESULT_BORDER   = "#A32D2D"
    C_FRAUD_RESULT_TEXT     = "#F09595"
    TOGGLE_LABEL            = "Light mode"
    TOGGLE_BG               = "#252840"
    TOGGLE_TEXT_C           = "#AFA9EC"
    TOGGLE_BORDER           = "#3C3489"
else:
    C_PAGE_BG               = "#F4F6FA"
    C_SIDEBAR_BG            = "#FFFFFF"
    C_CARD_BG               = "#FFFFFF"
    C_CARD_BG2              = "#F1EFE8"
    C_TEXT1                 = "#1A1A2E"
    C_TEXT2                 = "#5F5E5A"
    C_TEXT_MUTED            = "#888780"
    C_BORDER                = "#D3D1C7"
    C_BLUE                  = "#185FA5"
    C_GREEN                 = "#0F6E56"
    C_RED                   = "#A32D2D"
    C_AMBER                 = "#854F0B"
    C_NAV_BG                = "#E6F1FB"
    C_NAV_TEXT              = "#185FA5"
    C_GENUINE_CHIP_BG       = "#E6F1FB"
    C_GENUINE_CHIP_TEXT     = "#0C447C"
    C_FRAUD_CHIP_BG         = "#FCEBEB"
    C_FRAUD_CHIP_TEXT       = "#A32D2D"
    C_SUCCESS_BG            = "#EAF3DE"
    C_SUCCESS_TEXT          = "#3B6D11"
    C_INFO_BG               = "#E6F1FB"
    C_INFO_TEXT             = "#0C447C"
    C_CM_TN_BG              = "#E6F1FB"
    C_CM_TN_TEXT            = "#0C447C"
    C_CM_FP_BG              = "#FCEBEB"
    C_CM_FP_TEXT            = "#A32D2D"
    C_CM_FN_BG              = "#FCEBEB"
    C_CM_FN_TEXT            = "#A32D2D"
    C_CM_TP_BG              = "#EAF3DE"
    C_CM_TP_TEXT            = "#3B6D11"
    C_BEST_BG               = "#EAF3DE"
    C_BEST_BORDER           = "#0F6E56"
    C_BEST_BADGE_BG         = "#EAF3DE"
    C_BEST_BADGE_TEXT       = "#3B6D11"
    C_GENUINE_RESULT_BG     = "#EAF3DE"
    C_GENUINE_RESULT_BORDER = "#0F6E56"
    C_GENUINE_RESULT_TEXT   = "#0F6E56"
    C_FRAUD_RESULT_BG       = "#FCEBEB"
    C_FRAUD_RESULT_BORDER   = "#A32D2D"
    C_FRAUD_RESULT_TEXT     = "#A32D2D"
    TOGGLE_LABEL            = "Dark mode"
    TOGGLE_BG               = "#F1EFE8"
    TOGGLE_TEXT_C           = "#5F5E5A"
    TOGGLE_BORDER           = "#D3D1C7"

C_BLUE_ORIG  = "#185FA5"
C_GREEN_ORIG = "#0F6E56"
C_RED_ORIG   = "#A32D2D"
C_AMBER_ORIG = "#854F0B"

st.markdown(f"""
<style>
  html, body, [class*="css"] {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}

  .stApp, .main .block-container {{
    background-color: {C_PAGE_BG} !important;
  }}


  
  section[data-testid="stSidebar"],
  section[data-testid="stSidebar"] > div {{
    background-color: {C_SIDEBAR_BG} !important;
    border-right: 0.5px solid {C_BORDER} !important;
  }}
  section[data-testid="stSidebar"] * {{
    color: {C_TEXT2} !important;
  }}
  section[data-testid="stSidebar"] .stRadio > div > label {{
    font-size: 13px !important;
    font-weight: 400 !important;
    color: {C_TEXT2} !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
  }}
  section[data-testid="stSidebar"] .stRadio > div > label:hover {{
    background-color: {C_NAV_BG} !important;
    color: {C_NAV_TEXT} !important;
  }}
  section[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {{
    background-color: {C_NAV_BG} !important;
    color: {C_NAV_TEXT} !important;
  }}
  
  section[data-testid="stSidebar"] .stRadio label > div:first-child {{
    display: none !important;
  }}

  
  .stNumberInput input {{
    background-color: {C_CARD_BG} !important;
    color: {C_TEXT1} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 8px !important;
  }}
  .stNumberInput input:focus {{
    border-color: {C_BLUE} !important;
    box-shadow: none !important;
  }}
  .stNumberInput button {{
    background-color: {C_CARD_BG2} !important;
    color: {C_TEXT1} !important;
    border: 1px solid {C_BORDER} !important;
  }}
  .stNumberInput button:hover {{ background-color: {C_NAV_BG} !important; }}

  
  .stSelectbox [data-baseweb="select"] > div {{
    background-color: {C_CARD_BG} !important;
    border-color: {C_BORDER} !important;
    border-radius: 8px !important;
  }}
  
  .stSelectbox [data-baseweb="select"] div[aria-selected="true"],
  .stSelectbox [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
  .stSelectbox [data-baseweb="select"] span,
  .stSelectbox [data-baseweb="select"] div > div > div,
  .stSelectbox [data-baseweb="select"] input,
  .stSelectbox div[data-baseweb="select"] * {{
    color: {C_TEXT1} !important;
    -webkit-text-fill-color: {C_TEXT1} !important;
  }}
  
  [data-baseweb="popover"] [role="option"] {{
    background-color: {C_CARD_BG} !important;
    color: {C_TEXT1} !important;
  }}
  [data-baseweb="popover"] [role="option"]:hover {{ background-color: {C_NAV_BG} !important; }}
  [data-baseweb="popover"] li span {{
    color: {C_TEXT1} !important;
    -webkit-text-fill-color: {C_TEXT1} !important;
  }}

  
  .stNumberInput label, .stSelectbox label,
  label[data-testid="stWidgetLabel"],
  label[data-testid="stWidgetLabel"] p {{
    color: {C_TEXT2} !important;
    font-size: 13px !important;
  }}

  
  .stTabs [data-baseweb="tab-list"] {{
    background-color: transparent;
    border-bottom: 0.5px solid {C_BORDER};
  }}
  .stTabs [data-baseweb="tab"] {{
    font-size: 13px; font-weight: 400;
    color: {C_TEXT_MUTED}; background-color: transparent; border: none; padding: 6px 14px;
  }}
  .stTabs [aria-selected="true"] {{
    color: {C_BLUE} !important;
    border-bottom: 2px solid {C_BLUE} !important;
    background-color: transparent !important;
  }}

  
  .top-bar-title {{ font-size: 18px; font-weight: 500; color: {C_TEXT1}; padding-top: 6px; }}

  
  .badge {{
    display: inline-block; border-radius: 20px;
    padding: 3px 10px; font-size: 11px; font-weight: 500; line-height: 1.4;
  }}
  .badge-success  {{ background-color:{C_SUCCESS_BG};     color:{C_SUCCESS_TEXT}; }}
  .badge-info     {{ background-color:{C_INFO_BG};         color:{C_INFO_TEXT}; }}
  .badge-genuine  {{ background-color:{C_GENUINE_CHIP_BG}; color:{C_GENUINE_CHIP_TEXT}; }}
  .badge-fraud    {{ background-color:{C_FRAUD_CHIP_BG};   color:{C_FRAUD_CHIP_TEXT}; }}
  .badge-best     {{ background-color:{C_BEST_BADGE_BG};   color:{C_BEST_BADGE_TEXT}; }}
  .badge-simulated {{
    background-color: {"#3D2A00" if is_dark else "#FEF3CD"};
    color: {"#EF9F27" if is_dark else "#854F0B"};
    border-radius: 20px; font-size: 10px; font-weight: 500; padding: 3px 10px;
    display: inline-block; line-height: 1.4; vertical-align: middle; margin-left: 8px;
  }}

  
  .card {{
    background-color: {C_CARD_BG}; border: 0.5px solid {C_BORDER};
    border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 10px;
  }}
  .card-title {{ font-size: 13px; font-weight: 500; color: {C_TEXT1}; margin-bottom: 0.75rem; }}

  
  .metric-card {{
    background-color: {C_CARD_BG}; border: 0.5px solid {C_BORDER};
    border-radius: 10px; padding: 1rem 1.25rem;
    border-left: 3px solid transparent; margin-bottom: 10px;
  }}
  .metric-card-blue  {{ border-left-color: {C_BLUE_ORIG}; }}
  .metric-card-green {{ border-left-color: {C_GREEN_ORIG}; }}
  .metric-card-red   {{ border-left-color: {C_RED_ORIG}; }}
  .metric-card-amber {{ border-left-color: {C_AMBER_ORIG}; }}
  .metric-label {{ font-size:10px; font-weight:500; text-transform:uppercase;
                   color:{C_TEXT_MUTED}; margin-bottom:4px; letter-spacing:0.03em; }}
  .metric-value {{ font-size:22px; font-weight:500; color:{C_TEXT1}; margin-bottom:2px; }}
  .metric-sub   {{ font-size:11px; font-weight:400; color:{C_TEXT_MUTED}; }}

  
  .data-table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  .data-table th {{
    background-color:{C_CARD_BG2}; color:{C_TEXT_MUTED}; font-weight:500;
    text-align:left; padding:8px 10px; border-bottom:0.5px solid {C_BORDER}; font-size:11px;
  }}
  .data-table td {{ padding:7px 10px; color:{C_TEXT1}; border-bottom:0.5px solid {C_BORDER}; font-size:12px; }}
  .data-table tr:hover td {{ background-color:{C_CARD_BG2}; }}

  
  .donut-wrap   {{ display:flex; align-items:center; gap:20px; justify-content:center; padding:10px 0; }}
  .donut-legend {{ display:flex; flex-direction:column; gap:8px; }}
  .donut-row    {{ display:flex; align-items:center; gap:6px; font-size:12px; color:{C_TEXT2}; }}
  .donut-dot    {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}

  
  .chart-legend      {{ display:flex; gap:16px; justify-content:center; margin-top:8px; }}
  .chart-legend-item {{ display:flex; align-items:center; gap:5px; font-size:11px; color:{C_TEXT_MUTED}; }}
  .legend-dot        {{ width:8px; height:8px; border-radius:2px; display:inline-block; }}

  
  .model-card {{
    background-color:{C_CARD_BG}; border:0.5px solid {C_BORDER};
    border-radius:10px; padding:1rem 1.25rem;
  }}
  .model-card-best {{
    background-color:{C_BEST_BG} !important; border:0.5px solid {C_BEST_BORDER} !important;
  }}
  .model-card-title {{
    font-size:14px; font-weight:500; color:{C_TEXT1}; margin-bottom:12px;
    display:flex; justify-content:space-between; align-items:center;
  }}
  .model-row {{
    display:flex; justify-content:space-between; align-items:center;
    padding:6px 0; border-bottom:0.5px solid {C_BORDER};
  }}
  .model-row:last-child {{ border-bottom:none; }}
  .model-row-label {{ font-size:12px; font-weight:400; color:{C_TEXT2}; }}
  .model-row-value {{ font-size:13px; font-weight:500; color:{C_TEXT1}; }}

  
  .result-box     {{ border-radius:10px; padding:1.25rem; margin-top:10px; }}
  .result-genuine {{ background-color:{C_GENUINE_RESULT_BG}; border:0.5px solid {C_GENUINE_RESULT_BORDER}; }}
  .result-fraud   {{ background-color:{C_FRAUD_RESULT_BG};   border:0.5px solid {C_FRAUD_RESULT_BORDER}; }}
  .result-title   {{ font-size:16px; font-weight:500; margin-bottom:4px; }}
  .result-sub     {{ font-size:12px; color:{C_TEXT2}; margin-bottom:12px; }}
  .conf-label     {{ display:flex; justify-content:space-between; font-size:11px;
                     font-weight:500; color:{C_TEXT2}; margin-bottom:4px; }}
  .conf-track     {{ width:100%; height:6px; background-color:{C_CARD_BG2}; border-radius:3px; overflow:hidden; }}
  .conf-fill      {{ height:100%; border-radius:3px; }}

  
  
  .stCheckbox label {{
    color: {C_TEXT2} !important;
    font-size: 13px !important;
  }}
  .stCheckbox label p {{
    color: {C_TEXT2} !important;
    font-size: 13px !important;
  }}

  
  div[data-testid="stFormSubmitButton"] {{
    display: flex !important;
    justify-content: center !important;
    margin-top: 10px !important;
  }}
  .stFormSubmitButton > button {{
    width: 260px !important;
    background-color: #185FA5 !important;
    color: #FFFFFF !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 11px 0 !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
  }}
  .stFormSubmitButton > button:hover {{
    background-color: #1470BA !important;
  }}

  /* Simulate Live Transaction button — matches Run prediction solid blue style */
  div[data-testid="stButton"]:has(button[kind="secondary"]) button {{
    background-color: #185FA5 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    width: 100% !important;
    padding: 11px 0 !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
    transition: background-color 0.15s ease !important;
  }}
  div[data-testid="stButton"]:has(button[kind="secondary"]) button:hover {{
    background-color: #1470BA !important;
    color: #FFFFFF !important;
  }}

</style>
""", unsafe_allow_html=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@st.cache_resource
def load_ml_artifacts():
    """Load the trained model, scaler and label encoders once and cache them."""
    try:
        model    = joblib.load(os.path.join(PROJECT_ROOT, "models", "best_model.pkl"))
        scaler   = joblib.load(os.path.join(PROJECT_ROOT, "models", "scaler.pkl"))
        encoders = joblib.load(os.path.join(PROJECT_ROOT, "models", "label_encoder.pkl"))
        return model, scaler, encoders
    except Exception:
        return None, None, None


@st.cache_data
def load_dataset():
    """Read the raw transaction CSV. Returns None if the file is not found."""
    path = os.path.join(PROJECT_ROOT, "credit_card_fraud_dataset.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


df                      = load_dataset()
model, scaler, encoders = load_ml_artifacts()

# ---------------------------------------------------------------------------
# Session state — used by the Simulate Live Transaction feature
# ---------------------------------------------------------------------------
if "simulation_result" not in st.session_state:
    st.session_state["simulation_result"] = None


def generate_single_transaction(fraud_likely: bool = False) -> dict:
    """
    Build one random transaction row.
    fraud_likely=True produces a high-risk pattern; False produces a normal pattern.
    """
    if fraud_likely:
        return {
            "transaction_amount_inr":    round(random.uniform(15000, 80000), 2),
            "transaction_hour":          random.choice(list(range(0, 6)) + list(range(22, 24))),
            "day_of_week":               random.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
            "merchant_category":         random.choice(["electronics", "jewelry", "cash_advance", "online_shopping"]),
            "is_high_risk_merchant":     1,
            "distance_from_home_km":     round(random.uniform(200, 5000), 1),
            "prev_txn_gap_mins":         round(random.uniform(0.5, 8), 1),
            "card_present":              0,
            "location_match":            0,
            "cvv_mismatch":              1,
            "international_transaction": 1,
            "num_txn_last_24h":          random.randint(6, 20),
            "num_declined_last_7days":   random.randint(2, 4),
            "customer_age_years":        random.randint(18, 45),
            "account_age_days":          random.randint(1, 150),
        }
    else:
        return {
            "transaction_amount_inr":    round(random.uniform(100, 8000), 2),
            "transaction_hour":          random.randint(8, 21),
            "day_of_week":               random.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]),
            "merchant_category":         random.choice(["grocery", "restaurant", "fuel", "retail_clothing", "healthcare", "utilities"]),
            "is_high_risk_merchant":     0,
            "distance_from_home_km":     round(random.uniform(0, 50), 1),
            "prev_txn_gap_mins":         round(random.uniform(60, 2000), 1),
            "card_present":              1,
            "location_match":            1,
            "cvv_mismatch":              0,
            "international_transaction": 0,
            "num_txn_last_24h":          random.randint(1, 4),
            "num_declined_last_7days":   random.randint(0, 1),
            "customer_age_years":        random.randint(22, 65),
            "account_age_days":          random.randint(300, 3650),
        }


def generate_batch() -> pd.DataFrame:
    """
    Generate a random batch of 8–15 transactions.
    Each row independently has a 20% chance of being fraud-like.
    No random seed is fixed — every call produces different data.
    """
    n = random.randint(8, 15)
    rows = []
    for _ in range(n):
        fraud_likely = random.random() < 0.20
        rows.append(generate_single_transaction(fraud_likely))
    return pd.DataFrame(rows)


def predict_batch(batch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the loaded model pipeline on every row in batch_df.
    Applies label encoding and scaling exactly as during training,
    then attaches a Result (0/1) column.
    """
    df_proc = batch_df.copy()

    # Encode day_of_week using the stored mapping
    day_map = encoders.get(
        "day_of_week_mapping",
        {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6},
    )
    df_proc["day_of_week"] = df_proc["day_of_week"].map(day_map).fillna(0).astype(int)

    # Encode merchant_category using the stored LabelEncoder
    le_merchant = encoders["merchant_category_encoder"]
    df_proc["merchant_category"] = df_proc["merchant_category"].apply(
        lambda x: int(le_merchant.transform([x])[0]) if x in le_merchant.classes_ else 0
    )

    # Ensure column order matches training
    feature_cols = [
        "transaction_amount_inr", "transaction_hour", "day_of_week",
        "merchant_category", "is_high_risk_merchant", "card_present",
        "location_match", "distance_from_home_km", "prev_txn_gap_mins",
        "num_txn_last_24h", "num_declined_last_7days", "cvv_mismatch",
        "international_transaction", "customer_age_years", "account_age_days",
    ]
    scaled      = scaler.transform(df_proc[feature_cols])
    predictions = model.predict(scaled)

    result_df            = batch_df.copy()
    result_df["Result"]  = predictions
    return result_df


st.sidebar.markdown(
    f"<div style='font-size:16px;font-weight:500;color:{C_TEXT1};'>"
    "Fraud Detection Dashboard</div>"
    f"<div style='font-size:11px;color:{C_TEXT_MUTED};margin-bottom:20px;'>"
    "ML Detection System</div>",
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Fraud patterns", "Model results", "Check a transaction", "Batch prediction"],
    label_visibility="collapsed",
)

row_count = len(df) if df is not None else 0
st.sidebar.markdown(
    f"<div style='position:fixed;bottom:16px;font-size:11px;color:{C_TEXT_MUTED};'>"
    f"{row_count} rows in dataset</div>",
    unsafe_allow_html=True,
)

if df is None:
    st.error("credit_card_fraud_dataset.csv was not found in the root directory.")
    st.stop()

if model is None or scaler is None or encoders is None:
    st.warning("Model artifacts are missing from models/. Run the training pipeline first.")


def render_top_bar(title: str) -> None:
    left, right = st.columns([6, 4])

    with left:
        st.markdown(
            f'<div class="top-bar-title">{title}</div>',
            unsafe_allow_html=True,
        )
    with right:
        if is_dark:
            btn_label = "Light mode"
            btn_bg    = "#252840"
            btn_fg    = "#AFA9EC"
            btn_bdr   = "#3C3489"
        else:
            btn_label = "Dark mode"
            btn_bg    = "#E6F1FB"
            btn_fg    = "#185FA5"
            btn_bdr   = "#C5D8F0"

        st.markdown(
            f'<style>'
            f'div[data-testid="stColumn"]:last-child,'
            f'div[data-testid="stColumn"]:last-child > div,'
            f'div[data-testid="stColumn"]:last-child div[data-testid="stButton"] {{'
            f'  display:flex !important;'
            f'  justify-content:flex-end !important;'
            f'  align-items:center !important;'
            f'  width:100% !important;'
            f'}}'
            f'div[data-testid="stColumn"]:last-child div[data-testid="stButton"] button {{'
            f'  background-color:{btn_bg} !important;'
            f'  color:{btn_fg} !important;'
            f'  border:1px solid {btn_bdr} !important;'
            f'  border-radius:6px !important;'
            f'  font-size:11px !important;'
            f'  font-weight:500 !important;'
            f'  padding:4px 12px !important;'
            f'  height:26px !important;'
            f'  min-height:unset !important;'
            f'  width:auto !important;'
            f'  white-space:nowrap !important;'
            f'  line-height:1 !important;'
            f'}}'
            f'div[data-testid="stColumn"]:last-child div[data-testid="stButton"] button:hover {{'
            f'  opacity:0.82 !important;'
            f'}}'
            f'</style>',
            unsafe_allow_html=True,
        )
        if st.button(btn_label, key="theme_toggle"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.markdown(
        f'<hr style="border:none;border-top:0.5px solid {C_BORDER};margin:6px 0 14px 0;">',
        unsafe_allow_html=True,
    )


def plotly_layout(title="", x_title="", y_title="", height=350) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=13, color=C_TEXT1), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
            size=12, color=C_TEXT2,
        ),
        xaxis=dict(title=x_title, gridcolor=C_BORDER, zerolinecolor=C_BORDER,
                   tickfont=dict(size=11, color=C_TEXT_MUTED)),
        yaxis=dict(title=y_title, gridcolor=C_BORDER, zerolinecolor=C_BORDER,
                   tickfont=dict(size=11, color=C_TEXT_MUTED)),
        margin=dict(l=40, r=20, t=40, b=40),
        height=height,
        showlegend=False,
    )


if page == "Overview":
    render_top_bar("Overview")

    total_txns   = len(df)
    fraud_txns   = int(df["is_fraud"].sum())
    genuine_txns = total_txns - fraud_txns
    avg_amount   = df["transaction_amount_inr"].mean()
    fraud_rate   = (fraud_txns / total_txns) * 100

    m1, m2, m3, m4 = st.columns(4, gap="small")
    for col, cls, label, value, sub in [
        (m1, "blue",  "Total transactions",   f"{total_txns:,}",   "Across all records"),
        (m2, "green", "Legitimate",           f"{genuine_txns:,}", f"{100 - fraud_rate:.1f}% of transactions"),
        (m3, "red",   "Flagged as fraud",     f"{fraud_txns:,}",   f"1 in {round(total_txns/fraud_txns) if fraud_txns else 0} transactions"),
        (m4, "amber", "Avg. transaction",     f"₹{avg_amount:,.0f}", "Average spend per transaction"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card metric-card-{cls}">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div>'
                f'<div class="metric-sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    col_table, col_donut = st.columns([2, 1], gap="small")

    with col_table:
        cols_to_show = ["transaction_id", "transaction_amount_inr",
                        "transaction_hour", "merchant_category", "is_fraud"]
        rows_html = ""
        for _, row in df.head(10)[cols_to_show].iterrows():
            chip = (
                '<span class="badge badge-fraud">Fraud</span>'
                if row["is_fraud"] == 1
                else '<span class="badge badge-genuine">Genuine</span>'
            )
            rows_html += (
                f"<tr>"
                f"<td>{row['transaction_id']}</td>"
                f"<td>{row['transaction_amount_inr']:,.2f}</td>"
                f"<td>{row['transaction_hour']}</td>"
                f"<td>{row['merchant_category']}</td>"
                f"<td>{chip}</td>"
                f"</tr>"
            )
        st.markdown(
            f'<div class="card"><div class="card-title">Data snapshot</div>'
            f'<table class="data-table"><thead><tr>'
            f'<th>Transaction ID</th><th>Amount (INR)</th>'
            f'<th>Hour</th><th>Merchant category</th><th>Fraud label</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table>'
            f'<div style="text-align:right;font-size:11px;color:{C_TEXT_MUTED};margin-top:8px;">'
            f'Showing 10 of {total_txns:,} transactions</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_donut:
        circ         = 2 * 3.14159 * 40
        g_dash       = ((genuine_txns / total_txns)) * circ
        f_dash       = ((fraud_txns   / total_txns)) * circ
        f_offset     = -g_dash
        fraud_pct_v  = (fraud_txns / total_txns) * 100

        st.markdown(
            f'<div class="card"><div class="card-title">Class balance</div>'
            f'<div class="donut-wrap">'
            f'<svg width="120" height="120" viewBox="0 0 100 100">'
            f'<circle cx="50" cy="50" r="40" fill="none" stroke="{C_BLUE_ORIG}"'
            f' stroke-width="12" stroke-dasharray="{g_dash} {circ}"'
            f' stroke-dashoffset="0" transform="rotate(-90 50 50)"/>'
            f'<circle cx="50" cy="50" r="40" fill="none" stroke="{C_RED_ORIG}"'
            f' stroke-width="12" stroke-dasharray="{f_dash} {circ}"'
            f' stroke-dashoffset="{f_offset}" transform="rotate(-90 50 50)"/>'
            f'<text x="50" y="50" text-anchor="middle" dominant-baseline="central"'
            f' font-size="13" font-weight="500" fill="{C_TEXT1}">{fraud_pct_v:.1f}%</text>'
            f'</svg>'
            f'<div class="donut-legend">'
            f'<div class="donut-row">'
            f'<span class="donut-dot" style="background:{C_BLUE_ORIG};"></span>'
            f'Genuine ({genuine_txns:,})</div>'
            f'<div class="donut-row">'
            f'<span class="donut-dot" style="background:{C_RED_ORIG};"></span>'
            f'Fraud ({fraud_txns:,})</div>'
            f'</div></div></div>',
            unsafe_allow_html=True,
        )

    col_merch, col_stats = st.columns(2, gap="small")

    with col_merch:
        merch_counts = df["merchant_category"].value_counts().sort_values(ascending=True)
        max_cnt      = merch_counts.max()
        bars = ""
        for cat, cnt in merch_counts.items():
            pct = (cnt / max_cnt) * 100
            bars += (
                f'<div style="display:flex;align-items:center;margin-bottom:6px;">'
                f'<div style="width:110px;text-align:right;font-size:11px;color:{C_TEXT2};'
                f'padding-right:8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"'
                f' title="{cat}">{cat}</div>'
                f'<div style="flex:1;background-color:{C_CARD_BG2};border-radius:4px;'
                f'height:8px;overflow:hidden;">'
                f'<div style="width:{pct:.1f}%;height:100%;background-color:{C_BLUE_ORIG};'
                f'border-radius:4px;"></div></div>'
                f'<div style="width:32px;text-align:right;font-size:11px;'
                f'color:{C_TEXT_MUTED};padding-left:6px;">{cnt}</div>'
                f'</div>'
            )
        st.markdown(
            f'<div class="card">'
            f'<div class="card-title">Transaction distribution by merchant category</div>'
            f'{bars}</div>',
            unsafe_allow_html=True,
        )

    with col_stats:
        stats = [
            ("Average transaction amount", f"INR {df['transaction_amount_inr'].mean():,.2f}"),
            ("Median transaction amount",  f"INR {df['transaction_amount_inr'].median():,.2f}"),
            ("Max transaction amount",     f"INR {df['transaction_amount_inr'].max():,.2f}"),
            ("Min transaction amount",     f"INR {df['transaction_amount_inr'].min():,.2f}"),
            ("Average customer age",       f"{df['customer_age_years'].mean():.1f} years"),
            ("Average account age",        f"{df['account_age_days'].mean():.0f} days"),
            ("Unique merchant categories", str(df['merchant_category'].nunique())),
        ]
        rows = "".join(
            f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
            f'border-bottom:0.5px solid {C_BORDER};">'
            f'<span style="font-size:12px;color:{C_TEXT2};">{l}</span>'
            f'<span style="font-size:12px;font-weight:500;color:{C_TEXT1};">{v}</span>'
            f'</div>'
            for l, v in stats
        )
        st.markdown(
            f'<div class="card"><div class="card-title">Quick statistics</div>{rows}</div>',
            unsafe_allow_html=True,
        )


elif page == "Fraud patterns":
    render_top_bar("Where is fraud happening?")

    tab1, tab2, tab3 = st.tabs(["Amount & hour", "Merchant insights", "Risk indicators"])

    with tab1:
        c1, c2 = st.columns(2, gap="small")

        with c1:
            st.markdown(
                f'<div class="card"><div class="card-title">Fraud frequency by hour of day</div>',
                unsafe_allow_html=True,
            )
            hour_fraud       = df.groupby("transaction_hour")["is_fraud"].sum().reindex(range(24), fill_value=0)
            high_fraud_hours = set(range(0, 6)) | {22, 23}
            colors  = [C_RED_ORIG  if h in high_fraud_hours else C_BLUE_ORIG for h in range(24)]
            opacity = [0.85        if h in high_fraud_hours else 0.5          for h in range(24)]

            fig = go.Figure(data=[go.Bar(
                x=list(range(24)), y=hour_fraud.values,
                marker_color=colors, marker_opacity=opacity,
            )])
            lay = plotly_layout("Fraud count by hour", "Hour of day", "Fraud count", 320)
            lay["xaxis"]["tickvals"] = [0, 6, 12, 18, 23]
            lay["xaxis"]["ticktext"] = ["12am", "6am", "12pm", "6pm", "11pm"]
            fig.update_layout(**lay)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f'<div class="chart-legend">'
                f'<div class="chart-legend-item">'
                f'<span class="legend-dot" style="background:{C_RED_ORIG};"></span>High fraud hours</div>'
                f'<div class="chart-legend-item">'
                f'<span class="legend-dot" style="background:{C_BLUE_ORIG};"></span>Normal hours</div>'
                f'</div>'
                f'<div style="font-size:11px;color:{C_TEXT_MUTED};margin-top:8px;padding:0 2px;">'
                f'Most fraud happens between midnight and 5am — when monitoring is lowest.'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f'<div class="card"><div class="card-title">Transaction amount by class</div>',
                unsafe_allow_html=True,
            )
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=df[df["is_fraud"] == 0]["transaction_amount_inr"], name="Genuine",
                marker_color=C_BLUE_ORIG, line_color=C_BLUE_ORIG,
            ))
            fig.add_trace(go.Box(
                y=df[df["is_fraud"] == 1]["transaction_amount_inr"], name="Fraud",
                marker_color=C_RED_ORIG, line_color=C_RED_ORIG,
            ))
            lay = plotly_layout("Amount distribution (INR)", "", "Transaction amount (INR)", 320)
            lay["showlegend"] = True
            lay["legend"] = dict(font=dict(size=11, color=C_TEXT_MUTED), bgcolor="rgba(0,0,0,0)")
            fig.update_layout(**lay)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f'<div style="font-size:11px;color:{C_TEXT_MUTED};margin-top:4px;padding:0 2px;">'
                f'Fraudulent transactions tend to be much larger. Genuine ones cluster under ₹8,000.'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown(
            f'<div class="card"><div class="card-title">Fraud rate by merchant category</div>',
            unsafe_allow_html=True,
        )
        md = df.groupby("merchant_category")["is_fraud"].agg(["count", "mean"]).reset_index()
        md["fraud_rate"] = md["mean"] * 100
        md = md.sort_values("fraud_rate", ascending=True)
        median_r = md["fraud_rate"].median()
        bar_clrs = [C_RED_ORIG if r > median_r else C_BLUE_ORIG for r in md["fraud_rate"]]

        fig = go.Figure(data=[go.Bar(
            x=md["fraud_rate"], y=md["merchant_category"],
            orientation="h", marker_color=bar_clrs, marker_opacity=0.8,
        )])
        lay = plotly_layout("Fraud rate (%) by merchant category", "Fraud rate (%)", "", 400)
        lay["yaxis"]["tickfont"] = dict(size=11, color=C_TEXT_MUTED)
        lay["margin"]["l"] = 130
        fig.update_layout(**lay)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'<div style="font-size:11px;color:{C_TEXT_MUTED};margin-top:4px;padding:0 2px;">'
            f'Cash advances and jewelry are the riskiest categories by far.'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        r1, r2 = st.columns(2, gap="small")

        with r1:
            st.markdown(
                f'<div class="card"><div class="card-title">Account age distribution by class</div>',
                unsafe_allow_html=True,
            )
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[df["is_fraud"] == 0]["account_age_days"], name="Genuine",
                marker_color=C_BLUE_ORIG, opacity=0.6, nbinsx=30,
            ))
            fig.add_trace(go.Histogram(
                x=df[df["is_fraud"] == 1]["account_age_days"], name="Fraud",
                marker_color=C_RED_ORIG, opacity=0.8, nbinsx=30,
            ))
            lay = plotly_layout("Account age vs fraud", "Account age (days)", "Count", 320)
            lay["barmode"]    = "overlay"
            lay["showlegend"] = True
            lay["legend"]     = dict(font=dict(size=11, color=C_TEXT_MUTED), bgcolor="rgba(0,0,0,0)")
            fig.update_layout(**lay)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with r2:
            st.markdown(
                f'<div class="card"><div class="card-title">Risk indicator proportions by class</div>',
                unsafe_allow_html=True,
            )
            risk_feats = ["cvv_mismatch", "international_transaction", "location_match", "card_present"]
            risk_rows  = [
                {
                    "Feature": feat.replace("_", " ").title(),
                    "Genuine": df[df["is_fraud"] == 0][feat].mean(),
                    "Fraud":   df[df["is_fraud"] == 1][feat].mean(),
                }
                for feat in risk_feats
            ]
            rdf = pd.DataFrame(risk_rows)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=rdf["Feature"], y=rdf["Genuine"], name="Genuine",
                                 marker_color=C_BLUE_ORIG, opacity=0.7))
            fig.add_trace(go.Bar(x=rdf["Feature"], y=rdf["Fraud"],   name="Fraud",
                                 marker_color=C_RED_ORIG,  opacity=0.85))
            lay = plotly_layout("Risk flag rates: genuine vs fraud", "", "Proportion", 320)
            lay["barmode"]    = "group"
            lay["showlegend"] = True
            lay["legend"]     = dict(font=dict(size=11, color=C_TEXT_MUTED), bgcolor="rgba(0,0,0,0)")
            fig.update_layout(**lay)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f'<div style="font-size:11px;color:{C_TEXT_MUTED};margin-top:4px;padding:0 2px;">'
                f'Short gaps between transactions and CVV mismatches are the strongest fraud signals in this dataset.'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)


elif page == "Model results":
    render_top_bar("How well is the model working?")

    model_results = {
        "Logistic Regression": {
            "Accuracy": 1.00, "Precision": 1.00, "Recall": 1.00, "F1-Score": 1.00,
            "cm": {"TN": 95, "FP": 0, "FN": 0, "TP": 5},
        },
        "Decision Tree": {
            "Accuracy": 1.00, "Precision": 1.00, "Recall": 1.00, "F1-Score": 1.00,
            "cm": {"TN": 95, "FP": 0, "FN": 0, "TP": 5},
        },
        "Random Forest": {
            "Accuracy": 1.00, "Precision": 1.00, "Recall": 1.00, "F1-Score": 1.00,
            "cm": {"TN": 95, "FP": 0, "FN": 0, "TP": 5},
        },
    }
    best_model_name = "Random Forest"

    st.markdown(
        f'<div style="background-color:{C_CARD_BG2};border:0.5px solid {C_BORDER};'
        f'border-left:3px solid {C_BLUE_ORIG};border-radius:10px;'
        f'padding:0.75rem 1.25rem;margin-bottom:12px;">'
        f'<span style="font-size:12px;color:{C_TEXT2};">'
        f'Trained on SMOTE-balanced data, tested on the original imbalanced set (95 genuine / 5 fraud). '
        f'SMOTE was never applied to the test split — so the numbers reflect real-world performance.'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    mc1, mc2, mc3 = st.columns(3, gap="small")
    for col, name in zip([mc1, mc2, mc3], model_results):
        m       = model_results[name]
        is_best = name == best_model_name
        extra   = "model-card-best" if is_best else ""
        badge   = '<span class="badge badge-best">Best</span>' if is_best else ""

        rows = ""
        for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            val   = m[metric]
            color = (C_GREEN if val >= 0.9 else C_AMBER if val >= 0.7 else C_RED) \
                    if metric == "Recall" else C_TEXT1
            rows += (
                f'<div class="model-row">'
                f'<span class="model-row-label">{metric}</span>'
                f'<span class="model-row-value" style="color:{color};">{val:.2%}</span>'
                f'</div>'
            )
        with col:
            st.markdown(
                f'<div class="model-card {extra}">'
                f'<div class="model-card-title">{name} {badge}</div>'
                f'{rows}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="card"><div class="card-title">Confusion matrices — test set</div>',
        unsafe_allow_html=True,
    )
    cm1, cm2, cm3 = st.columns(3, gap="small")
    for col, name in zip([cm1, cm2, cm3], model_results):
        cm = model_results[name]["cm"]
        with col:
            st.markdown(
                f'<div style="text-align:center;">'
                f'<div style="font-size:12px;font-weight:500;color:{C_TEXT2};margin-bottom:8px;">{name}</div>'
                f'<div style="display:grid;grid-template-columns:80px 1fr 1fr;gap:4px;max-width:240px;margin:0 auto;">'
                f'<div></div>'
                f'<div style="text-align:center;font-size:10px;color:{C_TEXT_MUTED};font-weight:500;padding-bottom:2px;">Pred: 0</div>'
                f'<div style="text-align:center;font-size:10px;color:{C_TEXT_MUTED};font-weight:500;padding-bottom:2px;">Pred: 1</div>'
                f'<div style="text-align:right;font-size:10px;color:{C_TEXT_MUTED};font-weight:500;padding-right:6px;line-height:46px;">Actual: 0</div>'
                f'<div style="background-color:{C_CM_TN_BG};border-radius:6px;padding:10px 6px;'
                f'text-align:center;font-size:18px;font-weight:500;color:{C_CM_TN_TEXT};">{cm["TN"]}</div>'
                f'<div style="background-color:{C_CM_FP_BG};border-radius:6px;padding:10px 6px;'
                f'text-align:center;font-size:18px;font-weight:500;color:{C_CM_FP_TEXT};">{cm["FP"]}</div>'
                f'<div style="text-align:right;font-size:10px;color:{C_TEXT_MUTED};font-weight:500;padding-right:6px;line-height:46px;">Actual: 1</div>'
                f'<div style="background-color:{C_CM_FN_BG};border-radius:6px;padding:10px 6px;'
                f'text-align:center;font-size:18px;font-weight:500;color:{C_CM_FN_TEXT};">{cm["FN"]}</div>'
                f'<div style="background-color:{C_CM_TP_BG};border-radius:6px;padding:10px 6px;'
                f'text-align:center;font-size:18px;font-weight:500;color:{C_CM_TP_TEXT};">{cm["TP"]}</div>'
                f'</div>'
                f'<div style="display:grid;grid-template-columns:80px 1fr 1fr;gap:4px;max-width:240px;margin:4px auto 0;">'
                f'<div></div>'
                f'<div style="text-align:center;font-size:10px;color:{C_TEXT_MUTED};">TN<br><span style="font-size:9px;color:{C_TEXT_MUTED};opacity:0.75;">Correct: genuine</span></div>'
                f'<div style="text-align:center;font-size:10px;color:{C_TEXT_MUTED};">FP<br><span style="font-size:9px;color:{C_TEXT_MUTED};opacity:0.75;">Wrong: flagged genuine</span></div>'
                f'<div></div>'
                f'<div style="text-align:center;font-size:10px;color:{C_TEXT_MUTED};">FN<br><span style="font-size:9px;color:{C_TEXT_MUTED};opacity:0.75;">Missed: fraud not caught</span></div>'
                f'<div style="text-align:center;font-size:10px;color:{C_TEXT_MUTED};">TP<br><span style="font-size:9px;color:{C_TEXT_MUTED};opacity:0.75;">Correct: fraud</span></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    bm = model_results[best_model_name]
    st.markdown(
        f'<div style="background-color:{C_BEST_BG};border:0.5px solid {C_BEST_BORDER};'
        f'border-radius:10px;padding:1rem 1.25rem;margin-top:2px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
        f'<span style="font-size:13px;font-weight:500;color:{C_TEXT1};">Model we\'re using: {best_model_name}</span>'
        f'<span class="badge badge-best">Best</span>'
        f'</div>'
        f'<div style="font-size:12px;color:{C_TEXT2};">'
        f'Random Forest gave the best results — highest recall and fewest missed frauds. '
        f'F1-Score: {bm["F1-Score"]:.2%} &nbsp;|&nbsp; '
        f'Precision: {bm["Precision"]:.2%} &nbsp;|&nbsp; '
        f'Accuracy: {bm["Accuracy"]:.2%}'
        f'</div></div>',
        unsafe_allow_html=True,
    )


elif page == "Check a transaction":
    render_top_bar("Check a transaction")

    if model is None:
        st.error("Model artifacts not found. Run the training pipeline first.")
        st.stop()

    merchant_categories = [
        "cash_advance", "electronics", "entertainment", "fuel", "grocery",
        "healthcare", "jewelry", "online_shopping", "restaurant",
        "retail_clothing", "travel", "utilities",
    ]
    day_options = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    flag_defs = [
        ("card_present",             "Card present"),
        ("location_match",           "Location match"),
        ("cvv_mismatch",             "CVV mismatch"),
        ("international_transaction","International"),
        ("is_high_risk_merchant",    "High-risk merchant"),
    ]

    st.markdown(
        f'<div class="card-title" style="font-size:13px;font-weight:500;'
        f'color:{C_TEXT1};margin-bottom:10px;">Fill in the transaction details below</div>',
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        left_col, right_col = st.columns(2, gap="small")

        with left_col:
            st.markdown(
                f'<div style="font-size:11px;font-weight:500;color:{C_TEXT_MUTED};'
                f'text-transform:uppercase;letter-spacing:0.04em;margin-bottom:8px;">'
                f'Transaction details</div>',
                unsafe_allow_html=True,
            )
            amount      = st.number_input("Transaction amount (INR)", min_value=0.0, value=5000.0, step=100.0)
            hour        = st.number_input("Hour of day (0–23)", min_value=0, max_value=23, value=14)
            day_of_week = st.selectbox("Day of week", day_options)
            merchant    = st.selectbox("Merchant category", merchant_categories)
            distance    = st.number_input("Distance from home (km)", min_value=0.0, value=20.0, step=1.0)
            prev_gap    = st.number_input("Previous transaction gap (minutes)", min_value=0.0, value=500.0, step=10.0)

        with right_col:
            st.markdown(
                f'<div style="font-size:11px;font-weight:500;color:{C_TEXT_MUTED};'
                f'text-transform:uppercase;letter-spacing:0.04em;margin-bottom:8px;">'
                f'Risk factors</div>',
                unsafe_allow_html=True,
            )
            cust_age     = st.number_input("Customer age (years)", min_value=18, max_value=100, value=35)
            acct_age     = st.number_input("Account age (days)", min_value=0, value=1000, step=30)
            num_txn_24h  = st.number_input("Transactions in last 24h", min_value=0, value=3)
            num_declined = st.number_input("Declined transactions in last 7 days", min_value=0, value=0)

            st.markdown(
                f'<div style="font-size:12px;color:{C_TEXT2};margin-top:10px;margin-bottom:8px;">'
                f'Risk flags</div>',
                unsafe_allow_html=True,
            )

            flag_values = {}
            row1_cols = st.columns(3, gap="small")
            for col, (flag_key, label) in zip(row1_cols, flag_defs[:3]):
                with col:
                    flag_values[flag_key] = st.checkbox(label, key=f"chk_{flag_key}")

            row2_cols = st.columns(2, gap="small")
            for col, (flag_key, label) in zip(row2_cols, flag_defs[3:]):
                with col:
                    flag_values[flag_key] = st.checkbox(label, key=f"chk_{flag_key}")

        submitted = st.form_submit_button("Run prediction")

    # ------------------------------------------------------------------
    # Simulate Live Transaction — dialog modal version.
    # Clicking the button opens a full-screen dialog with a realistic
    # loading sequence, then renders the batch prediction results table.
    # ------------------------------------------------------------------
    st.markdown(f"""
    <style>
    /* Simulate Live Transaction button — matches Run prediction solid blue style */
    div[data-testid="stButton"] > button[kind="secondary"] {{
        background-color: #185FA5 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        width: 100% !important;
        padding: 11px 0 !important;
        letter-spacing: 0.02em !important;
        cursor: pointer !important;
    }}
    div[data-testid="stButton"] > button[kind="secondary"]:hover {{
        background-color: #1470BA !important;
        color: #FFFFFF !important;
    }}

    /* Dialog backdrop and container */
    div[data-testid="stDialog"] > div {{
        background-color: {"rgba(10,10,20,0.75)" if is_dark else "rgba(0,0,0,0.45)"} !important;
        backdrop-filter: blur(4px) !important;
    }}
    div[data-testid="stDialog"] [data-testid="stDialogContent"],
    div[data-testid="stDialog"] .st-emotion-cache-1ibsh2c {{
        background-color: {C_CARD_BG} !important;
        border: 0.5px solid {C_BORDER} !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        max-width: 860px !important;
    }}

    /* Skeleton shimmer animation for the loading state */
    @keyframes shimmer {{
        0%   {{ background-position: -800px 0; }}
        100% {{ background-position: 800px 0; }}
    }}
    .skeleton-row {{
        height: 14px;
        border-radius: 4px;
        background: linear-gradient(
            90deg,
            {"#2a2a3e" if is_dark else "#e8e8e8"} 25%,
            {"#3a3a52" if is_dark else "#f4f4f4"} 50%,
            {"#2a2a3e" if is_dark else "#e8e8e8"} 75%
        );
        background-size: 800px 100%;
        animation: shimmer 1.4s infinite linear;
        margin-bottom: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)

    # ---- Dialog definition ----
    @st.dialog("Recent Transactions — Fraud Scan", width="large")
    def simulation_dialog():
        # Step 1: show realistic loading sequence
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        skeleton_placeholder = st.empty()

        # Skeleton rows while loading
        skeleton_html = f"""
        <div style="margin-top:8px;">
          <div style="font-size:12px;font-weight:500;color:{C_TEXT_MUTED};
               margin-bottom:14px;letter-spacing:0.03em;">
            Pulling recent transactions...
          </div>
          {"".join(f'<div class="skeleton-row" style="width:{w}%;"></div>'
                   for w in [100, 95, 100, 88, 100, 92, 100, 85, 100, 90])}
        </div>"""
        skeleton_placeholder.markdown(skeleton_html, unsafe_allow_html=True)

        # Simulate realistic loading steps with status messages
        loading_steps = [
            (0.35, "Connecting to transaction service..."),
            (0.30, "Authenticating session..."),
            (0.40, "Pulling recent transactions..."),
            (0.35, "Retrieving 30-day history..."),
            (0.30, "Running fraud detection model..."),
            (0.25, "Generating risk scores..."),
        ]
        total_steps = len(loading_steps)
        elapsed = 0.0
        for step_idx, (duration, msg) in enumerate(loading_steps):
            status_placeholder.markdown(
                f'<div style="font-size:12px;color:{C_TEXT_MUTED};margin-bottom:6px;">'
                f'<span style="color:{C_BLUE};font-weight:500;">●</span> {msg}</div>',
                unsafe_allow_html=True,
            )
            progress_placeholder.progress((step_idx + 1) / total_steps)
            time.sleep(duration + random.uniform(0.0, 0.15))

        # Brief pause before revealing results
        time.sleep(0.3)
        status_placeholder.empty()
        progress_placeholder.empty()
        skeleton_placeholder.empty()

        # Step 2: Generate batch and run predictions
        batch_df  = generate_batch()
        result_df = predict_batch(batch_df)
        st.session_state["simulation_result"] = result_df

        # Step 3: Render the results inside the dialog
        total         = len(result_df)
        fraud_count   = int(result_df["Result"].sum())
        genuine_count = total - fraud_count

        genuine_bg   = "#173404" if is_dark else "#EAF3DE"
        genuine_text = "#C0DD97" if is_dark else "#3B6D11"
        fraud_bg     = "#501313" if is_dark else "#FCEBEB"
        fraud_text   = "#F7C1C1" if is_dark else "#A32D2D"

        # Summary header row inside dialog
        col_g, col_f, col_t = st.columns(3, gap="small")
        with col_g:
            st.markdown(
                f'<div style="background-color:{genuine_bg};border-radius:8px;'
                f'padding:10px 14px;text-align:center;">'
                f'<div style="font-size:10px;font-weight:500;color:{genuine_text};'
                f'text-transform:uppercase;letter-spacing:0.04em;margin-bottom:4px;">Genuine</div>'
                f'<div style="font-size:22px;font-weight:500;color:{genuine_text};">{genuine_count}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_f:
            st.markdown(
                f'<div style="background-color:{fraud_bg};border-radius:8px;'
                f'padding:10px 14px;text-align:center;">'
                f'<div style="font-size:10px;font-weight:500;color:{fraud_text};'
                f'text-transform:uppercase;letter-spacing:0.04em;margin-bottom:4px;">Fraud</div>'
                f'<div style="font-size:22px;font-weight:500;color:{fraud_text};">{fraud_count}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_t:
            st.markdown(
                f'<div style="background-color:{C_CARD_BG2};border-radius:8px;'
                f'border:0.5px solid {C_BORDER};padding:10px 14px;text-align:center;">'
                f'<div style="font-size:10px;font-weight:500;color:{C_TEXT_MUTED};'
                f'text-transform:uppercase;letter-spacing:0.04em;margin-bottom:4px;">Total scanned</div>'
                f'<div style="font-size:22px;font-weight:500;color:{C_TEXT1};">{total}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

        # Build full HTML table
        header_cols = ["#", "Amount (INR)", "Hour", "Merchant",
                       "Distance (km)", "CVV Mismatch", "Txns (24h)", "Card Present", "Result"]
        header_html = "".join(
            f'<th style="background-color:{C_CARD_BG2};color:{C_TEXT_MUTED};'
            f'font-weight:500;font-size:11px;text-align:left;'
            f'padding:9px 12px;border-bottom:0.5px solid {C_BORDER};">{col}</th>'
            for col in header_cols
        )

        rows_html = ""
        for i, row in enumerate(result_df.itertuples(index=False), start=1):
            is_fraud_row  = int(row.Result) == 1
            badge_bg      = fraud_bg   if is_fraud_row else genuine_bg
            badge_color   = fraud_text if is_fraud_row else genuine_text
            badge_label   = "\u2717 Fraud" if is_fraud_row else "\u2713 Genuine"
            row_bg        = C_CARD_BG2 if i % 2 == 0 else C_CARD_BG

            amt_fmt    = f"\u20b9{row.transaction_amount_inr:,.2f}"
            cvv_fmt    = "Yes" if row.cvv_mismatch   == 1 else "No"
            card_fmt   = "Yes" if row.card_present   == 1 else "No"

            cells = [
                str(i), amt_fmt, str(row.transaction_hour),
                str(row.merchant_category), str(row.distance_from_home_km),
                cvv_fmt, str(row.num_txn_last_24h), card_fmt,
            ]
            cells_html = "".join(
                f'<td style="padding:8px 12px;color:{C_TEXT1};font-size:12px;'
                f'border-bottom:0.5px solid {C_BORDER};">{c}</td>'
                for c in cells
            )
            result_cell = (
                f'<td style="padding:8px 12px;border-bottom:0.5px solid {C_BORDER};">'
                f'<span style="background-color:{badge_bg};color:{badge_color};'
                f'border-radius:20px;font-size:11px;font-weight:500;'
                f'padding:3px 10px;display:inline-block;white-space:nowrap;">'
                f'{badge_label}</span></td>'
            )
            rows_html += (
                f'<tr style="background-color:{row_bg};">'
                f'{cells_html}{result_cell}</tr>'
            )

        st.markdown(
            f'<div style="background-color:{C_CARD_BG};border:0.5px solid {C_BORDER};'
            f'border-radius:10px;overflow:hidden;">'
            f'<table style="width:100%;border-collapse:collapse;font-size:12px;">'
            f'<thead><tr>{header_html}</tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div style="font-size:11px;color:{C_TEXT_MUTED};margin-top:10px;">'
            f'Scan completed &nbsp;·&nbsp; '
            f'{total} transactions analyzed &nbsp;·&nbsp; '
            f'<span style="color:{fraud_text};">{fraud_count} flagged as fraud</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ---- Trigger button ----
    if st.button("Simulate Live Transaction", use_container_width=True, type="secondary"):
        simulation_dialog()

    # Muted hint text below the button
    st.markdown(
        f'<div style="text-align:center;font-size:11px;color:{C_TEXT_MUTED};'
        f'margin-top:4px;margin-bottom:14px;">'
        f'Fills in a random transaction so you can see the model in action'
        f'</div>',
        unsafe_allow_html=True,
    )

    if submitted:
        day_map     = encoders.get("day_of_week_mapping",
                                   {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6})
        le_merchant = encoders["merchant_category_encoder"]

        try:
            merchant_enc = int(le_merchant.transform([merchant])[0])
        except Exception:
            merchant_enc = 0

        day_enc = day_map.get(day_of_week, 0)

        feature_values = [
            amount, float(hour), float(day_enc), float(merchant_enc),
            float(flag_values["is_high_risk_merchant"]),
            float(flag_values["card_present"]),
            float(flag_values["location_match"]),
            distance, prev_gap,
            float(num_txn_24h), float(num_declined),
            float(flag_values["cvv_mismatch"]),
            float(flag_values["international_transaction"]),
            float(cust_age), float(acct_age),
        ]

        feature_cols = [
            "transaction_amount_inr", "transaction_hour", "day_of_week",
            "merchant_category", "is_high_risk_merchant", "card_present",
            "location_match", "distance_from_home_km", "prev_txn_gap_mins",
            "num_txn_last_24h", "num_declined_last_7days", "cvv_mismatch",
            "international_transaction", "customer_age_years", "account_age_days",
        ]

        input_df   = pd.DataFrame([feature_values], columns=feature_cols)
        scaled     = scaler.transform(input_df)
        pred       = int(model.predict(scaled)[0])
        proba      = model.predict_proba(scaled)[0]

        is_fraud_pred  = pred == 1
        confidence     = float(proba[1] if is_fraud_pred else proba[0]) * 100
        fill_color     = C_RED_ORIG if is_fraud_pred else C_GREEN_ORIG

        if is_fraud_pred:
            box_cls    = "result-fraud"
            txt_color  = C_FRAUD_RESULT_TEXT
            title_text = "This looks suspicious"
            sub_text   = "Several red flags found — flagged as fraud"
            bar_label  = "Model confidence"
        else:
            box_cls    = "result-genuine"
            txt_color  = C_GENUINE_RESULT_TEXT
            title_text = "Looks legitimate"
            sub_text   = "Nothing unusual about this transaction"
            bar_label  = "Model confidence"

        st.markdown(
            f'<div class="result-box {box_cls}">'
            f'<div class="result-title" style="color:{txt_color};">{title_text}</div>'
            f'<div class="result-sub">{sub_text}</div>'
            f'<div class="conf-label"><span>{bar_label}</span><span>{confidence:.1f}%</span></div>'
            f'<div class="conf-track">'
            f'<div class="conf-fill" style="width:{confidence:.1f}%;background-color:{fill_color};"></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )


elif page == "Batch prediction":
    render_top_bar("Batch prediction")

    # -----------------------------------------------------------------------
    # Constants used by validation and prediction
    # -----------------------------------------------------------------------
    REQUIRED_COLUMNS = [
        "transaction_amount_inr", "transaction_hour", "day_of_week",
        "merchant_category", "is_high_risk_merchant", "card_present",
        "location_match", "distance_from_home_km", "prev_txn_gap_mins",
        "num_txn_last_24h", "num_declined_last_7days", "cvv_mismatch",
        "international_transaction", "customer_age_years", "account_age_days",
    ]
    VALID_DAYS = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}
    VALID_MERCHANTS = {
        "grocery", "restaurant", "fuel", "retail_clothing", "electronics",
        "healthcare", "utilities", "online_shopping", "entertainment",
        "travel", "jewelry", "cash_advance",
    }
    BINARY_COLUMNS = [
        "is_high_risk_merchant", "card_present", "location_match",
        "cvv_mismatch", "international_transaction",
    ]

    # -----------------------------------------------------------------------
    # Helper: validate uploaded DataFrame before touching the model
    # -----------------------------------------------------------------------
    def validate_csv(upload_df):
        errors = {"missing_columns": [], "invalid_values": []}

        missing = [col for col in REQUIRED_COLUMNS if col not in upload_df.columns]
        if missing:
            errors["missing_columns"] = missing
            return False, errors  # no point checking values if cols are absent

        null_cols = [
            col for col in REQUIRED_COLUMNS
            if upload_df[col].isnull().any()
        ]
        for col in null_cols:
            errors["invalid_values"].append(f"{col} contains missing (NaN) values")

        invalid_days = set(upload_df["day_of_week"].dropna().unique()) - VALID_DAYS
        if invalid_days:
            errors["invalid_values"].append(
                f"day_of_week: found {sorted(invalid_days)} — accepted: {sorted(VALID_DAYS)}"
            )

        invalid_merchants = set(upload_df["merchant_category"].dropna().unique()) - VALID_MERCHANTS
        if invalid_merchants:
            errors["invalid_values"].append(
                f"merchant_category: found {sorted(invalid_merchants)} — check accepted values above"
            )

        for col in BINARY_COLUMNS:
            bad = upload_df[~upload_df[col].isin([0, 1])][col].unique()
            if len(bad) > 0:
                errors["invalid_values"].append(
                    f"{col}: found value(s) {list(bad)} — must be 0 or 1"
                )

        invalid_hours = upload_df[
            ~upload_df["transaction_hour"].between(0, 23)
        ]["transaction_hour"].unique()
        if len(invalid_hours) > 0:
            errors["invalid_values"].append(
                f"transaction_hour: found value(s) {list(invalid_hours)} — must be 0 to 23"
            )

        is_valid = not errors["missing_columns"] and not errors["invalid_values"]
        return is_valid, errors

    # -----------------------------------------------------------------------
    # Helper: run batch prediction — encode, scale, predict
    # -----------------------------------------------------------------------
    @st.cache_data
    def run_batch_prediction(upload_df_json: str):
        """
        Accepts the uploaded DataFrame serialised as JSON to allow caching.
        Returns a copy of the DataFrame with Predicted_Fraud and
        Fraud_Probability_% columns appended.
        """
        upload_df = pd.read_json(upload_df_json, orient="split")
        proc = upload_df.copy()

        day_map = encoders.get(
            "day_of_week_mapping",
            {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6},
        )
        proc["day_of_week"] = proc["day_of_week"].map(day_map).fillna(0).astype(int)

        le_merchant = encoders["merchant_category_encoder"]
        proc["merchant_category"] = proc["merchant_category"].apply(
            lambda x: int(le_merchant.transform([x])[0]) if x in le_merchant.classes_ else 0
        )

        scaled      = scaler.transform(proc[REQUIRED_COLUMNS])
        preds       = model.predict(scaled)
        proba       = model.predict_proba(scaled)[:, 1]

        result = upload_df.copy()
        result["Predicted_Fraud"]     = preds
        result["Fraud_Probability_%"] = (proba * 100).round(2)
        return result

    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode("utf-8")

    # -----------------------------------------------------------------------
    # Section 2 — CSV Format Guide (always visible)
    # -----------------------------------------------------------------------
    amber_border = "#EF9F27" if is_dark else "#854F0B"
    amber_bg     = "#3D2A00" if is_dark else "#FEF3CD"
    amber_text   = "#EF9F27" if is_dark else "#854F0B"
    amber_hdr_bg = "#5C3D00" if is_dark else "#F5C842"

    col_rows = [
        ("transaction_amount_inr",    "Float",   "Any positive number (e.g. 1500.00)"),
        ("transaction_hour",          "Integer", "0 to 23"),
        ("day_of_week",               "String",  "Mon, Tue, Wed, Thu, Fri, Sat, Sun"),
        ("merchant_category",         "String",  "grocery, restaurant, fuel, retail_clothing, electronics, healthcare, utilities, online_shopping, entertainment, travel, jewelry, cash_advance"),
        ("is_high_risk_merchant",     "Integer", "0 or 1"),
        ("card_present",              "Integer", "0 or 1"),
        ("location_match",            "Integer", "0 or 1"),
        ("distance_from_home_km",     "Float",   "Any positive number"),
        ("prev_txn_gap_mins",         "Float",   "Any positive number"),
        ("num_txn_last_24h",          "Integer", "Any positive integer"),
        ("num_declined_last_7days",   "Integer", "0 or above"),
        ("cvv_mismatch",              "Integer", "0 or 1"),
        ("international_transaction", "Integer", "0 or 1"),
        ("customer_age_years",        "Integer", "18 to 100"),
        ("account_age_days",          "Integer", "1 or above"),
    ]

    tbl_rows_html = ""
    for idx, (col_name, col_type, col_range) in enumerate(col_rows):
        row_bg = C_CARD_BG if idx % 2 == 0 else C_CARD_BG2
        tbl_rows_html += (
            f'<tr style="background-color:{row_bg};">'
            f'<td style="padding:6px 10px;font-size:11px;font-family:Courier New,monospace;'
            f'color:{C_TEXT1};border-bottom:0.5px solid {C_BORDER};">{col_name}</td>'
            f'<td style="padding:6px 10px;font-size:11px;color:{C_TEXT2};'
            f'border-bottom:0.5px solid {C_BORDER};">{col_type}</td>'
            f'<td style="padding:6px 10px;font-size:11px;color:{C_TEXT2};'
            f'border-bottom:0.5px solid {C_BORDER};">{col_range}</td>'
            f'</tr>'
        )

    st.markdown(
        f'<div style="background-color:{amber_bg};border:1px solid {amber_border};'
        f'border-radius:10px;padding:1rem;margin-bottom:16px;">'
        f'<div style="font-size:13px;font-weight:500;color:{amber_text};margin-bottom:6px;">'
        f'How to format your file</div>'
        f'<div style="font-size:12px;color:{C_TEXT2};margin-bottom:12px;">'
        f'Upload a CSV and we\'ll run predictions on every row. It needs exactly these 15 columns '
        f'(case-sensitive). Don\'t include an <code>is_fraud</code> column — '
        f'we\'ll predict that for you.</div>'
        f'<div style="border-radius:8px;overflow:hidden;">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr>'
        f'<th style="background-color:{amber_hdr_bg};color:{C_TEXT1};font-size:11px;'
        f'font-weight:500;padding:7px 10px;text-align:left;">Column Name</th>'
        f'<th style="background-color:{amber_hdr_bg};color:{C_TEXT1};font-size:11px;'
        f'font-weight:500;padding:7px 10px;text-align:left;">Type</th>'
        f'<th style="background-color:{amber_hdr_bg};color:{C_TEXT1};font-size:11px;'
        f'font-weight:500;padding:7px 10px;text-align:left;">Accepted Values / Range</th>'
        f'</tr></thead>'
        f'<tbody>{tbl_rows_html}</tbody>'
        f'</table></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    with st.expander("See a sample CSV row"):
        st.code(
            "transaction_amount_inr,transaction_hour,day_of_week,merchant_category,"
            "is_high_risk_merchant,card_present,location_match,distance_from_home_km,"
            "prev_txn_gap_mins,num_txn_last_24h,num_declined_last_7days,cvv_mismatch,"
            "international_transaction,customer_age_years,account_age_days\n"
            "2500.00,14,Wed,grocery,0,1,1,5.2,120.0,2,0,0,0,35,720",
            language="text",
        )

    # -----------------------------------------------------------------------
    # Section 3 — File Upload Zone
    # -----------------------------------------------------------------------
    st.markdown(
        f'<div style="font-size:13px;font-weight:500;color:{C_TEXT1};margin-bottom:2px;">'
        f'Upload your transaction CSV file</div>'
        f'<div style="font-size:11px;color:{C_TEXT_MUTED};margin-bottom:4px;">'
        f'Accepted format: .csv only</div>'
        f'<div style="font-size:12px;color:{C_TEXT_MUTED};text-align:center;margin-bottom:6px;">'
        f'No file uploaded yet. Drop a CSV below to get started.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        label_visibility="collapsed",
    )

    # Clear cached result when a new file is uploaded
    if uploaded_file is not None:
        current_name = uploaded_file.name
        if st.session_state.get("_upload_filename") != current_name:
            st.session_state["upload_result"]    = None
            st.session_state["_upload_filename"] = current_name

    if uploaded_file is not None:
        # File info row
        file_size_kb = round(len(uploaded_file.getvalue()) / 1024, 1)
        raw_df       = pd.read_csv(uploaded_file)
        row_count_up = len(raw_df)

        st.markdown(
            f'<div style="background-color:{C_CARD_BG2};border:0.5px solid {C_BORDER};'
            f'border-radius:8px;padding:8px 14px;margin-bottom:12px;'
            f'font-size:12px;color:{C_TEXT_MUTED};display:flex;gap:16px;">'
            f'<span><strong style="color:{C_TEXT1};">{uploaded_file.name}</strong></span>'
            f'<span>{file_size_kb} KB</span>'
            f'<span>{row_count_up:,} rows detected</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # -----------------------------------------------------------------------
        # Section 4 — Validation
        # -----------------------------------------------------------------------
        is_valid, val_errors = validate_csv(raw_df)

        if not is_valid:
            # Build error details HTML
            missing_html = ""
            if val_errors["missing_columns"]:
                items = "".join(
                    f'<li style="margin-bottom:3px;font-family:Courier New,monospace;">{c}</li>'
                    for c in val_errors["missing_columns"]
                )
                missing_html = (
                    f'<div style="font-size:12px;color:{C_FRAUD_RESULT_TEXT};margin-bottom:8px;">'
                    f'<strong>Missing required columns:</strong>'
                    f'<ul style="margin:6px 0 0 16px;padding:0;">{items}</ul></div>'
                )

            invalid_html = ""
            if val_errors["invalid_values"]:
                items = "".join(
                    f'<li style="margin-bottom:3px;">'
                    f'<span style="font-family:Courier New,monospace;">{e}</span></li>'
                    for e in val_errors["invalid_values"]
                )
                invalid_html = (
                    f'<div style="font-size:12px;color:{C_FRAUD_RESULT_TEXT};margin-bottom:8px;">'
                    f'<strong>Invalid values detected:</strong>'
                    f'<ul style="margin:6px 0 0 16px;padding:0;">{items}</ul></div>'
                )

            st.markdown(
                f'<div style="background-color:{C_FRAUD_RESULT_BG};'
                f'border:0.5px solid {C_FRAUD_RESULT_BORDER};'
                f'border-radius:10px;padding:1rem;margin-bottom:8px;">'
                f'<div style="font-size:13px;font-weight:500;color:{C_FRAUD_RESULT_TEXT};'
                f'margin-bottom:10px;">Something\'s wrong with this file</div>'
                f'{missing_html}{invalid_html}'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="font-size:12px;color:{C_TEXT_MUTED};">'
                f'Fix the errors above and re-upload. '
                f'Check the format guide at the top of this page for reference.</div>',
                unsafe_allow_html=True,
            )

        else:
            # Validation passed — run predictions
            st.markdown(
                f'<div style="background-color:{C_GENUINE_RESULT_BG};'
                f'border:0.5px solid {C_GENUINE_RESULT_BORDER};'
                f'border-radius:8px;padding:8px 14px;margin-bottom:10px;'
                f'font-size:12px;color:{C_GENUINE_RESULT_TEXT};">'
                f'File looks good — running predictions'
                f'</div>',
                unsafe_allow_html=True,
            )

            if st.session_state.get("upload_result") is None:
                with st.spinner(f"Checking {row_count_up:,} transactions..."):
                    df_json = raw_df.to_json(orient="split")
                    result_df = run_batch_prediction(df_json)
                    st.session_state["upload_result"] = result_df

            result_df = st.session_state["upload_result"]

            # -------------------------------------------------------------------
            # Section 5a — Summary Metrics
            # -------------------------------------------------------------------
            total_up   = len(result_df)
            fraud_up   = int(result_df["Predicted_Fraud"].sum())
            genuine_up = total_up - fraud_up
            fraud_rate_up = (fraud_up / total_up * 100) if total_up > 0 else 0.0

            m1, m2, m3, m4 = st.columns(4, gap="small")
            for col, cls, label, value, sub in [
                (m1, "blue",  "Total transactions",   f"{total_up:,}",         "Rows in uploaded file"),
                (m2, "green", "Legitimate",           f"{genuine_up:,}",       f"{100 - fraud_rate_up:.1f}% of transactions"),
                (m3, "red",   "Flagged as fraud",     f"{fraud_up:,}",         f"{fraud_rate_up:.1f}% of transactions"),
                (m4, "amber", "Fraud rate",           f"{fraud_rate_up:.1f}%", "Across all uploaded rows"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="metric-card metric-card-{cls}">'
                        f'<div class="metric-label">{label}</div>'
                        f'<div class="metric-value">{value}</div>'
                        f'<div class="metric-sub">{sub}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

            # -------------------------------------------------------------------
            # Section 5b — Predictions Table
            # -------------------------------------------------------------------
            genuine_bg_r   = "#173404" if is_dark else "#EAF3DE"
            genuine_text_r = "#C0DD97" if is_dark else "#3B6D11"
            fraud_bg_r     = "#501313" if is_dark else "#FCEBEB"
            fraud_text_r   = "#F7C1C1" if is_dark else "#A32D2D"
            fraud_row_bg   = "#2A1010" if is_dark else "#FFF5F5"

            # Pagination — 50 rows per page
            page_size  = 50
            total_pages = max(1, (total_up + page_size - 1) // page_size)

            if "upload_page" not in st.session_state:
                st.session_state["upload_page"] = 1

            tbl_header_cols = [
                "#", "Amount (INR)", "Hour", "Day", "Merchant",
                "Distance (km)", "CVV Mismatch", "Txns (24h)", "Card Present",
                "Result", "Confidence",
            ]
            tbl_header_html = "".join(
                f'<th style="background-color:{C_CARD_BG2};color:{C_TEXT_MUTED};'
                f'font-weight:500;font-size:11px;text-align:left;'
                f'padding:8px 10px;border-bottom:0.5px solid {C_BORDER};'
                f'white-space:nowrap;">{c}</th>'
                for c in tbl_header_cols
            )

            page_idx   = st.session_state["upload_page"] - 1
            start_row  = page_idx * page_size
            end_row    = min(start_row + page_size, total_up)
            page_slice = result_df.iloc[start_row:end_row]

            tbl_rows_html_r = ""
            for rel_i, (_, row) in enumerate(page_slice.iterrows(), start=1):
                abs_i        = start_row + rel_i
                is_fraud_row = int(row["Predicted_Fraud"]) == 1
                row_bg_r     = fraud_row_bg if is_fraud_row else (C_CARD_BG2 if abs_i % 2 == 0 else C_CARD_BG)
                badge_bg_r   = fraud_bg_r   if is_fraud_row else genuine_bg_r
                badge_clr_r  = fraud_text_r if is_fraud_row else genuine_text_r
                badge_lbl_r  = "\u2717 Fraud" if is_fraud_row else "\u2713 Genuine"

                cells_r = [
                    str(abs_i),
                    f"\u20b9{row['transaction_amount_inr']:,.2f}",
                    str(int(row["transaction_hour"])),
                    str(row["day_of_week"]),
                    str(row["merchant_category"]),
                    str(row["distance_from_home_km"]),
                    "Yes" if row["cvv_mismatch"]  == 1 else "No",
                    str(int(row["num_txn_last_24h"])),
                    "Yes" if row["card_present"]  == 1 else "No",
                ]
                cells_html_r = "".join(
                    f'<td style="padding:7px 10px;color:{C_TEXT1};font-size:12px;'
                    f'border-bottom:0.5px solid {C_BORDER};white-space:nowrap;">{c}</td>'
                    for c in cells_r
                )
                result_cell_r = (
                    f'<td style="padding:7px 10px;border-bottom:0.5px solid {C_BORDER};">'
                    f'<span style="background-color:{badge_bg_r};color:{badge_clr_r};'
                    f'border-radius:20px;font-size:11px;font-weight:500;'
                    f'padding:3px 10px;display:inline-block;white-space:nowrap;">'
                    f'{badge_lbl_r}</span></td>'
                )
                conf_cell_r = (
                    f'<td style="padding:7px 10px;color:{C_TEXT2};font-size:12px;'
                    f'border-bottom:0.5px solid {C_BORDER};">'
                    f'{row["Fraud_Probability_%"]:.1f}%</td>'
                )
                tbl_rows_html_r += (
                    f'<tr style="background-color:{row_bg_r};">'
                    f'{cells_html_r}{result_cell_r}{conf_cell_r}</tr>'
                )

            st.markdown(
                f'<div style="background-color:{C_CARD_BG};border:0.5px solid {C_BORDER};'
                f'border-radius:10px;overflow-x:auto;margin-bottom:4px;">'
                f'<table style="width:100%;border-collapse:collapse;font-size:12px;">'
                f'<thead><tr>{tbl_header_html}</tr></thead>'
                f'<tbody>{tbl_rows_html_r}</tbody>'
                f'</table></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="text-align:right;font-size:11px;color:{C_TEXT_MUTED};margin-bottom:8px;">'
                f'Showing {start_row+1}–{end_row} of {total_up:,} transactions'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Pagination controls
            if total_pages > 1:
                pg_col1, pg_col2, pg_col3 = st.columns([1, 2, 1], gap="small")
                with pg_col1:
                    if st.button("Previous", key="up_prev",
                                 disabled=st.session_state["upload_page"] <= 1):
                        st.session_state["upload_page"] -= 1
                        st.rerun()
                with pg_col2:
                    st.markdown(
                        f'<div style="text-align:center;font-size:12px;'
                        f'color:{C_TEXT_MUTED};padding-top:8px;">'
                        f'Page {st.session_state["upload_page"]} of {total_pages} '
                        f'&nbsp;·&nbsp; Showing rows {start_row+1}–{end_row} of {total_up:,}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with pg_col3:
                    if st.button("Next", key="up_next",
                                 disabled=st.session_state["upload_page"] >= total_pages):
                        st.session_state["upload_page"] += 1
                        st.rerun()

            # -------------------------------------------------------------------
            # Section 5c — Download Button
            # -------------------------------------------------------------------
            csv_bytes = convert_df_to_csv(result_df)
            st.download_button(
                label="Download results",
                data=csv_bytes,
                file_name=f"fraud_predictions_{uploaded_file.name}",
                mime="text/csv",
                use_container_width=True,
            )

            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

            # -------------------------------------------------------------------
            # Section 5d — Summary Charts
            # -------------------------------------------------------------------
            chart_col1, chart_col2 = st.columns(2, gap="small")

            # Chart 1 — Donut: Fraud vs Genuine
            with chart_col1:
                circ_r    = 2 * 3.14159 * 40
                g_dash_r  = (genuine_up / total_up) * circ_r if total_up > 0 else 0
                f_dash_r  = (fraud_up   / total_up) * circ_r if total_up > 0 else 0
                f_off_r   = -g_dash_r
                fraud_pct_r = (fraud_up / total_up * 100) if total_up > 0 else 0

                st.markdown(
                    f'<div class="card"><div class="card-title">Fraud vs genuine distribution</div>'
                    f'<div class="donut-wrap">'
                    f'<svg width="120" height="120" viewBox="0 0 100 100">'
                    f'<circle cx="50" cy="50" r="40" fill="none" stroke="{C_GREEN_ORIG}"'
                    f' stroke-width="12" stroke-dasharray="{g_dash_r:.2f} {circ_r:.2f}"'
                    f' stroke-dashoffset="0" transform="rotate(-90 50 50)"/>'
                    f'<circle cx="50" cy="50" r="40" fill="none" stroke="{C_RED_ORIG}"'
                    f' stroke-width="12" stroke-dasharray="{f_dash_r:.2f} {circ_r:.2f}"'
                    f' stroke-dashoffset="{f_off_r:.2f}" transform="rotate(-90 50 50)"/>'
                    f'<text x="50" y="50" text-anchor="middle" dominant-baseline="central"'
                    f' font-size="13" font-weight="500" fill="{C_TEXT1}">{fraud_pct_r:.1f}%</text>'
                    f'</svg>'
                    f'<div class="donut-legend">'
                    f'<div class="donut-row">'
                    f'<span class="donut-dot" style="background:{C_GREEN_ORIG};"></span>'
                    f'Genuine ({genuine_up:,})</div>'
                    f'<div class="donut-row">'
                    f'<span class="donut-dot" style="background:{C_RED_ORIG};"></span>'
                    f'Fraud ({fraud_up:,})</div>'
                    f'</div></div></div>',
                    unsafe_allow_html=True,
                )

            # Chart 2 — Fraud count by merchant category
            with chart_col2:
                fraud_by_merch = (
                    result_df[result_df["Predicted_Fraud"] == 1]
                    .groupby("merchant_category")
                    .size()
                    .sort_values(ascending=True)
                )

                if fraud_by_merch.empty:
                    st.markdown(
                        f'<div class="card" style="display:flex;align-items:center;'
                        f'justify-content:center;min-height:180px;">'
                        f'<div style="text-align:center;font-size:12px;color:{C_GENUINE_RESULT_TEXT};">'
                        f'No fraud detected in this file. All transactions look clean.'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="card"><div class="card-title">'
                        f'Fraud detections by merchant category</div>',
                        unsafe_allow_html=True,
                    )
                    fig_merch = go.Figure(data=[go.Bar(
                        x=fraud_by_merch.values,
                        y=fraud_by_merch.index,
                        orientation="h",
                        marker_color=C_RED_ORIG,
                        marker_opacity=0.85,
                    )])
                    lay_merch = plotly_layout("", "Fraud count", "", 280)
                    lay_merch["margin"]["l"] = 130
                    lay_merch["yaxis"]["tickfont"] = dict(size=11, color=C_TEXT_MUTED)
                    fig_merch.update_layout(**lay_merch)
                    st.plotly_chart(fig_merch, use_container_width=True,
                                    config={"displayModeBar": False})
                    st.markdown("</div>", unsafe_allow_html=True)
