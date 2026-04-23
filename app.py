# app.py
# MealSnap AI - Streamlit app (V2 logic + OLD color theme restored)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf

# ============================================================
# 1) PAGE CONFIG
# ============================================================
st.set_page_config(page_title="MealSnap AI", page_icon="🍽️", layout="wide")

# ============================================================
# 2) OPTIONAL: PREDICT MODULE (predict.py)
# ============================================================
try:
    from predict import predict_topk, should_abstain
    PREDICT_MODULE_OK = True
except Exception:
    PREDICT_MODULE_OK = False

# ============================================================
# 3) PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def p(*parts: str) -> str:
    return os.path.join(BASE_DIR, *parts)

MODEL_PATH = p("models", "food101_mobilenetv2.keras")
CLASS_NAMES_PATH = p("models", "class_names.txt")
NUTRITION_PATH = p("nutrition.csv")
FEEDBACK_LOG_PATH = p("feedback_log.csv")

# ============================================================
# 4) APP CONFIG
# ============================================================
CUSTOM_LABELS = [
    "avocado_toast",
    "avocado_toast_with_omelette",
    "grilled_chicken",
    "roast_chicken",
    "chicken_biryani",
    "mutton_biryani",
    "chicken_fried_rice",
]
MEAL_TYPES = ["Breakfast", "Lunch", "Dinner", "Snacks"]

HISTORY_COLUMNS = [
    "date",
    "meal_type",
    "final_label",
    "portion_g",
    "calories",
    "protein",
    "carbs",
    "fat",
    "fiber",
    "notes",
]

# ============================================================
# 5) CSS (OLD COLOR COMBO FROM YOUR ORIGINAL app.py)
#    - I kept your original CSS theme
#    - Added only ONE small extra section for "right panel size"
# ============================================================
# ONLY CHANGE: add this CSS block at the END of your <style> (just before </style>)
# Nothing else changes.

st.markdown(
    """
<style>
/* =========================================================
   A) APP BACKGROUND
   ========================================================= */
.stApp {
  background: linear-gradient(135deg, #DFF7EA 0%, #FFFFFF 55%, #F2FBF6 100%);
  padding-bottom: 320px !important;
}
main .block-container{ padding-bottom: 320px !important; }

/* =========================================================
   B) SIDEBAR: BLACK
   ========================================================= */
section[data-testid="stSidebar"]{
  background: #0b0f14 !important;
  border-right: 1px solid rgba(255,255,255,0.08);
  overflow: visible !important;
}
section[data-testid="stSidebar"] *{ color: #e5e7eb !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{ color: #ffffff !important; }

/* =========================================================
   C) SIDEBAR CONTROLS: GRAY INPUTS
   ========================================================= */
:root{
  --sb-box-bg: #151b23;
  --sb-box-border: rgba(255,255,255,0.14);
  --sb-box-text: #e5e7eb;
  --sb-muted: rgba(229,231,235,0.70);
  --sb-accent: #22c55e;
}
section[data-testid="stSidebar"] label{ color: #e5e7eb !important; }
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] p{ color: var(--sb-muted) !important; }

section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stTextArea textarea,
section[data-testid="stSidebar"] .stNumberInput input{
  background: var(--sb-box-bg) !important;
  color: var(--sb-box-text) !important;
  border: 1px solid var(--sb-box-border) !important;
  border-radius: 12px !important;
}
section[data-testid="stSidebar"] .stTextArea textarea::placeholder,
section[data-testid="stSidebar"] .stTextInput input::placeholder{
  color: rgba(229,231,235,0.55) !important;
}

/* =========================================================
   D) FILE UPLOADER VISIBLE
   ========================================================= */
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] > div{
  background: var(--sb-box-bg) !important;
  border: 1px solid var(--sb-box-border) !important;
  border-radius: 14px !important;
  padding: 14px !important;
}
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] *{ color: #e5e7eb !important; }
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] div[role="button"]{
  background: rgba(255,255,255,0.03) !important;
  border: 1px dashed rgba(255,255,255,0.22) !important;
  border-radius: 14px !important;
}
section[data-testid="stSidebar"] div[data-testid="stFileUploader"] button{
  background: rgba(255,255,255,0.08) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  color: #ffffff !important;
  border-radius: 12px !important;
}

/* =========================================================
   E) SELECTBOX CLOSED CONTROL
   ========================================================= */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div{
  background: var(--sb-box-bg) !important;
  border: 1px solid var(--sb-box-border) !important;
  border-radius: 12px !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] span,
section[data-testid="stSidebar"] div[data-baseweb="select"] div{
  color: var(--sb-box-text) !important;
}

/* =========================================================
   F) DROPDOWN MENU: ON TOP + DARK GRAY
   ========================================================= */
div[data-baseweb="popover"],
div[data-baseweb="select"],
div[role="listbox"],
ul[role="listbox"]{
  z-index: 999999 !important;
  background: #111827 !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  border-radius: 12px !important;
  box-shadow: 0 18px 45px rgba(0,0,0,0.45) !important;
}
div[data-baseweb="popover"] li,
div[role="option"]{ color: #e5e7eb !important; }
div[data-baseweb="popover"] li:hover,
div[role="option"]:hover{ background: rgba(34,197,94,0.14) !important; }
section[data-testid="stSidebar"] div[data-testid="stSelectbox"]{
  position: relative !important;
  z-index: 99999 !important;
}

/* =========================================================
   H) METRICS READABLE
   ========================================================= */
div[data-testid="stMetric"]{
  background: rgba(255,255,255,0.55);
  border: 1px solid rgba(255,255,255,0.35);
  box-shadow: 0 10px 30px rgba(2, 44, 25, 0.10);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border-radius: 15px;
  padding: 14px 14px;
}
div[data-testid="stMetric"] *{ color: #1e3a8a !important; }

/* =========================================================
   I) TABS
   ========================================================= */
div[data-testid="stTabs"]{
  background: rgba(255,255,255,0.40);
  border: 1px solid rgba(255,255,255,0.30);
  box-shadow: 0 10px 30px rgba(2, 44, 25, 0.08);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border-radius: 15px;
  padding: 10px 12px;
}
button[role="tab"]{
  border-radius: 12px !important;
  padding: 8px 12px !important;
}
button[role="tab"] *{ color: #0f172a !important; }

/* =========================================================
   J) HERO
   ========================================================= */
.hero {
  border-radius: 18px;
  padding: 18px 18px;
  margin: 6px 0 18px 0;
  background: linear-gradient(135deg, #0EA66B 0%, #0B7A4D 55%, #07563A 100%);
  color: white;
  box-shadow: 0 16px 40px rgba(2, 44, 25, 0.22);
  border: 1px solid rgba(255,255,255,0.18);
}
.hero-wrap{ display: flex; align-items: center; gap: 16px; }
.hero-icon{
  font-size: 54px;
  line-height: 1;
  background: rgba(255,255,255,0.18);
  border: 1px solid rgba(255,255,255,0.22);
  border-radius: 16px;
  padding: 12px 14px;
  box-shadow: 0 10px 24px rgba(0,0,0,0.14);
}
.hero-title{ font-size: 34px; font-weight: 800; margin: 0; }
.hero-subtitle{ font-size: 16px; opacity: 0.92; margin-top: 6px; }
.hero-kicker{ font-size: 13px; opacity: 0.85; margin: 0; }

/* =========================================================
   K) CODE CHIP FIX (FINAL LABEL VISIBILITY)
   ========================================================= */
code{
  background: transparent !important;
  color: #000000 !important;
  border: none !important;
  padding: 0 !important;
  border-radius: 0 !important;
}

/* =========================================================
   L) EXTRA (V2): MAKE RIGHT SIDE CONTENT A BIT BIGGER
   ========================================================= */
.right-panel { font-size: 18px; }

/* =========================================================
   M) FIX: MAIN CONTENT TEXT (THIS IS THE ONLY NEW PART)
   - makes your st.write / st.markdown / st.caption text BLACK
   ========================================================= */
div[data-testid="stTabs"] div[role="tabpanel"] p,
div[data-testid="stTabs"] div[role="tabpanel"] span,
div[data-testid="stTabs"] div[role="tabpanel"] label{
  color: #0f172a !important;
}
/* =========================================================
   N) BUTTON FIX: make buttons green + text white
   ========================================================= */
div.stButton > button{
  background: #0EA66B !important;
  color: #ffffff !important;
  border: 1px solid rgba(0,0,0,0.08) !important;
  border-radius: 12px !important;
  font-weight: 800 !important;
  padding: 10px 16px !important;
}

div.stButton > button:hover{
  filter: brightness(0.95) !important;
}

/* optional: remove ugly focus outline */
div.stButton > button:focus{
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(14,166,107,0.25) !important;
}
/* =========================================================
   O) DOWNLOAD BUTTON: Match the green Save button style
   ========================================================= */
div.stDownloadButton > button{
  background: #0EA66B !important;
  color: #ffffff !important;
  border: 1px solid rgba(0,0,0,0.08) !important;
  border-radius: 12px !important;
  font-weight: 800 !important;
  padding: 10px 16px !important;
}

div.stDownloadButton > button:hover{
  filter: brightness(0.95) !important;
}

div.stDownloadButton > button:focus{
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(14,166,107,0.25) !important;
}


</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# 6) HERO HEADER
# ============================================================
st.markdown(
    """
<div class="hero">
  <div class="hero-wrap">
    <div class="hero-icon">🍽️</div>
    <div>
      <p class="hero-kicker">AI-powered food recognition</p>
      <h1 class="hero-title">MealSnap AI</h1>
      <div class="hero-subtitle">Snap your plate → Instant calories &amp; nutrition</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True
)

# ============================================================
# 7) LOADERS
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

@st.cache_data
def load_nutrition():
    # V2: includes fiber_g
    cols = ["label", "calories_per_100g", "protein_g", "carbs_g", "fat_g", "fiber_g"]
    if not os.path.exists(NUTRITION_PATH) or os.path.getsize(NUTRITION_PATH) == 0:
        return pd.DataFrame(columns=cols)

    try:
        df = pd.read_csv(NUTRITION_PATH)
    except Exception:
        return pd.DataFrame(columns=cols)

    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df[cols].copy()
    df["label"] = df["label"].astype(str).str.strip()
    return df

# ============================================================
# 8) HISTORY HELPERS
# ============================================================
def ensure_feedback_log_exists():
    if not os.path.exists(FEEDBACK_LOG_PATH):
        pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(FEEDBACK_LOG_PATH, index=False)
def is_likely_non_food(top1_conf: float, top3: list) -> tuple:
    """Simple check to detect non-food images."""
    # Very low confidence = probably not food
    if top1_conf < 0.20:
        return True, f"Very low confidence ({top1_conf:.1%}) - might not be food"
    
    # All top-3 similar & low = model confused = not food
    confidences = [conf for _, conf in top3]
    if max(confidences) < 0.25 and (max(confidences) - min(confidences)) < 0.05:
        return True, "Model very uncertain - all predictions equally low"
    
    return False, "OK"
def safe_read_history():
    ensure_feedback_log_exists()
    try:
        df = pd.read_csv(FEEDBACK_LOG_PATH)
        if df.empty:
            return pd.DataFrame(columns=HISTORY_COLUMNS)
        for col in HISTORY_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[HISTORY_COLUMNS]
    except Exception:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

def save_history_row(row: dict):
    ensure_feedback_log_exists()
    pd.DataFrame([row], columns=HISTORY_COLUMNS).to_csv(
        FEEDBACK_LOG_PATH, mode="a", header=False, index=False
    )

# ============================================================
# 9) NUTRITION LOOKUP (PER-100G SCALED BY PORTION)
# ============================================================
def nutrition_lookup(df, label, portion_g: int):
    if df.empty:
        return None
    row = df[df["label"] == label]
    if row.empty:
        return None

    row = row.iloc[0]
    scale = float(portion_g) / 100.0

    def sf(x):
        try:
            return float(x)
        except Exception:
            return None

    cal = sf(row["calories_per_100g"])
    pro = sf(row["protein_g"])
    carb = sf(row["carbs_g"])
    fat = sf(row["fat_g"])
    fib = sf(row["fiber_g"])

    return {
        "calories": None if cal is None else cal * scale,
        "protein": None if pro is None else pro * scale,
        "carbs": None if carb is None else carb * scale,
        "fat": None if fat is None else fat * scale,
        "fiber": None if fib is None else fib * scale,
    }

# ============================================================
# 9.5) NON-FOOD DETECTION (OOD)
# ============================================================
def is_likely_non_food_v2(top1_conf: float, top3: list) -> tuple:
    """
    CALIBRATED non-food detection - only catches obvious non-food images.
    
    Args:
        top1_conf: Top prediction confidence (0-1)
        top3: List of (label, confidence) tuples
    
    Returns:
        (is_non_food, reason)
    """
    confidences = [conf for _, conf in top3]
    top3_confs = confidences[:3]
    
    # RULE 1: Extremely low confidence (< 15%)
    if top1_conf < 0.15:
        return True, f"Extremely low confidence ({top1_conf:.1%})"
    
    # RULE 2: Low (15-20%) + flat distribution
    if top1_conf < 0.20:
        conf_spread = max(top3_confs) - min(top3_confs)
        if conf_spread < 0.05:
            return True, f"Low confidence ({top1_conf:.1%}) with flat distribution"
    
    # RULE 3: All top-3 below 12%
    if all(conf < 0.12 for conf in top3_confs):
        return True, "All predictions extremely low (< 12%)"
    
    return False, "Looks like food"

# ============================================================
# 10) SAVE HISTORY
# ============================================================

# ============================================================
# 10) INIT + FILE CHECKS
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = safe_read_history()

missing = []
if not os.path.exists(MODEL_PATH):
    missing.append("Missing model file in models/")
if not os.path.exists(CLASS_NAMES_PATH):
    missing.append("Missing class_names.txt in models/")
if missing:
    st.error("Required files are missing:\n\n- " + "\n- ".join(missing))
    st.stop()

model = load_model()
class_names = load_class_names()
nutrition_df = load_nutrition()

# ============================================================
# 11) SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("Controls")

uploaded = st.sidebar.file_uploader(
    "Upload food image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photo of a single dish for best results."
)

# V2 request: typing input, default 0, max 1000
portion_g = st.sidebar.number_input(
    "🍽️ Estimated portion (grams)",
    min_value=0,
    max_value=1000,
    value=0,
    step=10,
    help="Type grams directly (0–1000)."
)

meal_type = st.sidebar.selectbox(
    "Meal type",
    MEAL_TYPES,
    index=0,
    help="Pick Breakfast/Lunch/Dinner/Snacks for your history log."
)

# Final label + notes
all_labels = ["(use top prediction)"] + CUSTOM_LABELS + class_names
final_label_choice = st.sidebar.selectbox(
    "Final label (for nutrition)",
    all_labels,
    index=0,
    help="Select the closest match for accurate nutrition + saving history."
)

notes = st.sidebar.text_area(
    "Notes (optional)",
    placeholder="Example: homemade, less oil, extra chicken",
    help="Optional notes saved into your meal history."
)

# ============================================================
# 12) MAIN LAYOUT
# ============================================================
col_img, col_out = st.columns([1.0, 1.4], gap="large")

with col_img:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Image Preview")

    if uploaded is None:
        st.info("📷 Upload an image from the sidebar to preview it here.")
        st.image(np.zeros((240, 360, 3), dtype=np.uint8), width=360)
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    img = Image.open(uploaded)
    # V2 request: smaller image
    st.image(img, width=360)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 13) PREDICTION
# ============================================================
if PREDICT_MODULE_OK:
    top1_label, top1_conf, top3, _ = predict_topk(model, img, class_names, k=3)
else:
    # fallback
    x = np.expand_dims(np.array(img.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0, axis=0)
    preds = model.predict(x, verbose=0)[0]
    idxs = preds.argsort()[-3:][::-1]
    top3 = [(class_names[int(i)], float(preds[int(i)])) for i in idxs]
    top1_label, top1_conf = top3[0]

final_label = top1_label if final_label_choice == "(use top prediction)" else final_label_choice

# ---- Decide abstain (unknown) ----
abstain = False
if PREDICT_MODULE_OK:
    abstain = should_abstain(top3, threshold=0.30, margin=0.02)   # start here; tune later
    st.sidebar.caption(
    f"Debug: top1={top1_conf:.2f}, top2={top3[1][1]:.2f}, margin={(top1_conf - top3[1][1]):.2f}, abstain={abstain}"
)


# ---- What to DISPLAY as prediction ----
display_top1_label = "unknown" if abstain else top1_label

# ---- What to USE for nutrition ----
# If user explicitly chose a label in the sidebar, respect it always.
# If user left "(use top prediction)" but we abstain, do NOT use the model guess.
if final_label_choice == "(use top prediction)":
    final_label = "unknown" if abstain else top1_label
else:
    final_label = final_label_choice

# ---- Warning for user ----
if abstain and final_label_choice == "(use top prediction)":
    st.sidebar.warning("Not sure this is a known food. Please select the correct Final label for nutrition.")

# Check if image is actually food
is_non_food, non_food_reason = is_likely_non_food(top1_conf, top3)
# ---- Warning for user ----
if abstain and final_label_choice == "(use top prediction)":
    st.sidebar.warning("Not sure this is a known food. Please select the correct Final label for nutrition.")

# ============================================================
# 🚨 CHECK IF IMAGE IS ACTUALLY FOOD (OOD DETECTION)
# ============================================================
is_non_food, non_food_reason = is_likely_non_food_v2(top1_conf, top3)
# Only calculate if portion > 0
macros = nutrition_lookup(nutrition_df, final_label, int(portion_g)) if int(portion_g) > 0 else None

# ============================================================
# 14) RESULTS (TABS)
# ============================================================
with col_out:
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)

    tab_pred, tab_nut, tab_hist = st.tabs(["Predictions", "Nutrition", "History"])

    # -----------------------------
    # A) Predictions tab
    # -----------------------------
    with tab_pred:
        # ============================================================
        # 🚨 CHECK IF NON-FOOD FIRST (BEFORE SHOWING ANYTHING)
        # ============================================================
        if is_non_food:
            st.error(f"""
            🚫 **This doesn't look like food!**
            
            **Reason:** {non_food_reason}
            
            **Common causes:**
            - Not a food item (person, animal, vehicle, object)
            - Image is too blurry or dark
            - Food is not clearly visible
            
            **Please upload a clear photo of a food dish.**
            """)
            st.stop()  # Stop execution - don't show predictions or nutrition
        
        # ===== ONLY RUNS IF IT'S FOOD =====
        pct_val = float(top1_conf) * 100.0
        st.metric("🥚 Top Prediction", f"{display_top1_label} ({pct_val:.2f}%)", delta=f"{pct_val:.2f}%")

        top3_df = pd.DataFrame(
            {"label": [x[0] for x in top3], "confidence": [x[1] for x in top3]}
        ).set_index("label")
        st.write("Top-3 confidence")
        st.bar_chart(top3_df)

    # -----------------------------
    # B) Nutrition tab
    # -----------------------------
    with tab_nut:
        # ---- Styled text (separate size + color for each) ----
        st.markdown(
    """
    <div style="font-size:22px; font-weight:800; color:#111827; margin-bottom:6px;">
      Final label used for nutrition:
    </div>
    """,
    unsafe_allow_html=True
)

        st.markdown(
    f"""
    <div style="font-size:26px; font-weight:900; color:#0EA66B; margin-bottom:6px;">
      {final_label}
    </div>
    """,
    unsafe_allow_html=True
)

        st.markdown(
    """
    <div style="font-size:14px; color:#475569;">
    If the model guessed wrong, pick a better label in the sidebar.
    </div>
    """,
    unsafe_allow_html=True
)


        if int(portion_g) == 0:
            st.info("Enter portion grams (0–1000) to calculate nutrition.")
        else:
            if macros is None:
                st.warning(f"No nutrition row found for `{final_label}` in nutrition.csv")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🔥 Calories", f"{macros['calories']:.0f} kcal" if macros["calories"] is not None else "—")
                c2.metric("💪 Protein", f"{macros['protein']:.1f} g" if macros["protein"] is not None else "—")
                c3.metric("🍞 Carbs", f"{macros['carbs']:.1f} g" if macros["carbs"] is not None else "—")
                c4.metric("🥑 Fat", f"{macros['fat']:.1f} g" if macros["fat"] is not None else "—")

                # Fiber (kept, shown below)
                c5, _ = st.columns([1, 3])
                c5.metric("🌾 Fiber", f"{macros['fiber']:.1f} g" if macros["fiber"] is not None else "—")

                st.caption(f"Portion used: {portion_g}g")

        st.divider()

        if st.button("Save to Meal History"):
            if int(portion_g) == 0:
                st.warning("Portion is 0g. Enter a valid portion before saving.")
            elif macros is None:
                st.warning("Nutrition not found for this label. Pick a different Final label before saving.")
            else:
                today = time.strftime("%Y-%m-%d")
                row = {
                    "date": today,
                    "meal_type": meal_type,
                    "final_label": final_label,
                    "portion_g": int(portion_g),
                    "calories": None if macros["calories"] is None else round(macros["calories"], 0),
                    "protein": None if macros["protein"] is None else round(macros["protein"], 1),
                    "carbs": None if macros["carbs"] is None else round(macros["carbs"], 1),
                    "fat": None if macros["fat"] is None else round(macros["fat"], 1),
                    "fiber": None if macros["fiber"] is None else round(macros["fiber"], 1),
                    "notes": notes.strip(),
                }
                save_history_row(row)
                st.session_state.history = pd.concat([pd.DataFrame([row]), st.session_state.history], ignore_index=True)
                st.success("Saved! Go to History tab.")
                st.rerun()

    # -----------------------------
    # C) History tab
    # -----------------------------
    with tab_hist:
        hist = st.session_state.history.copy()
        if hist.empty:
            st.info("No history yet. Save a meal to see it here.")
        else:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.dataframe(hist, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            csv_bytes = hist.to_csv(index=False).encode("utf-8")
            st.download_button("Export history (CSV)", data=csv_bytes, file_name="meal_history.csv", mime="text/csv")

    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Note: Calories/macros depend on nutrition.csv values; portion is user-provided.")





































####################################### BELOW IS THE WORKING CODE  #################################################


# # app.py
# # MealSnap AI - Streamlit app (V2 logic + OLD color theme restored)

# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# import time
# import numpy as np
# import pandas as pd
# import streamlit as st
# from PIL import Image
# import tensorflow as tf

# # ============================================================
# # 1) PAGE CONFIG
# # ============================================================
# st.set_page_config(page_title="MealSnap AI", page_icon="🍽️", layout="wide")

# # ============================================================
# # 2) OPTIONAL: PREDICT MODULE (predict.py)
# # ============================================================
# try:
#     from predict import predict_topk, should_abstain
#     PREDICT_MODULE_OK = True
# except Exception:
#     PREDICT_MODULE_OK = False

# # ============================================================
# # 3) PATHS
# # ============================================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# def p(*parts: str) -> str:
#     return os.path.join(BASE_DIR, *parts)

# MODEL_PATH = p("models", "food101_mobilenetv2.keras")
# CLASS_NAMES_PATH = p("models", "class_names.txt")
# NUTRITION_PATH = p("nutrition.csv")
# FEEDBACK_LOG_PATH = p("feedback_log.csv")

# # ============================================================
# # 4) APP CONFIG
# # ============================================================
# CUSTOM_LABELS = [
#     "avocado_toast",
#     "avocado_toast_with_omelette",
#     "grilled_chicken",
#     "roast_chicken",
#     "chicken_biryani",
#     "mutton_biryani",
#     "chicken_fried_rice",
# ]
# MEAL_TYPES = ["Breakfast", "Lunch", "Dinner", "Snacks"]

# HISTORY_COLUMNS = [
#     "date",
#     "meal_type",
#     "final_label",
#     "portion_g",
#     "calories",
#     "protein",
#     "carbs",
#     "fat",
#     "fiber",
#     "notes",
# ]

# # ============================================================
# # 5) CSS (OLD COLOR COMBO FROM YOUR ORIGINAL app.py)
# #    - I kept your original CSS theme
# #    - Added only ONE small extra section for "right panel size"
# # ============================================================
# # ONLY CHANGE: add this CSS block at the END of your <style> (just before </style>)
# # Nothing else changes.

# st.markdown(
#     """
# <style>
# /* =========================================================
#    A) APP BACKGROUND
#    ========================================================= */
# .stApp {
#   background: linear-gradient(135deg, #DFF7EA 0%, #FFFFFF 55%, #F2FBF6 100%);
#   padding-bottom: 320px !important;
# }
# main .block-container{ padding-bottom: 320px !important; }

# /* =========================================================
#    B) SIDEBAR: BLACK
#    ========================================================= */
# section[data-testid="stSidebar"]{
#   background: #0b0f14 !important;
#   border-right: 1px solid rgba(255,255,255,0.08);
#   overflow: visible !important;
# }
# section[data-testid="stSidebar"] *{ color: #e5e7eb !important; }
# section[data-testid="stSidebar"] h1,
# section[data-testid="stSidebar"] h2,
# section[data-testid="stSidebar"] h3{ color: #ffffff !important; }

# /* =========================================================
#    C) SIDEBAR CONTROLS: GRAY INPUTS
#    ========================================================= */
# :root{
#   --sb-box-bg: #151b23;
#   --sb-box-border: rgba(255,255,255,0.14);
#   --sb-box-text: #e5e7eb;
#   --sb-muted: rgba(229,231,235,0.70);
#   --sb-accent: #22c55e;
# }
# section[data-testid="stSidebar"] label{ color: #e5e7eb !important; }
# section[data-testid="stSidebar"] small,
# section[data-testid="stSidebar"] p{ color: var(--sb-muted) !important; }

# section[data-testid="stSidebar"] .stTextInput input,
# section[data-testid="stSidebar"] .stTextArea textarea,
# section[data-testid="stSidebar"] .stNumberInput input{
#   background: var(--sb-box-bg) !important;
#   color: var(--sb-box-text) !important;
#   border: 1px solid var(--sb-box-border) !important;
#   border-radius: 12px !important;
# }
# section[data-testid="stSidebar"] .stTextArea textarea::placeholder,
# section[data-testid="stSidebar"] .stTextInput input::placeholder{
#   color: rgba(229,231,235,0.55) !important;
# }

# /* =========================================================
#    D) FILE UPLOADER VISIBLE
#    ========================================================= */
# section[data-testid="stSidebar"] div[data-testid="stFileUploader"] > div{
#   background: var(--sb-box-bg) !important;
#   border: 1px solid var(--sb-box-border) !important;
#   border-radius: 14px !important;
#   padding: 14px !important;
# }
# section[data-testid="stSidebar"] div[data-testid="stFileUploader"] *{ color: #e5e7eb !important; }
# section[data-testid="stSidebar"] div[data-testid="stFileUploader"] div[role="button"]{
#   background: rgba(255,255,255,0.03) !important;
#   border: 1px dashed rgba(255,255,255,0.22) !important;
#   border-radius: 14px !important;
# }
# section[data-testid="stSidebar"] div[data-testid="stFileUploader"] button{
#   background: rgba(255,255,255,0.08) !important;
#   border: 1px solid rgba(255,255,255,0.14) !important;
#   color: #ffffff !important;
#   border-radius: 12px !important;
# }

# /* =========================================================
#    E) SELECTBOX CLOSED CONTROL
#    ========================================================= */
# section[data-testid="stSidebar"] div[data-baseweb="select"] > div{
#   background: var(--sb-box-bg) !important;
#   border: 1px solid var(--sb-box-border) !important;
#   border-radius: 12px !important;
# }
# section[data-testid="stSidebar"] div[data-baseweb="select"] span,
# section[data-testid="stSidebar"] div[data-baseweb="select"] div{
#   color: var(--sb-box-text) !important;
# }

# /* =========================================================
#    F) DROPDOWN MENU: ON TOP + DARK GRAY
#    ========================================================= */
# div[data-baseweb="popover"],
# div[data-baseweb="select"],
# div[role="listbox"],
# ul[role="listbox"]{
#   z-index: 999999 !important;
#   background: #111827 !important;
#   border: 1px solid rgba(255,255,255,0.14) !important;
#   border-radius: 12px !important;
#   box-shadow: 0 18px 45px rgba(0,0,0,0.45) !important;
# }
# div[data-baseweb="popover"] li,
# div[role="option"]{ color: #e5e7eb !important; }
# div[data-baseweb="popover"] li:hover,
# div[role="option"]:hover{ background: rgba(34,197,94,0.14) !important; }
# section[data-testid="stSidebar"] div[data-testid="stSelectbox"]{
#   position: relative !important;
#   z-index: 99999 !important;
# }

# /* =========================================================
#    H) METRICS READABLE
#    ========================================================= */
# div[data-testid="stMetric"]{
#   background: rgba(255,255,255,0.55);
#   border: 1px solid rgba(255,255,255,0.35);
#   box-shadow: 0 10px 30px rgba(2, 44, 25, 0.10);
#   backdrop-filter: blur(10px);
#   -webkit-backdrop-filter: blur(10px);
#   border-radius: 15px;
#   padding: 14px 14px;
# }
# div[data-testid="stMetric"] *{ color: #1e3a8a !important; }

# /* =========================================================
#    I) TABS
#    ========================================================= */
# div[data-testid="stTabs"]{
#   background: rgba(255,255,255,0.40);
#   border: 1px solid rgba(255,255,255,0.30);
#   box-shadow: 0 10px 30px rgba(2, 44, 25, 0.08);
#   backdrop-filter: blur(10px);
#   -webkit-backdrop-filter: blur(10px);
#   border-radius: 15px;
#   padding: 10px 12px;
# }
# button[role="tab"]{
#   border-radius: 12px !important;
#   padding: 8px 12px !important;
# }
# button[role="tab"] *{ color: #0f172a !important; }

# /* =========================================================
#    J) HERO
#    ========================================================= */
# .hero {
#   border-radius: 18px;
#   padding: 18px 18px;
#   margin: 6px 0 18px 0;
#   background: linear-gradient(135deg, #0EA66B 0%, #0B7A4D 55%, #07563A 100%);
#   color: white;
#   box-shadow: 0 16px 40px rgba(2, 44, 25, 0.22);
#   border: 1px solid rgba(255,255,255,0.18);
# }
# .hero-wrap{ display: flex; align-items: center; gap: 16px; }
# .hero-icon{
#   font-size: 54px;
#   line-height: 1;
#   background: rgba(255,255,255,0.18);
#   border: 1px solid rgba(255,255,255,0.22);
#   border-radius: 16px;
#   padding: 12px 14px;
#   box-shadow: 0 10px 24px rgba(0,0,0,0.14);
# }
# .hero-title{ font-size: 34px; font-weight: 800; margin: 0; }
# .hero-subtitle{ font-size: 16px; opacity: 0.92; margin-top: 6px; }
# .hero-kicker{ font-size: 13px; opacity: 0.85; margin: 0; }

# /* =========================================================
#    K) CODE CHIP FIX (FINAL LABEL VISIBILITY)
#    ========================================================= */
# code{
#   background: transparent !important;
#   color: #000000 !important;
#   border: none !important;
#   padding: 0 !important;
#   border-radius: 0 !important;
# }

# /* =========================================================
#    L) EXTRA (V2): MAKE RIGHT SIDE CONTENT A BIT BIGGER
#    ========================================================= */
# .right-panel { font-size: 18px; }

# /* =========================================================
#    M) FIX: MAIN CONTENT TEXT (THIS IS THE ONLY NEW PART)
#    - makes your st.write / st.markdown / st.caption text BLACK
#    ========================================================= */
# div[data-testid="stTabs"] div[role="tabpanel"] p,
# div[data-testid="stTabs"] div[role="tabpanel"] span,
# div[data-testid="stTabs"] div[role="tabpanel"] label{
#   color: #0f172a !important;
# }
# /* =========================================================
#    N) BUTTON FIX: make buttons green + text white
#    ========================================================= */
# div.stButton > button{
#   background: #0EA66B !important;
#   color: #ffffff !important;
#   border: 1px solid rgba(0,0,0,0.08) !important;
#   border-radius: 12px !important;
#   font-weight: 800 !important;
#   padding: 10px 16px !important;
# }

# div.stButton > button:hover{
#   filter: brightness(0.95) !important;
# }

# /* optional: remove ugly focus outline */
# div.stButton > button:focus{
#   outline: none !important;
#   box-shadow: 0 0 0 3px rgba(14,166,107,0.25) !important;
# }


# </style>
# """,
#     unsafe_allow_html=True
# )


# # ============================================================
# # 6) HERO HEADER
# # ============================================================
# st.markdown(
#     """
# <div class="hero">
#   <div class="hero-wrap">
#     <div class="hero-icon">🍽️</div>
#     <div>
#       <p class="hero-kicker">AI-powered food recognition</p>
#       <h1 class="hero-title">MealSnap AI</h1>
#       <div class="hero-subtitle">Snap your plate → Instant calories &amp; nutrition</div>
#     </div>
#   </div>
# </div>
# """,
#     unsafe_allow_html=True
# )

# # ============================================================
# # 7) LOADERS
# # ============================================================
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model(MODEL_PATH)

# @st.cache_data
# def load_class_names():
#     with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
#         return [line.strip() for line in f if line.strip()]

# @st.cache_data
# def load_nutrition():
#     # V2: includes fiber_g
#     cols = ["label", "calories_per_100g", "protein_g", "carbs_g", "fat_g", "fiber_g"]
#     if not os.path.exists(NUTRITION_PATH) or os.path.getsize(NUTRITION_PATH) == 0:
#         return pd.DataFrame(columns=cols)

#     try:
#         df = pd.read_csv(NUTRITION_PATH)
#     except Exception:
#         return pd.DataFrame(columns=cols)

#     for c in cols:
#         if c not in df.columns:
#             df[c] = np.nan

#     df = df[cols].copy()
#     df["label"] = df["label"].astype(str).str.strip()
#     return df

# # ============================================================
# # 8) HISTORY HELPERS
# # ============================================================
# def ensure_feedback_log_exists():
#     if not os.path.exists(FEEDBACK_LOG_PATH):
#         pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(FEEDBACK_LOG_PATH, index=False)

# def safe_read_history():
#     ensure_feedback_log_exists()
#     try:
#         df = pd.read_csv(FEEDBACK_LOG_PATH)
#         if df.empty:
#             return pd.DataFrame(columns=HISTORY_COLUMNS)
#         for col in HISTORY_COLUMNS:
#             if col not in df.columns:
#                 df[col] = None
#         return df[HISTORY_COLUMNS]
#     except Exception:
#         return pd.DataFrame(columns=HISTORY_COLUMNS)

# def save_history_row(row: dict):
#     ensure_feedback_log_exists()
#     pd.DataFrame([row], columns=HISTORY_COLUMNS).to_csv(
#         FEEDBACK_LOG_PATH, mode="a", header=False, index=False
#     )

# # ============================================================
# # 9) NUTRITION LOOKUP (PER-100G SCALED BY PORTION)
# # ============================================================
# def nutrition_lookup(df, label, portion_g: int):
#     if df.empty:
#         return None
#     row = df[df["label"] == label]
#     if row.empty:
#         return None

#     row = row.iloc[0]
#     scale = float(portion_g) / 100.0

#     def sf(x):
#         try:
#             return float(x)
#         except Exception:
#             return None

#     cal = sf(row["calories_per_100g"])
#     pro = sf(row["protein_g"])
#     carb = sf(row["carbs_g"])
#     fat = sf(row["fat_g"])
#     fib = sf(row["fiber_g"])

#     return {
#         "calories": None if cal is None else cal * scale,
#         "protein": None if pro is None else pro * scale,
#         "carbs": None if carb is None else carb * scale,
#         "fat": None if fat is None else fat * scale,
#         "fiber": None if fib is None else fib * scale,
#     }

# # ============================================================
# # 10) INIT + FILE CHECKS
# # ============================================================
# if "history" not in st.session_state:
#     st.session_state.history = safe_read_history()

# missing = []
# if not os.path.exists(MODEL_PATH):
#     missing.append("Missing model file in models/")
# if not os.path.exists(CLASS_NAMES_PATH):
#     missing.append("Missing class_names.txt in models/")
# if missing:
#     st.error("Required files are missing:\n\n- " + "\n- ".join(missing))
#     st.stop()

# model = load_model()
# class_names = load_class_names()
# nutrition_df = load_nutrition()

# # ============================================================
# # 11) SIDEBAR CONTROLS
# # ============================================================
# st.sidebar.header("Controls")

# uploaded = st.sidebar.file_uploader(
#     "Upload food image",
#     type=["jpg", "jpeg", "png"],
#     help="Upload a clear photo of a single dish for best results."
# )

# # V2 request: typing input, default 0, max 1000
# portion_g = st.sidebar.number_input(
#     "🍽️ Estimated portion (grams)",
#     min_value=0,
#     max_value=1000,
#     value=0,
#     step=10,
#     help="Type grams directly (0–1000)."
# )

# meal_type = st.sidebar.selectbox(
#     "Meal type",
#     MEAL_TYPES,
#     index=0,
#     help="Pick Breakfast/Lunch/Dinner/Snacks for your history log."
# )

# # Final label + notes
# all_labels = ["(use top prediction)"] + CUSTOM_LABELS + class_names
# final_label_choice = st.sidebar.selectbox(
#     "Final label (for nutrition)",
#     all_labels,
#     index=0,
#     help="Select the closest match for accurate nutrition + saving history."
# )

# notes = st.sidebar.text_area(
#     "Notes (optional)",
#     placeholder="Example: homemade, less oil, extra chicken",
#     help="Optional notes saved into your meal history."
# )

# # ============================================================
# # 12) MAIN LAYOUT
# # ============================================================
# col_img, col_out = st.columns([1.0, 1.4], gap="large")

# with col_img:
#     st.markdown('<div class="glass-card">', unsafe_allow_html=True)
#     st.subheader("Image Preview")

#     if uploaded is None:
#         st.info("📷 Upload an image from the sidebar to preview it here.")
#         st.image(np.zeros((240, 360, 3), dtype=np.uint8), width=360)
#         st.markdown("</div>", unsafe_allow_html=True)
#         st.stop()

#     img = Image.open(uploaded)
#     # V2 request: smaller image
#     st.image(img, width=360)
#     st.markdown("</div>", unsafe_allow_html=True)

# # ============================================================
# # 13) PREDICTION
# # ============================================================
# if PREDICT_MODULE_OK:
#     top1_label, top1_conf, top3, _ = predict_topk(model, img, class_names, k=3)
# else:
#     # fallback
#     x = np.expand_dims(np.array(img.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0, axis=0)
#     preds = model.predict(x, verbose=0)[0]
#     idxs = preds.argsort()[-3:][::-1]
#     top3 = [(class_names[int(i)], float(preds[int(i)])) for i in idxs]
#     top1_label, top1_conf = top3[0]

# final_label = top1_label if final_label_choice == "(use top prediction)" else final_label_choice

# if PREDICT_MODULE_OK and should_abstain(float(top1_conf), threshold=0.50):
#     st.sidebar.warning("Low confidence prediction. Please select the correct Final label.")

# # Only calculate if portion > 0
# macros = nutrition_lookup(nutrition_df, final_label, int(portion_g)) if int(portion_g) > 0 else None

# # ============================================================
# # 14) RESULTS (TABS)
# # ============================================================
# with col_out:
#     st.markdown('<div class="right-panel">', unsafe_allow_html=True)

#     tab_pred, tab_nut, tab_hist = st.tabs(["Predictions", "Nutrition", "History"])

#     # -----------------------------
#     # A) Predictions tab
#     # -----------------------------
#     with tab_pred:
#         pct_val = float(top1_conf) * 100.0
#         st.metric("🥚 Top Prediction", f"{top1_label} ({pct_val:.2f}%)", delta=f"{pct_val:.2f}%")

#         top3_df = pd.DataFrame(
#             {"label": [x[0] for x in top3], "confidence": [x[1] for x in top3]}
#         ).set_index("label")
#         st.write("Top-3 confidence")
#         st.bar_chart(top3_df)

#     # -----------------------------
#     # B) Nutrition tab
#     # -----------------------------
#     with tab_nut:
#         # ---- Styled text (separate size + color for each) ----
#         st.markdown(
#     """
#     <div style="font-size:22px; font-weight:800; color:#111827; margin-bottom:6px;">
#       Final label used for nutrition:
#     </div>
#     """,
#     unsafe_allow_html=True
# )

#         st.markdown(
#     f"""
#     <div style="font-size:26px; font-weight:900; color:#0EA66B; margin-bottom:6px;">
#       {final_label}
#     </div>
#     """,
#     unsafe_allow_html=True
# )

#         st.markdown(
#     """
#     <div style="font-size:14px; color:#475569;">
#       If the model guessed wrong, pick a better label in the sidebar.
#     </div>
#     """,
#     unsafe_allow_html=True
# )


#         if int(portion_g) == 0:
#             st.info("Enter portion grams (0–1000) to calculate nutrition.")
#         else:
#             if macros is None:
#                 st.warning(f"No nutrition row found for `{final_label}` in nutrition.csv")
#             else:
#                 c1, c2, c3, c4 = st.columns(4)
#                 c1.metric("🔥 Calories", f"{macros['calories']:.0f} kcal" if macros["calories"] is not None else "—")
#                 c2.metric("💪 Protein", f"{macros['protein']:.1f} g" if macros["protein"] is not None else "—")
#                 c3.metric("🍞 Carbs", f"{macros['carbs']:.1f} g" if macros["carbs"] is not None else "—")
#                 c4.metric("🥑 Fat", f"{macros['fat']:.1f} g" if macros["fat"] is not None else "—")

#                 # Fiber (kept, shown below)
#                 c5, _ = st.columns([1, 3])
#                 c5.metric("🌾 Fiber", f"{macros['fiber']:.1f} g" if macros["fiber"] is not None else "—")

#                 st.caption(f"Portion used: {portion_g}g")

#         st.divider()

#         if st.button("Save to Meal History"):
#             if int(portion_g) == 0:
#                 st.warning("Portion is 0g. Enter a valid portion before saving.")
#             elif macros is None:
#                 st.warning("Nutrition not found for this label. Pick a different Final label before saving.")
#             else:
#                 today = time.strftime("%Y-%m-%d")
#                 row = {
#                     "date": today,
#                     "meal_type": meal_type,
#                     "final_label": final_label,
#                     "portion_g": int(portion_g),
#                     "calories": None if macros["calories"] is None else round(macros["calories"], 0),
#                     "protein": None if macros["protein"] is None else round(macros["protein"], 1),
#                     "carbs": None if macros["carbs"] is None else round(macros["carbs"], 1),
#                     "fat": None if macros["fat"] is None else round(macros["fat"], 1),
#                     "fiber": None if macros["fiber"] is None else round(macros["fiber"], 1),
#                     "notes": notes.strip(),
#                 }
#                 save_history_row(row)
#                 st.session_state.history = pd.concat([pd.DataFrame([row]), st.session_state.history], ignore_index=True)
#                 st.success("Saved! Go to History tab.")
#                 st.rerun()

#     # -----------------------------
#     # C) History tab
#     # -----------------------------
#     with tab_hist:
#         hist = st.session_state.history.copy()
#         if hist.empty:
#             st.info("No history yet. Save a meal to see it here.")
#         else:
#             st.markdown('<div class="glass-card">', unsafe_allow_html=True)
#             st.dataframe(hist, use_container_width=True)
#             st.markdown("</div>", unsafe_allow_html=True)

#             csv_bytes = hist.to_csv(index=False).encode("utf-8")
#             st.download_button("Export history (CSV)", data=csv_bytes, file_name="meal_history.csv", mime="text/csv")

#     st.markdown("</div>", unsafe_allow_html=True)

# st.caption("Note: Calories/macros depend on nutrition.csv values; portion is user-provided.")