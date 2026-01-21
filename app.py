# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hides INFO + WARNING logs
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # optional: disable oneDNN logs

# import time
# import numpy as np
# import pandas as pd
# import streamlit as st
# from PIL import Image
# import tensorflow as tf

# # -----------------------------
# # Config
# # -----------------------------
# IMG_SIZE = (224, 224)

# MODEL_PATH = "models/food101_mobilenetv2.keras"
# CLASS_NAMES_PATH = "models/class_names.txt"
# NUTRITION_PATH = "nutrition.csv"
# FEEDBACK_LOG_PATH = "feedback_log.csv"

# CUSTOM_LABELS = [
#     "grilled_chicken",
#     "roast_chicken",
#     "chicken_biryani",
#     "mutton_biryani",
#     "chicken_fried_rice",
# ]

# MEAL_TYPES = ["Breakfast", "Lunch", "Dinner", "Snacks"]

# HISTORY_COLUMNS = [
#     "timestamp",
#     "date",
#     "meal_type",
#     "meal_title",
#     "image_name",
#     "predicted_label",
#     "predicted_confidence",
#     "corrected_label",
#     "portion_g",
#     "calories",
#     "protein",
#     "carbs",
#     "fat",
#     "user_notes",
# ]

# # -----------------------------
# # Helpers: cached loaders
# # -----------------------------
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model(MODEL_PATH)

# @st.cache_data
# def load_class_names():
#     with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
#         return [line.strip() for line in f.readlines() if line.strip()]

# @st.cache_data
# def load_nutrition():
#     cols = ["label", "calories_per_100g", "protein_g", "carbs_g", "fat_g"]

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

# def preprocess_image(pil_img: Image.Image) -> np.ndarray:
#     img = pil_img.convert("RGB").resize(IMG_SIZE)
#     arr = np.array(img).astype(np.float32) / 255.0
#     return np.expand_dims(arr, axis=0)

# def get_topk(preds: np.ndarray, class_names: list[str], k: int = 3):
#     idxs = preds.argsort()[-k:][::-1]
#     return [(class_names[int(i)], float(preds[int(i)])) for i in idxs]

# def confidence_band(conf: float):
#     if conf >= 0.70:
#         return ("High", "Model is confident.")
#     if conf >= 0.40:
#         return ("Medium", "Some uncertainty. Check Top-3 to confirm.")
#     return ("Low", "Low confidence. Please verify using Top-3 or correct the label.")

# def nutrition_lookup(df: pd.DataFrame, label: str, portion_g: int):
#     if df.empty:
#         return None
#     if label is None:
#         return None

#     label = str(label).strip()
#     row = df[df["label"] == label]
#     if row.empty:
#         return None

#     row = row.iloc[0]
#     scale = float(portion_g) / 100.0

#     def safe_float(x):
#         try:
#             return float(x)
#         except Exception:
#             return None

#     cal = safe_float(row["calories_per_100g"])
#     p = safe_float(row["protein_g"])
#     c = safe_float(row["carbs_g"])
#     f = safe_float(row["fat_g"])

#     return {
#         "calories": (cal * scale) if cal is not None else None,
#         "protein": (p * scale) if p is not None else None,
#         "carbs": (c * scale) if c is not None else None,
#         "fat": (f * scale) if f is not None else None,
#     }

# def ensure_feedback_log_exists():
#     if not os.path.exists(FEEDBACK_LOG_PATH):
#         pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(FEEDBACK_LOG_PATH, index=False)

# def log_feedback(
#     meal_type: str,
#     meal_title: str,
#     image_name: str,
#     predicted: str,
#     confidence: float,
#     corrected: str,
#     portion_g: int,
#     info: dict | None,
#     user_notes: str
# ):
#     ensure_feedback_log_exists()

#     ts = time.strftime("%Y-%m-%d %H:%M:%S")
#     date_only = ts.split(" ")[0]

#     row = {
#         "timestamp": ts,
#         "date": date_only,
#         "meal_type": (meal_type or "").strip(),
#         "meal_title": (meal_title or "").strip(),
#         "image_name": image_name,
#         "predicted_label": predicted,
#         "predicted_confidence": confidence,
#         "corrected_label": corrected,
#         "portion_g": portion_g,
#         "calories": None if (info is None) else info.get("calories"),
#         "protein": None if (info is None) else info.get("protein"),
#         "carbs": None if (info is None) else info.get("carbs"),
#         "fat": None if (info is None) else info.get("fat"),
#         "user_notes": (user_notes or "").strip(),
#     }

#     pd.DataFrame([row]).to_csv(FEEDBACK_LOG_PATH, mode="a", header=False, index=False)

# def suggested_correction_index(options: list[str], top3: list[tuple[str, float]]):
#     option_to_idx = {opt: i for i, opt in enumerate(options)}
#     top3_labels = [t[0] for t in top3]

#     trigger = any(x in top3_labels for x in [
#         "fried_rice", "chicken_wings", "baby_back_ribs", "steak", "pork_chop", "prime_rib"
#     ])

#     if trigger:
#         for p in [
#             "chicken_fried_rice",
#             "chicken_biryani",
#             "mutton_biryani",
#             "grilled_chicken",
#             "roast_chicken",
#             "chicken_wings",
#         ]:
#             if p in option_to_idx:
#                 return option_to_idx[p]

#     top1 = top3[0][0]
#     if top1 in option_to_idx:
#         return option_to_idx[top1]

#     return 0

# def load_history_with_fixes(nutrition_df: pd.DataFrame):
#     """
#     Loads feedback_log.csv, ensures required columns exist,
#     and tries to fill missing portion/macros for older rows if possible.
#     """
#     ensure_feedback_log_exists()

#     try:
#         hist = pd.read_csv(FEEDBACK_LOG_PATH)
#     except Exception:
#         return pd.DataFrame(columns=HISTORY_COLUMNS)

#     # Add missing columns (for old log formats)
#     for col in HISTORY_COLUMNS:
#         if col not in hist.columns:
#             hist[col] = np.nan

#     # Fill date from timestamp if missing
#     def make_date(x):
#         try:
#             return str(x).split(" ")[0]
#         except Exception:
#             return ""

#     hist["date"] = hist["date"].fillna(hist["timestamp"].apply(make_date))

#     # If portion missing, default to 200 (so we can compute macros for old rows)
#     hist["portion_g"] = pd.to_numeric(hist["portion_g"], errors="coerce").fillna(200).astype(int)

#     # If macros missing, try compute again from nutrition.csv
#     for idx, row in hist.iterrows():
#         needs_macros = pd.isna(row["calories"]) or pd.isna(row["protein"]) or pd.isna(row["carbs"]) or pd.isna(row["fat"])
#         if needs_macros:
#             label = row["corrected_label"]
#             portion = int(row["portion_g"])
#             info = nutrition_lookup(nutrition_df, label, portion)
#             if info is not None:
#                 hist.at[idx, "calories"] = info["calories"]
#                 hist.at[idx, "protein"] = info["protein"]
#                 hist.at[idx, "carbs"] = info["carbs"]
#                 hist.at[idx, "fat"] = info["fat"]

#     return hist

# # -----------------------------
# # UI
# # -----------------------------
# st.set_page_config(page_title="Food Calorie Tracker", page_icon="🍽️", layout="wide")

# st.title("🍽️ Image-Based Food Calorie Tracker")
# st.write(
#     "Upload a food image → the ML model guesses the food → you can correct it → "
#     "the app estimates calories/macros using a nutrition lookup table."
# )

# # Pre-checks
# missing = []
# if not os.path.exists(MODEL_PATH):
#     missing.append(f"- Missing model file: `{MODEL_PATH}` (run `python train.py` first)")
# if not os.path.exists(CLASS_NAMES_PATH):
#     missing.append(f"- Missing labels file: `{CLASS_NAMES_PATH}` (created by training)")

# if missing:
#     st.error("Required files are missing:\n\n" + "\n".join(missing))
#     st.stop()

# model = load_model()
# class_names = load_class_names()
# nutrition_df = load_nutrition()
# correction_options = ["(use model prediction)"] + class_names + CUSTOM_LABELS

# left, right = st.columns([1, 1])

# with left:
#     st.subheader("1) Upload Image")
#     uploaded = st.file_uploader("Choose a food image", type=["jpg", "jpeg", "png"])

#     st.subheader("2) Portion Size")
#     portion_g = st.slider("Estimated portion (grams)", min_value=50, max_value=600, value=200, step=10)

# with right:
#     st.subheader("Results")

#     if uploaded is None:
#         st.info("Upload an image to see predictions and calorie estimates.")
#         st.stop()

#     img = Image.open(uploaded)
#     st.image(img, caption="Uploaded Image", use_container_width=True)

#     x = preprocess_image(img)
#     preds = model.predict(x, verbose=0)[0]
#     top1_label, top1_conf = get_topk(preds, class_names, k=1)[0]
#     top3 = get_topk(preds, class_names, k=3)

#     band, band_msg = confidence_band(top1_conf)
#     if band == "High":
#         st.success(f"Confidence: **{band}** — {band_msg}")
#     elif band == "Medium":
#         st.warning(f"Confidence: **{band}** — {band_msg}")
#     else:
#         st.error(f"Confidence: **{band}** — {band_msg}")

#     st.markdown("### 3) Model Guess (Top predictions)")
#     st.write(f"**Top guess:** `{top1_label}` (**{top1_conf:.2%}**)")
#     st.write("**Top-3 guesses:**")
#     for label, conf in top3:
#         st.write(f"- `{label}`: {conf:.2%}")

#     st.divider()

#     st.markdown("### 4) Correct the prediction")
#     try:
#         default_idx = suggested_correction_index(correction_options, top3)
#     except Exception:
#         default_idx = 0

#     corrected = st.selectbox("Correct label:", options=correction_options, index=default_idx)

#     user_corrected = corrected != "(use model prediction)"
#     label_for_nutrition = top1_label if not user_corrected else corrected

#     if (top1_conf < 0.70) and (not user_corrected):
#         st.warning("Prediction is uncertain. For accurate calories/macros, choose the correct label above.")

#     if user_corrected:
#         st.info(f"Using corrected label **`{label_for_nutrition}`** for nutrition calculation.")

#     st.markdown("### 5) Estimated Nutrition")
#     info = nutrition_lookup(nutrition_df, label_for_nutrition, portion_g)

#     if info is None:
#         st.warning(
#             f"Nutrition not found for label: `{label_for_nutrition}`.\n\n"
#             "Add a row to `nutrition.csv` like this:\n\n"
#             f"`{label_for_nutrition},<calories_per_100g>,<protein_g>,<carbs_g>,<fat_g>`"
#         )
#     else:
#         st.write(f"Portion used: **{portion_g}g**")
#         c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
#         with c1:
#             st.metric("🔥 Calories (kcal)", f"{info['calories']:.0f}" if info["calories"] is not None else "—")
#         with c2:
#             st.metric("💪 Protein (g)", f"{info['protein']:.1f}" if info["protein"] is not None else "—")
#         with c3:
#             st.metric("🍞 Carbs (g)", f"{info['carbs']:.1f}" if info["carbs"] is not None else "—")
#         with c4:
#             st.metric("🥑 Fat (g)", f"{info['fat']:.1f}" if info["fat"] is not None else "—")

#     st.divider()

#     st.markdown("### 6) Save to Meal History")

#     # Dropdown for meal type + optional title
#     meal_type = st.selectbox("Meal type", MEAL_TYPES, index=1)  # default Lunch
#     meal_title = st.text_input("Meal title (optional)", placeholder="Example: Chicken fried rice")

#     notes = st.text_area(
#         "Notes (optional)",
#         placeholder="Example: Extra chicken added, less oil, homemade.",
#         height=90
#     )

#     # Allow saving even if user didn't correct (still useful)
#     save_allowed = True
#     if label_for_nutrition is None or str(label_for_nutrition).strip() == "":
#         save_allowed = False

#     colA, colB = st.columns([1, 2])
#     with colA:
#         save_clicked = st.button("Save to history", disabled=not save_allowed)
#     with colB:
#         st.caption("Saves date + meal type + label + portion + macros into feedback_log.csv and shows in Meal History.")

#     if save_clicked:
#         log_feedback(
#             meal_type=meal_type,
#             meal_title=meal_title,
#             image_name=getattr(uploaded, "name", "uploaded_image"),
#             predicted=top1_label,
#             confidence=top1_conf,
#             corrected=label_for_nutrition,
#             portion_g=int(portion_g),
#             info=info,
#             user_notes=notes
#         )
#         st.success("Saved! Added to Meal History below.")

#     st.divider()

#     show_hist = st.checkbox("Show Meal History", value=True)

#     if show_hist:
#         hist = load_history_with_fixes(nutrition_df)

#         if hist.empty:
#             st.info("No history yet. Save a meal to see it here.")
#         else:
#             hist = hist.copy()
#             # latest first
#             hist = hist.sort_values("timestamp", ascending=False)

#             # Display only Date (not full timestamp)
#             display = pd.DataFrame()
#             display["date"] = hist["date"].fillna("")
#             display["meal_type"] = hist["meal_type"].fillna("")
#             display["meal_title"] = hist["meal_title"].fillna("")
#             display["corrected_label"] = hist["corrected_label"].fillna("")
#             display["portion_g"] = pd.to_numeric(hist["portion_g"], errors="coerce").fillna(0).astype(int)

#             for col in ["calories", "protein", "carbs", "fat"]:
#                 display[col] = pd.to_numeric(hist[col], errors="coerce")

#             # Round
#             display["calories"] = display["calories"].round(0)
#             display["protein"] = display["protein"].round(1)
#             display["carbs"] = display["carbs"].round(1)
#             display["fat"] = display["fat"].round(1)
#             display["user_notes"] = hist["user_notes"].fillna("")

#             st.dataframe(display, use_container_width=True)

# st.caption(
#     "Note: Calories/macros come from a nutrition lookup table. Portion size is user-provided; nutrition varies by recipe/brand."
# )







##### New CODE ##########

# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# import time
# import numpy as np
# import pandas as pd
# import streamlit as st
# from PIL import Image
# import tensorflow as tf

# # -----------------------------
# # Paths (stable)
# # -----------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# def p(*parts: str) -> str:
#     return os.path.join(BASE_DIR, *parts)

# IMG_SIZE = (224, 224)

# MODEL_PATH = p("models", "food101_mobilenetv2.keras")
# CLASS_NAMES_PATH = p("models", "class_names.txt")
# NUTRITION_PATH = p("nutrition.csv")
# FEEDBACK_LOG_PATH = p("feedback_log.csv")  # always in same folder as app.py

# CUSTOM_LABELS = [
#     "avocado_toast",
#     "avocado_toast_with_egg",
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
#     "notes",
# ]

# # -----------------------------
# # Loaders
# # -----------------------------
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model(MODEL_PATH)

# @st.cache_data
# def load_class_names():
#     with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
#         return [line.strip() for line in f if line.strip()]

# @st.cache_data
# def load_nutrition():
#     cols = ["label", "calories_per_100g", "protein_g", "carbs_g", "fat_g"]
#     if not os.path.exists(NUTRITION_PATH) or os.path.getsize(NUTRITION_PATH) == 0:
#         return pd.DataFrame(columns=cols)

#     df = pd.read_csv(NUTRITION_PATH)
#     for c in cols:
#         if c not in df.columns:
#             df[c] = np.nan
#     df = df[cols].copy()
#     df["label"] = df["label"].astype(str).str.strip()
#     return df

# def ensure_feedback_log_exists():
#     if not os.path.exists(FEEDBACK_LOG_PATH):
#         pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(FEEDBACK_LOG_PATH, index=False)

# def safe_read_history():
#     ensure_feedback_log_exists()
#     try:
#         df = pd.read_csv(FEEDBACK_LOG_PATH)
#         # if file exists but has weird formatting
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
#     df = pd.DataFrame([row], columns=HISTORY_COLUMNS)
#     df.to_csv(FEEDBACK_LOG_PATH, mode="a", header=False, index=False)

# def preprocess_image(pil_img: Image.Image) -> np.ndarray:
#     img = pil_img.convert("RGB").resize(IMG_SIZE)
#     arr = np.array(img).astype(np.float32) / 255.0
#     return np.expand_dims(arr, axis=0)

# def get_topk(preds: np.ndarray, class_names: list[str], k: int = 3):
#     idxs = preds.argsort()[-k:][::-1]
#     return [(class_names[int(i)], float(preds[int(i)])) for i in idxs]

# def nutrition_lookup(df: pd.DataFrame, label: str, portion_g: int):
#     if df.empty:
#         return None
#     row = df[df["label"] == label]
#     if row.empty:
#         return None

#     row = row.iloc[0]
#     scale = float(portion_g) / 100.0
#     return {
#         "calories": float(row["calories_per_100g"]) * scale,
#         "protein": float(row["protein_g"]) * scale,
#         "carbs": float(row["carbs_g"]) * scale,
#         "fat": float(row["fat_g"]) * scale,
#     }

# # -----------------------------
# # UI
# # -----------------------------
# st.set_page_config("Food Calorie Tracker", "🍽️", layout="wide")
# st.title("🍽️ Image-Based Food Calorie Tracker")

# # init session history (SHOWS IMMEDIATELY even if CSV issues)
# if "history" not in st.session_state:
#     st.session_state.history = safe_read_history()

# # Checks
# if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
#     st.error("Model or class names missing. Please ensure models/ folder is correct.")
#     st.stop()

# model = load_model()
# class_names = load_class_names()
# nutrition_df = load_nutrition()

# left, right = st.columns([1, 1])

# with left:
#     st.subheader("1) Upload Image")
#     uploaded = st.file_uploader("Choose a food image", ["jpg", "jpeg", "png"])

#     st.subheader("2) Portion")
#     portion_g = st.slider("Portion (grams)", 50, 600, 200, 10)

# with right:
#     if uploaded is None:
#         st.info("Upload an image to begin.")
#         st.stop()

#     img = Image.open(uploaded)
#     st.image(img, use_container_width=True)

#     preds = model.predict(preprocess_image(img), verbose=0)[0]
#     top3 = get_topk(preds, class_names, k=3)
#     st.divider()

#     st.markdown("### 3) Model Prediction")
#     for lab, conf in top3:
#         st.write(f"- `{lab}`: **{conf:.2%}**")
#     st.divider()

#     # ✅ final label chooser (solves avocado toast problem)
#     st.markdown("### 4) Select Final Label (for nutrition)")
#     all_labels = ["(use top prediction)"] + CUSTOM_LABELS + class_names
#     default_final = "(use top prediction)"
#     final_label_choice = st.selectbox("Final label", all_labels, index=0)

#     final_label = top3[0][0] if final_label_choice == default_final else final_label_choice
#     st.info(f"Using **{final_label}** for nutrition + saving history.")
#     st.divider()

#     st.markdown("### 5) Estimated Nutrition")
#     macros = nutrition_lookup(nutrition_df, final_label, portion_g)
#     meal_type = st.selectbox("Meal type", MEAL_TYPES, index=1)

#     if macros is None:
#         st.warning(f"No nutrition row found for `{final_label}` in nutrition.csv")
#     else:
#         c1, c2, c3, c4 = st.columns(4)
#         c1.metric("🔥 Calories", f"{macros['calories']:.0f}")
#         c2.metric("💪 Protein (g)", f"{macros['protein']:.1f}")
#         c3.metric("🍞 Carbs (g)", f"{macros['carbs']:.1f}")
#         c4.metric("🥑 Fat (g)", f"{macros['fat']:.1f}")
#         st.divider()


#     st.markdown("### 6) Save to Meal History")
#     notes = st.text_area("Notes (optional)", placeholder="Example: homemade, less oil, extra chicken")
#     if st.button("Save meal"):
#         today = time.strftime("%Y-%m-%d")
#         row = {
#             "date": today,
#             "meal_type": meal_type,
#             "final_label": final_label,
#             "portion_g": int(portion_g),
#             "calories": None if macros is None else round(macros["calories"], 0),
#             "protein": None if macros is None else round(macros["protein"], 1),
#             "carbs": None if macros is None else round(macros["carbs"], 1),
#             "fat": None if macros is None else round(macros["fat"], 1),
#             "notes": notes.strip(),
#         }

#         # Save CSV
#         save_history_row(row)

#         # Update UI immediately (this is the main fix)
#         st.session_state.history = pd.concat(
#             [pd.DataFrame([row]), st.session_state.history],
#             ignore_index=True
#         )

#         st.success("Saved! History updated.")
#         st.rerun()

#     st.divider()

#     show_hist = st.checkbox("Show Meal History", value=True)
#     if show_hist:
#         if st.button("Reload history from file"):
#             st.session_state.history = safe_read_history()
#             st.rerun()

#         hist = st.session_state.history.copy()
#         if hist.empty:
#             st.info("No history yet. Save a meal to see it here.")
#         else:
#             st.dataframe(hist, use_container_width=True)






##################### NEW CODE AFTER THE UI UPDATE ##############################





# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# import time
# import numpy as np
# import pandas as pd
# import streamlit as st
# from PIL import Image
# import tensorflow as tf

# # -----------------------------
# # Page config (WIDE)
# # -----------------------------
# st.set_page_config(page_title="Food Calorie Tracker", page_icon="🍽️", layout="wide")

# # -----------------------------
# # Paths
# # -----------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# def p(*parts: str) -> str:
#     return os.path.join(BASE_DIR, *parts)

# IMG_SIZE = (224, 224)

# MODEL_PATH = p("models", "food101_mobilenetv2.keras")
# CLASS_NAMES_PATH = p("models", "class_names.txt")
# NUTRITION_PATH = p("nutrition.csv")
# FEEDBACK_LOG_PATH = p("feedback_log.csv")

# # -----------------------------
# # App config
# # -----------------------------
# CUSTOM_LABELS = [
#     "avocado_toast",
#     "avocado_toast_with_egg",
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
#     "notes",
# ]

# # -----------------------------
# # Loaders
# # -----------------------------
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model(MODEL_PATH)

# @st.cache_data
# def load_class_names():
#     with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
#         return [line.strip() for line in f if line.strip()]

# @st.cache_data
# def load_nutrition():
#     cols = ["label", "calories_per_100g", "protein_g", "carbs_g", "fat_g"]
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

# # -----------------------------
# # History helpers
# # -----------------------------
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

# # -----------------------------
# # ML helpers
# # -----------------------------
# def preprocess_image(pil_img: Image.Image) -> np.ndarray:
#     img = pil_img.convert("RGB").resize(IMG_SIZE)
#     arr = np.array(img).astype(np.float32) / 255.0
#     return np.expand_dims(arr, axis=0)

# def get_topk(preds: np.ndarray, class_names: list[str], k: int = 3):
#     idxs = preds.argsort()[-k:][::-1]
#     return [(class_names[int(i)], float(preds[int(i)])) for i in idxs]

# def nutrition_lookup(df: pd.DataFrame, label: str, portion_g: int):
#     if df.empty:
#         return None
#     row = df[df["label"] == label]
#     if row.empty:
#         return None
#     row = row.iloc[0]
#     scale = float(portion_g) / 100.0
#     return {
#         "calories": float(row["calories_per_100g"]) * scale,
#         "protein": float(row["protein_g"]) * scale,
#         "carbs": float(row["carbs_g"]) * scale,
#         "fat": float(row["fat_g"]) * scale,
#     }

# def pct(x: float) -> str:
#     return f"{x*100:.2f}%"

# # -----------------------------
# # Header
# # -----------------------------
# st.title("🍽️ Image-Based Food Calorie Tracker")
# st.write("Upload a food photo → Get instant calorie estimate.")

# # -----------------------------
# # Init state
# # -----------------------------
# if "history" not in st.session_state:
#     st.session_state.history = safe_read_history()

# # -----------------------------
# # Checks
# # -----------------------------
# missing = []
# if not os.path.exists(MODEL_PATH):
#     missing.append(f"Missing model: {MODEL_PATH}")
# if not os.path.exists(CLASS_NAMES_PATH):
#     missing.append(f"Missing class names: {CLASS_NAMES_PATH}")
# if missing:
#     st.error("Required files are missing:\n\n- " + "\n- ".join(missing))
#     st.stop()

# model = load_model()
# class_names = load_class_names()
# nutrition_df = load_nutrition()

# # -----------------------------
# # Sidebar controls
# # -----------------------------
# st.sidebar.header("Controls")

# uploaded = st.sidebar.file_uploader(
#     "Upload food image",
#     type=["jpg", "jpeg", "png"],
#     help="Upload a clear photo of a single dish for best results."
# )

# portion_g = st.sidebar.slider(
#     "Portion size (grams)",
#     50, 600, 200, 10,
#     help="This scales calories/macros from per-100g values in nutrition.csv."
# )

# meal_type = st.sidebar.selectbox(
#     "Meal type",
#     MEAL_TYPES,
#     index=1,
#     help="Pick Breakfast/Lunch/Dinner/Snacks for your history log."
# )

# # notes = st.sidebar.text_area(
# #     "Notes (optional)",
# #     placeholder="Example: homemade, less oil, extra chicken",
# #     help="Optional notes saved into your meal history."
# # )

# # -----------------------------
# # Main area layout
# # -----------------------------
# col_img, col_out = st.columns([1.3, 1.0], gap="large")

# # Image preview placeholder
# with col_img:
#     st.subheader("Image Preview")
#     if uploaded is None:
#         st.info("📷 Upload an image from the sidebar to preview it here.")
#         st.image(
#             np.zeros((300, 500, 3), dtype=np.uint8),
#             caption="Your uploaded image (preview will appear here)",
#             use_container_width=True
#         )
#         st.stop()

#     img = Image.open(uploaded)
#     st.image(img, caption="Your uploaded image", use_container_width=True)

# # Predict once
# x = preprocess_image(img)
# preds = model.predict(x, verbose=0)[0]
# top3 = get_topk(preds, class_names, k=3)
# top1_label, top1_conf = top3[0]

# # Final label choices
# all_labels = ["(use top prediction)"] + CUSTOM_LABELS + class_names
# final_label_choice = st.sidebar.selectbox(
#     "Final label (for nutrition)",
#     all_labels,
#     index=0,
#     help="Select the closest match. This label is used for nutrition + saving history."
# )
# final_label = top1_label if final_label_choice == "(use top prediction)" else final_label_choice
# # Sidebar: Notes AFTER final label (as requested)
# notes = st.sidebar.text_area(
#     "Notes (optional)",
#     placeholder="Example: homemade, less oil, extra chicken",
#     help="Optional notes saved into your meal history."
# )

# # Nutrition
# macros = nutrition_lookup(nutrition_df, final_label, int(portion_g))

# # -----------------------------
# # Tabs: Predictions / Nutrition / History
# # -----------------------------
# with col_out:
#     st.subheader("Results")
#     tab_pred, tab_nut, tab_hist = st.tabs(["Predictions", "Nutrition", "History"])

#     # -------- Predictions tab --------
#     with tab_pred:
#         # Top prediction metric
#         st.metric(
#             "Top Prediction",
#             f"{top1_label} ({pct(top1_conf)})",
#             delta=pct(top1_conf)
#         )

#         # Bar chart for Top-3
#         top3_df = pd.DataFrame(
#             {"label": [x[0] for x in top3], "confidence": [x[1] for x in top3]}
#         ).set_index("label")

#         st.write("Top-3 confidence")
#         st.bar_chart(top3_df)

#         st.caption("Tip: Food-101 has only 101 fixed classes. Some foods (like avocado toast) may map to the closest known class.")

#     # -------- Nutrition tab --------
#     with tab_nut:
#         st.write("**Final label used for nutrition:**", f"`{final_label}`")
#         st.caption("Select the closest match in the sidebar if the model guesses wrong.")

#         if macros is None:
#             st.warning(f"No nutrition row found for `{final_label}` in nutrition.csv")
#         else:
#             # Icons + metrics
#             c1, c2, c3, c4 = st.columns(4)
#             c1.metric("🔥 Calories", f"{macros['calories']:.0f} kcal")
#             c2.metric("💪 Protein", f"{macros['protein']:.1f} g")
#             c3.metric("🍞 Carbs", f"{macros['carbs']:.1f} g")
#             c4.metric("🥑 Fat", f"{macros['fat']:.1f} g")

#             st.caption(f"Portion used: {portion_g}g (scaled from per-100g nutrition.csv values)")

#         # Save button in Nutrition tab (so it feels logical)
#         st.divider()
#         if st.button("Save to Meal History"):
#             today = time.strftime("%Y-%m-%d")
#             row = {
#                 "date": today,
#                 "meal_type": meal_type,
#                 "final_label": final_label,
#                 "portion_g": int(portion_g),
#                 "calories": None if macros is None else round(macros["calories"], 0),
#                 "protein": None if macros is None else round(macros["protein"], 1),
#                 "carbs": None if macros is None else round(macros["carbs"], 1),
#                 "fat": None if macros is None else round(macros["fat"], 1),
#                 "notes": notes.strip(),
#             }
#             save_history_row(row)

#             # Update UI immediately
#             st.session_state.history = pd.concat(
#                 [pd.DataFrame([row]), st.session_state.history],
#                 ignore_index=True
#             )

#             st.success("Saved! Go to the History tab to view it.")
#             st.rerun()

#     # -------- History tab --------
#     with tab_hist:
#         st.write("Meal History")

#         # Reload button (reads file back)
#         if st.button("Reload history from file"):
#             st.session_state.history = safe_read_history()
#             st.rerun()

#         hist = st.session_state.history.copy()

#         if hist.empty:
#             st.info("No history yet. Save a meal to see it here.")
#         else:
#             # show newest first
#             hist = hist.sort_values("date", ascending=False)

#             # nice display
#             st.dataframe(hist, use_container_width=True)

#             # Export button
#             csv_bytes = hist.to_csv(index=False).encode("utf-8")
#             st.download_button(
#                 "Export history (CSV)",
#                 data=csv_bytes,
#                 file_name="meal_history.csv",
#                 mime="text/csv"
#             )

# st.caption("Note: Calories/macros depend on your nutrition.csv values; portion is user-provided.")











import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="MealSnap AI", page_icon="🍽️", layout="wide")

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def p(*parts: str) -> str:
    return os.path.join(BASE_DIR, *parts)

IMG_SIZE = (224, 224)
MODEL_PATH = p("models", "food101_mobilenetv2.keras")
CLASS_NAMES_PATH = p("models", "class_names.txt")
NUTRITION_PATH = p("nutrition.csv")
FEEDBACK_LOG_PATH = p("feedback_log.csv")

# -----------------------------
# App config
# -----------------------------
CUSTOM_LABELS = [
    "avocado_toast",
    "avocado_toast_with_egg",
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
    "notes",
]

# -----------------------------
# CSS (BLACK SIDEBAR + GRAY CONTROLS) + FIX "FINAL LABEL NOT VISIBLE" IN NUTRITION TAB
# -----------------------------
st.markdown(
    """
<style>
/* ===== App background ===== */
.stApp {
  background: linear-gradient(135deg, #DFF7EA 0%, #FFFFFF 55%, #F2FBF6 100%);
  padding-bottom: 320px !important;
}
main .block-container{ padding-bottom: 320px !important; }

/* ===== Sidebar: BLACK ===== */
section[data-testid="stSidebar"]{
  background: #0b0f14 !important;
  border-right: 1px solid rgba(255,255,255,0.08);
  overflow: visible !important;
}
section[data-testid="stSidebar"] *{ color: #e5e7eb !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{ color: #ffffff !important; }

/* ===== Controls: GRAY inputs ===== */
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

/* ===== File uploader visible ===== */
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

/* ===== Selectbox closed control ===== */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div{
  background: var(--sb-box-bg) !important;
  border: 1px solid var(--sb-box-border) !important;
  border-radius: 12px !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] span,
section[data-testid="stSidebar"] div[data-baseweb="select"] div{
  color: var(--sb-box-text) !important;
}

/* ===== Dropdown menu: dark gray (not black) + on top ===== */
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

/* ===== Glass cards ===== */
.glass-card{
  background: rgba(255,255,255,0.55);
  border: 1px solid rgba(255,255,255,0.35);
  box-shadow: 0 10px 30px rgba(2, 44, 25, 0.10);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border-radius: 15px;
  padding: 16px 18px;
}
.glass-card, .glass-card *{ color: #0f172a !important; }

/* ===== Metrics readable ===== */
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

/* ===== Tabs ===== */
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

/* ===== Hero ===== */
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
@media (max-width: 640px){
  .hero-wrap{ gap: 12px; }
  .hero-icon{ font-size: 44px; padding: 10px 12px; }
  .hero-title{ font-size: 28px; }
  .hero-subtitle{ font-size: 14px; }
}

/* =========================================================
   FIX YOU REPORTED:
   In Nutrition tab, the "Final label used..." line looked blank.
   This happened because the <code> tag got styled too lightly.
   Force <code> chips to be visible everywhere.
   ========================================================= */
code{
  background: #0f172a !important;
  color: #e5e7eb !important;
  padding: 2px 8px !important;
  border-radius: 8px !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  font-weight: 600 !important;
}
</style>
""",
    unsafe_allow_html=True
)

# -----------------------------
# Hero header
# -----------------------------
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

# -----------------------------
# Loaders
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

@st.cache_data
def load_nutrition():
    cols = ["label", "calories_per_100g", "protein_g", "carbs_g", "fat_g"]
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

def ensure_feedback_log_exists():
    if not os.path.exists(FEEDBACK_LOG_PATH):
        pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(FEEDBACK_LOG_PATH, index=False)

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

def preprocess_image(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def get_topk(preds, class_names, k=3):
    idxs = preds.argsort()[-k:][::-1]
    return [(class_names[int(i)], float(preds[int(i)])) for i in idxs]

def nutrition_lookup(df, label, portion_g):
    if df.empty:
        return None
    row = df[df["label"] == label]
    if row.empty:
        return None
    row = row.iloc[0]
    scale = float(portion_g) / 100.0
    return {
        "calories": float(row["calories_per_100g"]) * scale,
        "protein": float(row["protein_g"]) * scale,
        "carbs": float(row["carbs_g"]) * scale,
        "fat": float(row["fat_g"]) * scale,
    }

def pct(x: float) -> str:
    return f"{x*100:.2f}%"

# -----------------------------
# Init state
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = safe_read_history()

# -----------------------------
# Checks
# -----------------------------
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

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

uploaded = st.sidebar.file_uploader(
    "Upload food image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photo of a single dish for best results."
)

portion_g = st.sidebar.slider(
    "🍽️ Estimated portion (grams)",
    50, 600, 200, 10,
    help="This scales calories/macros from nutrition.csv values per 100g. Example: 200g = 2× the per-100g values."
)

meal_type = st.sidebar.selectbox(
    "Meal type",
    MEAL_TYPES,
    index=0,
    help="Pick Breakfast/Lunch/Dinner/Snacks for your history log."
)

# -----------------------------
# Main layout
# -----------------------------
col_img, col_out = st.columns([1.3, 1.0], gap="large")

with col_img:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Image Preview")
    if uploaded is None:
        st.info("📷 Upload an image from the sidebar to preview it here.")
        st.image(
            np.zeros((320, 520, 3), dtype=np.uint8),
            caption="Your uploaded image (preview will appear here)",
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    img = Image.open(uploaded)
    st.image(img, caption="Your uploaded image", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Predict once
x = preprocess_image(img)
preds = model.predict(x, verbose=0)[0]
top3 = get_topk(preds, class_names, k=3)
top1_label, top1_conf = top3[0]

# Sidebar final label + notes (final label above notes)
all_labels = ["(use top prediction)"] + CUSTOM_LABELS + class_names
final_label_choice = st.sidebar.selectbox(
    "Final label (for nutrition)",
    all_labels,
    index=0,
    help="Select the closest match for accurate nutrition + saving history."
)
final_label = top1_label if final_label_choice == "(use top prediction)" else final_label_choice

notes = st.sidebar.text_area(
    "Notes (optional)",
    placeholder="Example: homemade, less oil, extra chicken",
    help="Optional notes saved into your meal history."
)

macros = nutrition_lookup(nutrition_df, final_label, int(portion_g))

with col_out:
    st.subheader("Results")
    tab_pred, tab_nut, tab_hist = st.tabs(["Predictions", "Nutrition", "History"])

    # Predictions tab
    with tab_pred:
        st.metric("🥚 Top Prediction", f"{top1_label} ({pct(top1_conf)})", delta=pct(top1_conf))
        top3_df = pd.DataFrame(
            {"label": [x[0] for x in top3], "confidence": [x[1] for x in top3]}
        ).set_index("label")
        st.write("Top-3 confidence")
        st.bar_chart(top3_df)

    # Nutrition tab
    with tab_nut:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        # IMPORTANT: use st.code so it ALWAYS looks like a visible chip, even if CSS changes
        st.write("**Final label used for nutrition:**")
        st.code(final_label, language="text")
        st.caption("If the model guessed wrong, pick a better label in the sidebar.")
        st.markdown("</div>", unsafe_allow_html=True)

        if macros is None:
            st.warning(f"No nutrition row found for `{final_label}` in nutrition.csv")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🔥 Calories", f"{macros['calories']:.0f} kcal")
            c2.metric("💪 Protein", f"{macros['protein']:.1f} g")
            c3.metric("🍞 Carbs", f"{macros['carbs']:.1f} g")
            c4.metric("🥑 Fat", f"{macros['fat']:.1f} g")
            st.caption(f"Portion used: {portion_g}g")

        st.divider()
        if st.button("Save to Meal History"):
            today = time.strftime("%Y-%m-%d")
            row = {
                "date": today,
                "meal_type": meal_type,
                "final_label": final_label,
                "portion_g": int(portion_g),
                "calories": None if macros is None else round(macros["calories"], 0),
                "protein": None if macros is None else round(macros["protein"], 1),
                "carbs": None if macros is None else round(macros["carbs"], 1),
                "fat": None if macros is None else round(macros["fat"], 1),
                "notes": notes.strip(),
            }
            save_history_row(row)
            st.session_state.history = pd.concat([pd.DataFrame([row]), st.session_state.history], ignore_index=True)
            st.success("Saved! Go to History tab.")
            st.rerun()

    # History tab
    with tab_hist:
        hist = st.session_state.history.copy()
        if hist.empty:
            st.info("No history yet. Save a meal to see it here.")
        else:
            recent = hist.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🔥 Last Calories", f"{recent['calories']} kcal" if pd.notna(recent["calories"]) else "—")
            m2.metric("💪 Last Protein", f"{recent['protein']} g" if pd.notna(recent["protein"]) else "—")
            m3.metric("🍞 Last Carbs", f"{recent['carbs']} g" if pd.notna(recent["carbs"]) else "—")
            m4.metric("🥑 Last Fat", f"{recent['fat']} g" if pd.notna(recent["fat"]) else "—")

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.dataframe(hist, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            cA, cB = st.columns([1, 1])
            with cA:
                if st.button("Reload history from file"):
                    st.session_state.history = safe_read_history()
                    st.rerun()
            with cB:
                csv_bytes = hist.to_csv(index=False).encode("utf-8")
                st.download_button("Export history (CSV)", data=csv_bytes, file_name="meal_history.csv", mime="text/csv")

st.caption("Note: Calories/macros depend on your nutrition.csv values; portion is user-provided.")
    