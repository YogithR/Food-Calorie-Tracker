# 🍽️ MealSnap AI — Image-Based Food Calorie Tracker

MealSnap AI is an end-to-end ML web app that recognizes food from an uploaded image and estimates calories + macros (protein, carbs, fat) based on a nutrition lookup table and a user-selected portion size.

**Flow:** Upload photo → Model predicts food label → App calculates nutrition (per 100g × portion) → User can correct label → Save to meal history.

---

## ✨ Features
- ✅ Food image recognition using a deep learning model (Food-101 classes)
- ✅ Top-3 predictions with confidence scores
- ✅ Portion-based calorie + macro estimation (grams slider)
- ✅ “Final label” override when the model is wrong (improves accuracy for nutrition)
- ✅ Meal history logging (Breakfast/Lunch/Dinner/Snacks + notes)
- ✅ Export meal history as CSV

---

## 🧠 How It Works (ML + Nutrition)
### 1) Food Recognition (Computer Vision)
- The model predicts a **food label** from the uploaded image.
- It returns **Top-1 + Top-3** predictions with confidence scores.

### 2) Nutrition Estimation (Explainable)
The app uses a nutrition table (`nutrition.csv`) that contains values **per 100g**, then scales it using your portion:

> **scaled_value = (portion_g / 100) × value_per_100g**

This makes nutrition estimation transparent and easy to explain in interviews.

### 3) Correction + Logging
If the model prediction is wrong, users can select the closest matching label using **Final label**.  
The corrected result + nutrition is then saved into `feedback_log.csv` as meal history.

---

## 🧰 Tech Stack
- **Frontend / UI:** Streamlit
- **ML / Inference:** TensorFlow / Keras
- **Data Handling:** Pandas, NumPy
- **Image Processing:** Pillow (PIL)

---

## 📁 Project Structure
food-calorie-tracker/
│── app.py
│── requirements.txt
│── nutrition.csv
│── feedback_log.csv # created/updated when saving meals
│
└── models/
│── food101_mobilenetv2.keras # trained / saved model
└── class_names.txt # label names for predictions

---

## ▶️ Run Locally
### 1) Create venv (Windows)
```bash
python -m venv venv
venv\Scripts\activate

2) Install dependencies
pip install -r requirements.txt

3) Run the app
streamlit run app.py