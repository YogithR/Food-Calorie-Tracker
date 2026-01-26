# 🍽️ MealSnap AI — Image-Based Food Calorie & Nutrition Tracker

MealSnap AI is an end-to-end machine learning web application that recognizes food items from an uploaded image and estimates calories and macronutrients (protein, carbohydrates, and fat) based on a portion-aware nutrition system.

The app combines **computer vision**, **deep learning**, and an **explainable nutrition pipeline**, while allowing users to correct model predictions to improve accuracy — making it practical, transparent, and interview-ready.

---

## 🌐 Live Demo
🚀 **Deployed on Streamlit Community Cloud**  
👉 *(Paste your Streamlit deployment link here)*

---

## ✨ Key Features
- 📸 **Food Image Recognition** using a deep learning model trained on the Food-101 dataset
- 🧠 **Top-3 Predictions with Confidence Scores** for transparency
- 🍽️ **Portion-Based Nutrition Estimation** (grams → calories & macros)
- ✍️ **Human-in-the-Loop Correction Workflow** to override incorrect predictions
- 🗂️ **Meal History Logging** (date, meal type, nutrition values, notes)
- 📥 **CSV Export** of meal history for personal tracking
- 🌐 **Publicly Deployed Web App**

---

## 🧠 How the System Works

### 1️⃣ Food Recognition (Computer Vision)
- The user uploads a food image.
- A convolutional neural network (CNN) predicts the food category.
- The app displays:
  - **Top-1 prediction**
  - **Top-3 predictions with confidence scores**

This helps users understand model uncertainty instead of relying on a single guess.

---

### 2️⃣ Nutrition Estimation (Explainable Logic)
Nutrition values are stored in `nutrition.csv` **per 100g serving**.

The app scales nutrition using the formula:
scaled_value = (portion_g / 100) × nutrition_per_100g


This avoids black-box nutrition estimation and keeps calculations **simple, transparent, and explainable**.

---

### 3️⃣ User Correction (Human-in-the-Loop)
Food recognition models are not always perfect.

To handle this:
- Users can select a **Final Label** if the model prediction is incorrect
- Nutrition values are recalculated using the corrected label
- The corrected result is saved into meal history

This design reflects **real-world ML systems** where human feedback improves reliability.

---

## 🧰 Technology Stack

| Layer | Technology |
|------|-----------|
| Frontend / UI | Streamlit |
| Machine Learning | TensorFlow / Keras |
| Dataset | Food-101 |
| Data Processing | Pandas, NumPy |
| Image Handling | Pillow (PIL) |
| Storage | CSV-based logging |
| Deployment | Streamlit Community Cloud |

---

## 📁 Project Structure

Food-Calorie-Tracker/
│
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── nutrition.csv # Nutrition values per 100g
├── feedback_log.csv # Meal history (auto-created)
│
└── models/
├── food101_mobilenetv2.keras # Trained model
└── class_names.txt # Class labels


---

## ▶️ Run the App Locally

### 1️⃣ Clone the repository

git clone https://github.com/YogithR/Food-Calorie-Tracker.git
cd Food-Calorie-Tracker

2️⃣ Create and activate a virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the application
streamlit run app.py

###🌐 Deployment

This project is deployed using Streamlit Community Cloud:
Push the project to GitHub
Log in to Streamlit Community Cloud
Click New App
Select the GitHub repository and branch
Set app.py as the main file
Deploy and share the generated public URL

---


###⚠️ Known Limitations

Food-101 dataset does not cover all real-world or regional dishes
Portion size is user-entered (not estimated from the image)
Assumes one primary food item per image
Nutrition accuracy depends on nutrition.csv quality

---


###🚀 Future Enhancements

Fine-tuning the model with regional and custom food datasets
Multi-food detection within a single image
Portion estimation using food segmentation
Persistent database storage (SQLite / Firebase)
Mobile-first UI optimization
