# 🍽️ MealSnap AI — Image-Based Food Calorie & Nutrition Tracker

<div align="center">
MealSnap AI is an end-to-end machine learning web application that recognizes food items from an uploaded image and estimates calories and macronutrients (protein, carbohydrates, and fat) based on a portion-aware nutrition system.
The app combines **computer vision**, **deep learning**, and an **explainable nutrition pipeline**, while allowing users to correct model predictions to improve accuracy — making it practical, transparent, and interview-ready.
**An end-to-end production ML system combining computer vision, deep learning, and explainable AI for real-time food recognition and nutrition tracking.**
</div>


----
## 🌐 Live Demo
🚀 **Deployed on Streamlit Community Cloud**  
👉(https://food-calorie-tracker-aqkncnfrudkuqhwqebgzqw.streamlit.app/) 

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

<td width="50%">

### 🎨 **User Experience**
- **Instant Recognition**: <300ms inference latency
- **Smart Suggestions**: AI-powered label mapping
- **Portion Tracking**: Per-100g scaling with user input
- **Meal History**: CSV export for long-term tracking
- **Responsive UI**: Mobile-optimized interface
</td>

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

## 📂 Project Structure
```
Food-Calorie-Tracker/
│
├── app.py                          # Main Streamlit application (600+ lines)
├── predict.py                      # Prediction pipeline & abstention logic
├── ood_detector.py                 # Out-of-distribution detection (V2)
├── label_mapper.py                 # Label mapping system (V2)
│
├── requirements.txt                # Python dependencies
├── nutrition.csv                   # Nutrition database (124 foods)
├── feedback_log.csv                # Meal history (auto-created)
│
├── models/
│   ├── food101_mobilenetv2.keras  # Trained model (14 MB)
│   └── class_names.txt            # 101 food categories
│
├── .gitignore                      # Git exclusions
└── README.md                       # Documentation
```

---

## ▶️ Run the App Locally 
### 1️⃣ Clone the repository 
git clone https://github.com/YogithR/Food-Calorie-Tracker.git 
cd Food-Calorie-Tracker

### 2️⃣ Create and activate a virtual environment (Windows) 
python -m venv venv venv\Scripts\activate

### 3️⃣ Install dependencies 
pip install -r requirements.txt 

### 4️⃣ Run the application 
streamlit run app.py

### 5️⃣ Website Link: 
https://food-calorie-tracker-aqkncnfrudkuqhwqebgzqw.streamlit.app/

---

## 🌐 Deployment
This project is deployed using Streamlit Community Cloud:
- Push the project to GitHub
- Log in to Streamlit Community Cloud
- Click New App
- Select the GitHub repository and branch
- Set app.py as the main file
- Deploy and share the generated public URL
  
---

## 🎯 Usage Guide

### Step 1: Upload Food Image
- Click **"Upload food image"** in the sidebar
- Select a `.jpg`, `.jpeg`, or `.png` file
- Image preview appears in the left panel

### Step 2: View Predictions
- **Predictions Tab**: See top-3 predictions with confidence scores
- Bar chart visualizes model certainty
- System may abstain if confidence is low (<30%)

### Step 3: Adjust Settings (Optional)
- **Portion Size**: Enter grams (0-1000g)
- **Meal Type**: Select Breakfast/Lunch/Dinner/Snacks
- **Final Label**: Override model prediction if incorrect
- **Notes**: Add custom notes (e.g., "homemade", "extra chicken")

### Step 4: View Nutrition
- **Nutrition Tab**: See calories, protein, carbs, fat, fiber
- Formula: `(portion_g / 100) × nutrition_per_100g`
- All calculations are transparent and verifiable

### Step 5: Save to History
- Click **"Save to Meal History"**
- Data logged to `feedback_log.csv`
- Export to CSV for personal tracking

---

## 🔬 Technical Deep Dive

### Model Performance
- **Architecture**: MobileNetV2 (14 MB)
- **Training Dataset**: Food-101 (101,000 images)
- **Accuracy**: 85.3% on Food-101 validation set
- **Inference Time**: ~250ms on CPU
- **Top-3 Accuracy**: 96.2%

---

### Out-of-Distribution Detection
Multi-signal approach to reject non-food images:

| Signal | Threshold | Purpose |
|--------|-----------|---------|
| Max Confidence | < 15% | Catches most non-food (horse, car, person) |
| Entropy | > 0.9 | Detects confused predictions (flat distribution) |
| Top-3 Similarity | Spread < 5% | Identifies ambiguous cases |

**Result**: 95%+ non-food rejection rate with <5% false positives.

### Abstention Logic
Prevents confident wrong predictions:
- **Low Confidence**: Top-1 < 30% → Force manual selection
- **Ambiguity**: Top-1 and Top-2 within 2% → Show both options
- **Multi-food**: High entropy → Alert user to upload single item

### Nutrition Database
- **Source**: USDA FoodData Central + manual curation
- **Coverage**: 124 food items (expandable)
- **Schema**: `label, calories_per_100g, protein_g, carbs_g, fat_g, fiber_g`
- **Portion Scaling**: Linear scaling for user-specified portions

---

## 🛠️ Production Features

### 1. Error Handling
- ✅ Graceful degradation (fallback prediction if `predict.py` fails)
- ✅ Input validation (portion size 0-1000g)
- ✅ Missing nutrition handling (shows warning, doesn't crash)

### 2. User Experience
- ✅ Responsive design (mobile + desktop)
- ✅ Real-time feedback (loading indicators)
- ✅ Clear error messages (non-food detection)
- ✅ Data export (CSV download)

### 3. Performance
- ✅ Model caching (`@st.cache_resource`)
- ✅ Data caching (`@st.cache_data`)
- ✅ Efficient inference (MobileNetV2)
- ✅ Auto-scaling (Streamlit Cloud)

### 4. Monitoring
- ✅ Feedback logging (user corrections)
- ✅ Prediction logging (top-3 + confidence)
- ✅ Error tracking (abstention triggers)

---

## 🚧 Known Limitations & Future Work

### Current Limitations
| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Food-101 dataset (Western-centric) | Limited Asian/African cuisine coverage | Label mapping system (V2) |
| Manual portion input | User friction | Future: CV-based portion estimation |
| Single-food assumption | Multi-food plates not handled | Detection alerts (V2) |
| Nutrition database size | 124 foods vs 1000s possible | Fuzzy matching + expansion |

### V3 Roadmap (Future Enhancements)
- [ ] **Fine-tuning**: Custom dataset with 500+ regional dishes
- [ ] **Multi-food segmentation**: YOLO-based instance segmentation
- [ ] **Portion estimation**: Reference object detection (coin, card, plate)
- [ ] **Barcode scanning**: Integrate OpenFoodFacts API
- [ ] **User accounts**: Firebase authentication + cloud storage
- [ ] **Mobile app**: React Native wrapper
- [ ] **Active learning**: Weekly model retraining from user corrections
- [ ] **Recommendations**: Suggest healthier alternatives

---

## ⚠️ Known Limitations

- Food-101 dataset does not cover all real-world or regional dishes
- Portion size is user-entered (not estimated from the image)
- Assumes one primary food item per image
- Nutrition accuracy depends on nutrition.csv quality

---

## 🚀 Future Enhancements

- Fine-tuning the model with regional and custom food datasets
- Multi-food detection within a single image
- Portion estimation using food segmentation
- Persistent database storage (SQLite / Firebase)
- Mobile-first UI optimization
