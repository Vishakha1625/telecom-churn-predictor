# 📡 TeleChurn AI — Streamlit App

A professional, dark-themed Streamlit application for **Telecom Customer Churn Prediction**.

---

## 🗂️ Project Structure

```
telecom_churn_app/
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
└── models/                 ← Place your pkl files here
    ├── model.pkl
    ├── scaler.pkl
    ├── label_encoders.pkl
    ├── feature_columns.pkl
    └── model_info.pkl
```

---

## ⚙️ Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your model files
Run the notebook (`Chustomer_churn__2_.ipynb`) and copy the pkl files into a `models/` folder next to `app.py`:

```bash
mkdir models
# Copy pkl files output by the notebook:
cp model.pkl            models/
cp scaler.pkl           models/
cp label_encoders.pkl   models/
cp feature_columns.pkl  models/
cp model_info.pkl       models/
```

### 3. Run the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 🧭 Pages

| Page | Description |
|------|-------------|
|  Dashboard | KPI cards, churn distribution, key business insights |
|  Predict Churn | Enter customer details → get churn probability + risk factors |
|  Analytics | EDA charts: demographics, services, billing patterns |
|  Model Info | Model metrics, baseline vs tuned comparison, feature pipeline |

---

## 🔄 Preprocessing Pipeline (matches notebook exactly)

1. Collect raw input (same feature names as training data)
2. Apply `LabelEncoder.transform()` for all categorical columns
3. Reorder columns using `feature_columns.pkl`
4. Apply `StandardScaler.transform()` using `scaler.pkl`
5. `model.predict_proba()` → churn probability
6. Compare against optimal threshold from `model_info.pkl`

---

## 📦 Tech Stack

- **Streamlit** — UI framework
- **Plotly** — Interactive charts
- **Scikit-learn / XGBoost** — ML models
- **Pandas / NumPy** — Data processing
- **Pickle** — Model serialisation