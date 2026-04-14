# ❤️ Heart Disease Prediction App

An AI-powered web application that predicts the likelihood of heart disease based on clinical parameters using a Machine Learning model.

🔗 **Live App:** https://piyushgupta-analyst-heart-disease-prediction-app-app-9jcqrr.streamlit.app/

---

## 🚀 Features

- 🧠 Machine Learning-based prediction (KNN model)
- 📊 Real-time risk probability analysis
- 📈 Dynamic visualization of key risk factors
- 🩺 Actionable health recommendations
- ⚡ Fast and interactive UI built with Streamlit
- 🛡️ Input validation and robust preprocessing

---

## 🧠 Model Details

- Algorithm: K-Nearest Neighbors (KNN)
- Preprocessing:
  - Feature scaling (StandardScaler)
  - One-hot encoding for categorical features
- Custom probability handling for better interpretability

---

## 📊 Input Parameters

The model takes the following medical attributes:
- Age
- Sex(gender)
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol Level
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- ST Depression (Oldpeak)
- ST Slope

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **ML Libraries:** Scikit-learn, NumPy, Pandas  
- **Model Persistence:** Joblib  

---

## ⚙️ Installation & Run Locally

```bash
git clone https://github.com/PiyushGupta-Analyst/heart-disease-prediction-app.git
cd heart-disease-prediction-app
pip install -r requirements.txt
streamlit run app.py
