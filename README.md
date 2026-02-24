# 🏠 California Housing Price Prediction

An end-to-end Machine Learning web application that predicts median house prices using housing features.

This project demonstrates a production-ready ML system with model training, preprocessing pipeline, API integration, Docker containerization, and cloud deployment.

---

## LIVE DEMO

🔗 https://house-price-ml-app-jefg.onrender.com/

---

## THE PROBLEM STATEMENT

Predict the median house value in California districts using demographic and housing data.

---

## SYSTEM ARCHITECTURE

User → Streamlit UI → FastAPI Backend → ML Pipeline → Prediction

- Streamlit: User interface
- FastAPI: Backend API
- Scikit-learn Pipeline: Data preprocessing + model
- Docker: Containerization
- Render: Cloud deployment

---

##  FEATURES USED 

- Stratified Train-Test Split
- ColumnTransformer Pipeline
- Random Forest Regressor
- Modular Prediction Pipeline
- REST API using FastAPI
- Streamlit Web Interface
- Dockerized Deployment
- Cloud Hosted on Render

---

##  TECH STACK 

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Streamlit
- Docker
- Render

---

## PROJECT STRUCTURE
house-price-ml-app/
│
├── backend/
│ ├── fast_app.py
│ └── artifacts/
│ ├── model.pkl
│ └── pipeline.pkl
│
├── frontend/
│ └── streamlit_app.py
│
├── src/
│ ├── train_pipeline.py
│ └── predict_pipeline.py
│
├── Dockerfile
├── requirements.txt
└── README.md
