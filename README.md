🎓 Student Performance Prediction Web App

This is a complete end-to-end Machine Learning project where a student's performance in mathematics is predicted based on various demographic and academic features. The model is deployed using **Railway** and accessible via a responsive Flask-based web interface.

---

🌐 Live Demo

🚀 Experience the live application here:  
👉 [**Student Performance Predictor Web App**](http://student-performance-app-production.up.railway.app)


📌 Project Overview

The main objective of this project is to build a regression model that predicts a student’s **math score** using various independent variables such as gender, ethnicity, parental education level, and other academic indicators.

This project was developed as part of my learning journey in data science and machine learning, focusing on:
- End-to-end ML model development
- Clean modular project structure (following MLOps practices)
- Scalable deployment using cloud platforms (Railway)
- Simple yet clean user interface using Flask, HTML, and CSS


🎯 Problem Statement

Education systems often seek to identify early factors that influence academic performance. This model aims to help educators, analysts, and institutions forecast student performance and better allocate support based on predicted needs.



🔍 Features Used for Prediction

- **Gender**
- **Race/Ethnicity**
- **Parental level of education**
- **Lunch status (standard or free/reduced)**
- **Test preparation course (completed or not)**
- **Reading score**
- **Writing score**

These inputs are collected through a web form and passed to a pre-trained regression model which returns the predicted math score.



📦 Tech Stack

| Layer             | Tools Used                                 |
|------------------|---------------------------------------------|
| Language          | Python 3                                    |
| Backend Framework | Flask                                       |
| Machine Learning  | scikit-learn, Pandas, NumPy, joblib         |
| Model Selection   | Linear Regression, Random Forest, XGBoost, etc. |
| Deployment        | Railway (PaaS, free-tier supported)         |
| Frontend          | HTML, CSS, Jinja2 templating                |
| Version Control   | Git & GitHub                                |