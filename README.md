# 🧠 Symptom-Based Disease Prediction System (ML + Full Stack + Docker)

## 📌 Project Overview

This project is an end-to-end Machine Learning web application that predicts possible diseases based on user-provided symptoms.

It integrates:
- Machine Learning models for classification
- A FastAPI backend for serving predictions
- A React frontend for user interaction
- Docker & Docker Compose for deployment
- MLflow for experiment tracking and model management

The main objective is to provide an intelligent system that can assist in early disease prediction using symptoms analysis.

---

## ⚙️ Tech Stack

### 🎨 Frontend
- React.js
- Axios
- HTML / CSS

### ⚙️ Backend
- FastAPI
- Python
- Pydantic

### 🧠 Machine Learning
- Scikit-learn

#### Models Used:
- Decision Tree Classifier 🌳
- K-Nearest Neighbors (KNN) 📍
- Random Forest Classifier 🌲
- GridSearchCV (Hyperparameter tuning) 🔍

### 📊 MLOps
- MLflow (experiment tracking and model versioning)

### 📦 Dataset
- Kaggle Symptom-based Disease Dataset

### 🐳 DevOps
- Docker
- Docker Compose

---

## 🧠 Machine Learning Pipeline

1. Data collection from Kaggle
2. Data preprocessing (symptom encoding)
3. Model training using multiple classifiers
4. Hyperparameter tuning using GridSearchCV
5. Model evaluation and comparison
6. Best model selection
7. Experiment tracking with MLflow
8. Deployment via FastAPI

---

## 🏗️ System Architecture

Frontend (React)
        ↓
FastAPI Backend (/predict)
        ↓
Machine Learning Model (Scikit-learn)
        ↓
Prediction Result
        ↓
Displayed on UI

---

## 🚀 How to Run the Project (Docker)

### 🔧 Prerequisites
- Docker
- Docker Compose

---

### 📥 Step 1: Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
