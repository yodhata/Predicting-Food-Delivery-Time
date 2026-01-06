# Predicting Food Delivery Time (Machine Learning)

This repository contains an end-to-end machine learning project to **predict food delivery time (in minutes)** and translate the results into simple operational insights (dispatching, routing, and staffing).

## Repository Structure

- **`streamlit_app.py`** — Streamlit web app for interactive delivery time prediction  
- **`delivery_lr_artifact.pkl`** — Saved model artifact (Linear Regression + feature list, and optional preprocessing objects)  
- **`Food_Delivery_Times.csv`** — Dataset used for training and evaluation  
- **`1_Take_Home_Test_DS___Yodha_Pranata (1).ipynb`** — Full notebook: EDA → preprocessing → modeling → evaluation → insights  
- **`requirements.txt`** — Python dependencies  
- **`.DS_Store`** — macOS system file (recommended to ignore/remove from git)

---

## Project Goal

### Objective
Build and evaluate a model that predicts **Delivery_Time_min**, and provide clear insights about the key drivers of delay.

### Why it matters
More accurate predictions help:
- reduce customer dissatisfaction from late deliveries,
- improve courier utilization and dispatch decisions,
- reduce operational “noise” (reassignment, support tickets, complaints).

---

## Dataset

- File: `Food_Delivery_Times.csv`
- Target: `Delivery_Time_min`
- Main features:
  - `Distance_km`
  - `Preparation_Time_min`
  - `Courier_Experience_yrs`
  - `Traffic_Level` (Low/Medium/High)
  - `Weather` (Clear/Rainy/Foggy/Snowy/Windy)

---

## Modeling Summary

Multiple models were tested (e.g., Linear Regression and other regressors).  
In this project, **Linear Regression** was selected as the final model because it provided strong baseline performance and was easy to interpret.

Key evaluation notes:
- Performance is generally better for **short and normal delivery times**
- Error tends to increase for **very long delivery times** (higher variance / harder cases)

---

## Saved Model Artifact

The app loads `delivery_lr_artifact.pkl`.  
The artifact is used to ensure Streamlit input is processed in the **same format** as training (feature order + encoding + scaling, if included).

Typical keys inside the artifact:
- `model` — trained regression model
- `features` — list of final feature columns used in training (order matters)
- `scaler` (optional) — training scaler if numeric features were scaled
- `scale_cols` (optional) — numeric columns that must be scaled
- `iqr` (optional) — thresholds for segmenting predictions (fast/normal/long)

---

## How to Run Locally

### 1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
