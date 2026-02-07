
<div align="center">

# Architecting a Data-Driven Framework for Road Safety
### Geospatial Hotspot Analysis and Predictive Severity Modeling

![Python](https://img.shields.io/badge/Python-3.8%2B-007ACC?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

</div>

---

## Abstract

> **English**
>
> This repository contains the implementation of a B.Sc. dissertation titled *"Architecting a Data-Driven Framework for Road Safety"*. Traditional traffic analysis often relies on retrospective statistics. This project introduces a hybrid proactive system that integrates **Density-Based Spatial Clustering (DBSCAN)** with **Gradient Boosting Decision Trees (LightGBM & XGBoost)**. The primary objective is to engineer a robust mechanism for identifying high-density accident zones and predicting accident severity based on real-time environmental and temporal variables.

> **فارسی**
>
> این مخزن شامل پیاده‌سازی پروژه پایانی کارشناسی با عنوان *"طراحی چارچوب داده‌محور برای ایمنی راه‌ها"* است. تحلیل‌های ترافیکی سنتی اغلب مبتنی بر آمارهای گذشته‌نگر هستند. این پژوهش یک سیستم ترکیبی پیش‌بینانه را معرفی می‌کند که **خوشه‌بندی مکانی مبتنی بر چگالی (DBSCAN)** را با **درخت‌های تصمیم گرادیان بوستینگ (LightGBM & XGBoost)** ادغام می‌نماید. هدف اصلی، مهندسی یک مکانیزم مقاوم برای شناسایی کانون‌های پرخطر و پیش‌بینی شدت تصادفات بر اساس متغیرهای محیطی و زمانی است.

---

## Methodology and Technical Approach

The research framework is delineated into four subsequent technical phases:

### 1. Advanced Data Engineering
Implementation of a rigorous preprocessing pipeline to sanitize over 25,000 raw GPS records.
* **Techniques:** Outlier detection via Bounding Box filtering, Temporal decomposition.
* **Outcome:** A cleansed dataset ensuring high spatial integrity.

### 2. Geospatial Intelligence
Utilization of unsupervised learning to detect non-linear accident hotspots.
* **Algorithm:** DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
* **Density Estimation:** Kernel Density Estimation (KDE) for continuous risk scoring.

### 3. Risk Index Formulation
Development of a weighted "Risk Index" (PRI) to quantify the lethality of road segments.
* **Formula:** A composite metric weighing fatal incidents significantly higher than slight ones, integrated with environmental penalties.

### 4. Predictive Modeling
Training ensemble learning models to classify accident severity (Slight, Serious, Fatal).
* **Models:** LightGBM (Primary), XGBoost (Comparative).
* **Handling Imbalance:** Stratified sampling and class-weighted loss functions.

---

## Repository Architecture

The project is structured to separate concerns between data, documentation, research notebooks, and deployment code.

```text
accident-analysis/
│
├── app/                       # Deployment Module (Web App)
│   ├── app.py                 # Streamlit Dashboard Interface
│   ├── lgb_model.pkl          # Serialized LightGBM Model
│   └── xgb_model.pkl          # Serialized XGBoost Model
│
├── data/                      # Data Management
│   ├── cleaned_accident_data.csv
│   └── readme.md              # Data Dictionary
│
├── docs/                      # Technical Documentation
│   ├── Spatial_Analysis.md    # Clustering Logic & Hyperparameters
│   ├── Risk_Index.md          # Mathematical Formulation of Risk
│   └── Prediction.md          # Model Evaluation Metrics
│
├── notebooks/                 # Research & Development (Lab)
│   ├── 1_preprocess.ipynb     # Data Cleaning Pipeline
│   ├── 2_spatial_analysis.ipynb # DBSCAN Implementation
│   ├── 3_risk_index.ipynb     # Risk Calculation Logic
│   └── 4_prediction.ipynb     # Model Training & Validation
│
├── requirements.txt           # Dependency Manifest
└── README.md                  # Project Overview

```

---

## Execution Guide

Follow these steps to set up the environment and launch the application.

### 1. Environment Setup

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/alirzashams/accident-analysis.git](https://github.com/alirzashams/accident-analysis.git)
cd accident-analysis
pip install -r requirements.txt

```

### 2. Deployment

To launch the interactive analytical dashboard:

```bash
cd app
streamlit run app.py

```

### 3. Reproducibility

The Jupyter notebooks located in the `notebooks/` directory correspond to the sequential phases of the research. They can be executed sequentially to reproduce the data cleaning, clustering, and training processes.

---

## Experimental Results

* **Model Performance:** The LightGBM classifier achieved a **Macro F1-Score of 0.76**, demonstrating robust capability in detecting minority classes (Fatal accidents).
* **Feature Importance:** Analysis reveals that **Speed Limit** and **Spatial Cluster ID** are the most significant determinants of accident severity.

---

<div align="center">

### Credits and Affiliation

**Author:** Alireza Shams | **Supervisor:** Dr. Keyvan Borna
Faculty of Mathematics and Computer Science, Kharazmi University
Fall 2025

</div>

```

```

