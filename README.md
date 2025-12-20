# 🚗 UK Road Accident Analysis & Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Completed-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Project Overview | معرفی پروژه
این پروژه یک سیستم جامع تحلیل داده و هوش مصنوعی است که بر روی **داده‌های تصادفات جاده‌ای بریتانیا** انجام شده است. هدف ما فراتر از تحلیل آماری ساده بوده و شامل شناسایی نقاط حادثه‌خیز (Hotspots)، محاسبه شاخص خطر (Risk Index) و **پیش‌بینی شدت تصادفات** با استفاده از الگوریتم‌های یادگیری ماشین است.

پروژه در ۴ فاز اصلی پیاده‌سازی شده است که مستندات کامل هر بخش در پوشه `docs/` موجود است.

---

## 🚀 Key Features | ویژگی‌های کلیدی
* **Data Cleaning:** پاکسازی پیشرفته و فیلتر کردن مختصات جغرافیایی پرت.
* **Spatial Clustering (DBSCAN):** شناسایی کانون‌های خطر و حذف نویزها.
* **Density Estimation (KDE):** تخمین تراکم وقوع تصادفات در سطح نقشه.
* **Risk Index Scoring:** محاسبه نمره ریسک اختصاصی برای هر نقطه (ترکیبی از شدت، تراکم و محیط).
* **AI Prediction (XGBoost):** پیش‌بینی شدت تصادف (سطحی/جدی/کشنده) با دقت بالا.

---

## 📂 Project Structure | ساختار فایل‌ها

```text
accident-analysis/
│
├── data/                    # داده‌های خام و پردازش شده
├── notebooks/               # کدهای پروژه (ژوپیتر نوت‌بوک‌ها)
│   ├── preprocess.py                 # اسکریپت پاکسازی
│   ├── spatial_analysis.ipynb     # تحلیل مکانی
│   ├── risk_index.ipynb           # شاخص خطر
│   ├── prediction.ipynb           # مدل پیش‌بینی
│   └── risk_map.html                 # خروجی نقشه
│
├── docs/                    # 📄 مستندات کامل پروژه
│   ├── Preprocessing.md           # فاز ۱: پاکسازی
│   ├── Spatial_Analysis.md        # فاز ۲: تحلیل مکانی
│   ├── Risk_Index.md              # فاز ۳: شاخص خطر
│   └── Prediction.md        # فاز ۴: پیش‌بینی هوشمند
│
├── requirements.txt         # کتابخانه‌های مورد نیاز
└── README.md                # همین فایل
