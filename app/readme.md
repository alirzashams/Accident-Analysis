# 🚗 Road Accident Severity Prediction System (Web App)
### سامانه هوشمند پیش‌بینی شدت تصادفات جاده‌ای

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Status](https://img.shields.io/badge/Status-Deployment%20Ready-success)

## 📌 Overview (معرفی)
This web application is the deployment interface for the **Road Accident Analysis** project. It leverages trained Machine Learning models (**LightGBM** and **XGBoost**) to predict the severity of traffic accidents based on user-provided environmental and temporal inputs.

این وب‌اپلیکیشن رابط کاربری نهایی برای پروژه **تحلیل تصادفات جاده‌ای** است. این سامانه با استفاده از مدل‌های آموزش‌دیده یادگیری ماشین (**LightGBM** و **XGBoost**)، شدت احتمالی تصادف را بر اساس شرایط محیطی و زمانی ورودی پیش‌بینی می‌کند.

---

## ✨ Features (ویژگی‌ها)
- **🌍 Bilingual Interface:** Fully supports **Persian (RTL)** and **English (LTR)** with dynamic switching.
- **🧠 Multi-Model Analysis:**
  - Primary Model: **LightGBM** (High efficiency & accuracy).
  - Comparative Model: **XGBoost** (For result validation).
- **📊 Interactive Visualizations:** Dynamic probability charts using **Plotly**.
- **💡 Smart Safety Recommendations:** AI-driven suggestions (e.g., speed reduction, lighting improvements) to mitigate fatal risks.
- **⚡ Real-time Prediction:** Instant results based on live inputs.

- **🌍 رابط کاربری دو زبانه:** پشتیبانی کامل از **فارسی (راست‌چین)** و **انگلیسی (چپ‌چین)**.
- **🧠 تحلیل چند مدلی:** استفاده همزمان از LightGBM (مدل اصلی) و XGBoost (برای مقایسه).
- **📊 نمودارهای تعاملی:** نمایش احتمال وقوع هر کلاس با نمودارهای Plotly.
- **💡 سیستم پیشنهاددهنده:** ارائه راهکارهای هوشمند (مانند کاهش سرعت یا اصلاح روشنایی) برای کاهش ریسک مرگبار.

---

## 🚀 Installation & Setup (نصب و اجرا)

### 1. Prerequisites (پیش‌نیازها)
Make sure you have Python installed. Then, install the required libraries:
ابتدا مطمئن شوید پایتون نصب است. سپس کتابخانه‌های مورد نیاز را نصب کنید:

```bash
pip install -r requirements.txt