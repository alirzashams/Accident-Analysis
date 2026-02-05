import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import xgboost as xgb

# تنظیمات اولیه صفحه
st.set_page_config(page_title="پیش‌بینی شدت تصادفات (مقایسه مدل‌ها)", layout="wide")

st.title("🚗 سامانه هوشمند پیش‌بینی تصادفات (مقایسه LightGBM و XGBoost)")
st.markdown("این برنامه پیش‌بینی شدت تصادف را با استفاده از دو مدل قدرتمند انجام می‌دهد تا بتوانید نتایج را مقایسه کنید.")

# ۱. بارگذاری مدل‌ها
@st.cache_resource
def load_models():
    models = {}
    try:
        with open('lgb_model.pkl', 'rb') as f:
            models['LightGBM'] = pickle.load(f)
    except FileNotFoundError:
        models['LightGBM'] = None
        
    try:
        with open('xgb_model.pkl', 'rb') as f:
            models['XGBoost'] = pickle.load(f)
    except FileNotFoundError:
        models['XGBoost'] = None
        
    return models

loaded_models = load_models()

# بررسی وضعیت مدل‌ها
if loaded_models['LightGBM'] is None and loaded_models['XGBoost'] is None:
    st.error("❌ هیچ فایل مدلی یافت نشد! لطفاً نوت‌بوک را اجرا کنید تا فایل‌های lgb_model.pkl و xgb_model.pkl ساخته شوند.")
else:
    if loaded_models['LightGBM'] is None:
        st.warning("⚠️ مدل LightGBM یافت نشد.")
    if loaded_models['XGBoost'] is None:
        st.warning("⚠️ مدل XGBoost یافت نشد.")

    # ۲. ورودی‌ها در سایدبار
    st.sidebar.header("📋 مشخصات تصادف")

    def get_user_inputs():
        day = st.sidebar.selectbox("روز هفته", options=[0,1,2,3,4,5,6], 
                                   format_func=lambda x: ["شنبه","یکشنبه","دوشنبه","سه‌شنبه","چهارشنبه","پنج‌شنبه","جمعه"][x])
        road_type = st.sidebar.selectbox("نوع جاده", [0, 1, 2, 3])
        speed = st.sidebar.slider("محدودیت سرعت", 10, 120, 50)
        light = st.sidebar.selectbox("وضعیت نور", [0, 1, 2, 3])
        weather = st.sidebar.selectbox("وضعیت آب و هوا", [0, 1, 2, 3])
        area = st.sidebar.radio("نوع منطقه", [0, 1], format_func=lambda x: "شهری" if x==1 else "روستایی")
        hour = st.sidebar.slider("ساعت وقوع", 0, 23, 12)
        
        # مقادیر پیش‌فرض برای ستون‌هایی که ممکن است در مدل باشند
        cluster = -1 
        density = 1.0

        features_dict = {
            'Day_of_Week': day,
            'Road_Type': road_type,
            'Speed_limit': speed,
            'Light_Conditions': light,
            'Weather_Conditions': weather,
            'Urban_or_Rural_Area': area,
            'Cluster': cluster,
            'Density': density,
            'Hour': hour
        }
        return pd.DataFrame([features_dict])

    input_df = get_user_inputs()

    st.subheader("📍 ورودی‌های شما:")
    st.dataframe(input_df)
    st.markdown("---")

    # ۳. دکمه پیش‌بینی و نمایش مقایسه‌ای
    if st.button("🚀 اجرای پیش‌بینی با هر دو مدل"):
        
        # دیکشنری تبدیل اعداد به متن
        severity_map = {0: 'خفیف (Slight)', 1: 'شدید (Serious)', 2: 'مرگبار (Fatal)'}

        col1, col2 = st.columns(2)

        # --- نمایش نتیجه LightGBM ---
        with col1:
            st.info("### ⚡ مدل LightGBM")
            if loaded_models['LightGBM']:
                pred_lgb = loaded_models['LightGBM'].predict(input_df)[0]
                text_lgb = severity_map.get(pred_lgb, "نامشخص")
                
                if pred_lgb == 0:
                    st.success(f"**{text_lgb}**")
                elif pred_lgb == 1:
                    st.warning(f"**{text_lgb}**")
                else:
                    st.error(f"**{text_lgb}**")
            else:
                st.write("مدل موجود نیست")

        # --- نمایش نتیجه XGBoost ---
        with col2:
            st.info("### 🌲 مدل XGBoost")
            if loaded_models['XGBoost']:
                pred_xgb = loaded_models['XGBoost'].predict(input_df)[0]
                text_xgb = severity_map.get(pred_xgb, "نامشخص")
                
                if pred_xgb == 0:
                    st.success(f"**{text_xgb}**")
                elif pred_xgb == 1:
                    st.warning(f"**{text_xgb}**")
                else:
                    st.error(f"**{text_xgb}**")
            else:
                st.write("مدل موجود نیست")