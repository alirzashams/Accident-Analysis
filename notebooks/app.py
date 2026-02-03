import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lightgbm as lgb
import xgboost as xgb

# تنظیمات اولیه صفحه
st.set_page_config(page_title="پیش‌بینی شدت تصادفات", layout="wide")

st.title("🚗 سامانه هوشمند پیش‌بینی شدت تصادفات رانندگی")
st.markdown("این برنامه با استفاده از مدل‌های یادگیری ماشین، شدت احتمالی تصادف را پیش‌بینی می‌کند.")

# ۱. بارگذاری مدل آموزش‌دیده
@st.cache_resource
def load_trained_model():
    try:
        # فایل باید در کنار app.py باشد
        with open('accident_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model = load_trained_model()

if model is None:
    st.error("❌ فایل مدل (accident_model.pkl) یافت نشد. ابتدا نوت‌بوک را اجرا کنید تا فایل مدل ساخته شود.")
else:
    # ۲. طراحی رابط کاربری ورودی‌ها در سایدبار
    st.sidebar.header("📋 مشخصات تصادف را وارد کنید")

    def get_user_inputs():
        # دریافت ویژگی‌های زمانی و محیطی
        day = st.sidebar.selectbox("روز هفته", options=[0,1,2,3,4,5,6], 
                                   format_func=lambda x: ["شنبه","یکشنبه","دوشنبه","سه‌شنبه","چهارشنبه","پنج‌شنبه","جمعه"][x])
        road_type = st.sidebar.selectbox("نوع جاده (کد شده)", [0, 1, 2, 3])
        speed = st.sidebar.slider("محدودیت سرعت", 10, 120, 50)
        light = st.sidebar.selectbox("وضعیت نور (کد شده)", [0, 1, 2, 3])
        weather = st.sidebar.selectbox("وضعیت آب و هوا (کد شده)", [0, 1, 2, 3])
        area = st.sidebar.radio("نوع منطقه", [0, 1], format_func=lambda x: "شهری" if x==1 else "روستایی")
        hour = st.sidebar.slider("ساعت وقوع (۰-۲۳)", 0, 23, 12)
        
        # مقادیر ثابت برای ویژگی‌های خوشه‌بندی و تراکم (Cluster/Density)
        cluster = -1 
        density = 1.0

        # تشکیل دیتافریم با رعایت دقیق ترتیب ستون‌ها
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

    # ۳. نمایش اطلاعات ورودی به کاربر
    st.subheader("📍 مشخصات ورودی:")
    st.table(input_df)

    # ۴. دکمه پیش‌بینی و نمایش نتیجه
    if st.button("🚀 تحلیل و پیش‌بینی"):
        prediction = model.predict(input_df)
        
        # نگاشت اعداد به دسته‌های متنی
        severity_map = {0: 'خفیف (Slight)', 1: 'شدید (Serious)', 2: 'مرگبار (Fatal)'}
        result_text = severity_map.get(prediction[0], "نامشخص")
        
        # نمایش نتیجه با استایل‌های مختلف بر اساس شدت
        st.write("---")
        if prediction[0] == 0:
            st.success(f"### نتیجه پیش‌بینی: **{result_text}**")
        elif prediction[0] == 1:
            st.warning(f"### نتیجه پیش‌بینی: **{result_text}**")
        else:
            st.error(f"### نتیجه پیش‌بینی: **{result_text}**")

st.markdown("---")
st.caption("توسعه یافته توسط تیم تحلیل تصادفات | ۲۰۲۶")