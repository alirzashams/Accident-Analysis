
import streamlit as st
import pandas as pd
import xgboost as xgb

st.title("پیش‌بینی شدت تصادفات رانندگی")

# ساخت سایدبار برای دریافت ورودی‌ها
st.sidebar.header("ورودی‌های مدل")

def user_input_features():
    day_of_week = st.sidebar.selectbox("روز هفته", (0, 1, 2, 3, 4, 5, 6))
    road_type = st.sidebar.selectbox("نوع جاده", (0, 1, 2, 3))
    speed_limit = st.sidebar.slider("محدودیت سرعت", 10, 120, 50)
    light_conditions = st.sidebar.selectbox("وضعیت نور", (0, 1, 2, 3))
    weather = st.sidebar.selectbox("وضعیت آب و هوا", (0, 1, 2, 3, 4))
    hour = st.sidebar.slider("ساعت", 0, 23, 12)
    
    data = {
        'Day_of_Week': day_of_week,
        'Road_Type': road_type,
        'Speed_limit': speed_limit,
        'Light_Conditions': light_conditions,
        'Weather_Conditions': weather,
        'Hour': hour,
        # سایر ویژگی‌هایی که در مدل استفاده کردید را اینجا اضافه کنید
    }
    return pd.DataFrame(data, index=[0])

df_input = user_input_features()

st.subheader("پارامترهای ورودی انتخاب شده:")
st.write(df_input)

# بارگذاری مدل (باید قبلاً مدل را به صورت فایل ذخیره کرده باشید)
# model = joblib.load('accident_model.pkl')

st.subheader("نتیجه پیش‌بینی:")
# prediction = model.predict(df_input)
# st.write(f"شدت تصادف پیش‌بینی شده: {prediction}")
st.info("نکته: برای نمایش نتیجه، ابتدا مدل خود را آموزش داده و ذخیره کنید.")