import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import os

# ==========================================
# 1. پیکربندی صفحه و استایل‌های آکادمیک
# ==========================================
st.set_page_config(
    page_title="Accident Risk AI | Kharazmi Uni", 
    page_icon="🚦", 
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# دیکشنری متون برای پشتیبانی حرفه‌ای از دو زبان
TEXTS = {
    'fa': {
        'dir': 'rtl', 'font': 'Tahoma',
        'header': "سامانه هوشمند پیش‌بینی شدت تصادفات جاده‌ای",
        'subheader': "پروژه پایانی کارشناسی | تحلیل مبتنی بر یادگیری ماشین",
        'sb_title': "پارامترهای محیطی و ترافیکی",
        'lbl_month': "ماه وقوع حادثه",
        'lbl_day': "روز هفته",
        'btn_predict': "اجرای مدل و تحلیل ریسک",
        'res_head': "توزیع احتمالات خروجی مدل",
        'metrics_title': "شاخص‌های اطمینان",
        'risk_levels': ['مرگبار (Fatal)', 'جدی (Serious)', 'سطحی (Slight)'],
        'footer': "طراحی و توسعه: علیرضا شمس | دانشگاه خوارزمی",
        'days': {1:"دوشنبه", 2:"سه‌شنبه", 3:"چهارشنبه", 4:"پنج‌شنبه", 5:"جمعه", 6:"شنبه", 7:"یکشنبه"},
        'months': {1:"ژانویه", 2:"فوریه", 3:"مارس", 4:"آوریل", 5:"مه", 6:"ژوئن", 7:"ژوئیه", 8:"اوت", 9:"سپتامبر", 10:"اکتبر", 11:"نوامبر", 12:"دسامبر"}
    },
    'en': {
        'dir': 'ltr', 'font': 'sans-serif',
        'header': "Road Accident Severity Prediction AI",
        'subheader': "B.Sc. Final Project | Machine Learning Analysis",
        'sb_title': "Environmental & Traffic Parameters",
        'lbl_month': "Month",
        'lbl_day': "Day of Week",
        'btn_predict': "Run Model & Analyze Risk",
        'res_head': "Model Probability Distribution",
        'metrics_title': "Confidence Metrics",
        'risk_levels': ['Fatal', 'Serious', 'Slight'],
        'footer': "Developed by: Alireza Shams | Kharazmi University",
        'days': {1:"Monday", 2:"Tuesday", 3:"Wednesday", 4:"Thursday", 5:"Friday", 6:"Saturday", 7:"Sunday"},
        'months': {1:"January", 2:"February", 3:"March", 4:"April", 5:"May", 6:"June", 7:"July", 8:"August", 9:"September", 10:"October", 11:"November", 12:"December"}
    }
}

lang_opt = st.sidebar.radio("🌐 Language / زبان سیستم", ['فارسی', 'English'], horizontal=True)
L = 'fa' if lang_opt == 'فارسی' else 'en'
T = TEXTS[L]

# اعمال استایل راست‌چین/چپ‌چین به صورت داینامیک
st.markdown(f"""
<style>
    .main {{ direction: {T['dir']}; font-family: {T['font']}, sans-serif; }}
    h1, h2, h3, p, label {{ text-align: {'right' if L == 'fa' else 'left'} !important; }}
    .stButton>button {{ background-color: #2c3e50; color: white; border-radius: 8px; height: 50px; font-weight: bold; font-size: 16px; }}
    .stButton>button:hover {{ background-color: #34495e; color: #f1c40f; }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. بارگذاری موتور هوش مصنوعی (مدل‌ها)
# ==========================================
@st.cache_resource
def load_ai_engine():
    try:
        with open(os.path.join(BASE_DIR, 'models_dict.pkl'), 'rb') as f:
            models = pickle.load(f)
        with open(os.path.join(BASE_DIR, 'target_encoder.pkl'), 'rb') as f:
            encoder = pickle.load(f)
        with open(os.path.join(BASE_DIR, 'features_list.pkl'), 'rb') as f:
            features = pickle.load(f)
        return models, encoder, features
    except Exception as e:
        return None, None, None

models_dict, target_encoder, features_list = load_ai_engine()

if models_dict is None:
    st.error("❌ سیستم قادر به بارگذاری هسته هوش مصنوعی نیست. فایل‌های pkl را بررسی کنید.")
    st.stop()

# ==========================================
# 3. پنل تنظیمات ورودی (Sidebar)
# ==========================================
st.sidebar.header(T['sb_title'])
selected_model = st.sidebar.selectbox("🧠 الگوریتم پردازشی:", list(models_dict.keys()))
st.sidebar.markdown("---")

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    speed = st.slider("سرعت (mph)", 10, 70, 30, step=10)
    hour = st.slider("ساعت (0-23)", 0, 23, 14)
    # اضافه شدن نام ماه‌ها
    month = st.selectbox(T['lbl_month'], range(1, 13), index=5, format_func=lambda x: f"{T['months'][x]} ({x})")
    
with col_sb2:
    # اضافه شدن نام روزهای هفته
    day = st.selectbox(T['lbl_day'], range(1, 8), format_func=lambda x: T['days'][x])
    area = st.radio("بافت منطقه", [1, 2], format_func=lambda x: "شهری" if x==1 else "روستایی", horizontal=True)

light = st.sidebar.selectbox("وضعیت روشنایی", [1, 2, 3], format_func=lambda x: {1:"روز (روشن)", 2:"شب (با چراغ)", 3:"شب (تاریک مطلق)"}[x])
weather = st.sidebar.selectbox("شرایط جوی", [1, 2, 3, 4], format_func=lambda x: {1:"صاف", 2:"بارانی", 3:"برفی", 4:"مه‌آلود"}[x])
road_surface = st.sidebar.selectbox("وضعیت سطح جاده", [1, 2, 3, 4, 5], format_func=lambda x: {1:"خشک", 2:"خیس", 3:"برف", 4:"یخ‌زده", 5:"آب‌گرفتگی"}[x])

# ==========================================
# 4. هسته پردازشی و پیش‌بینی
# ==========================================
st.title(T['header'])
st.caption(T['subheader'])
st.divider()

if st.button(T['btn_predict'], use_container_width=True):
    
    # 💡 مهندسی ویژگی‌ها (Feature Engineering)
    is_weekend = 1 if day in [6, 7] else 0  
    speed_light_inter = speed * light       
    
    # شبیه‌سازی مختصات برای جلوگیری از بایاس مکانی مدل
    rand_lat = np.random.uniform(51.0, 54.0)
    rand_lon = np.random.uniform(-2.0, 1.0)
    
    # تجمیع داده‌ها در قالب استاندارد
    raw_data = {
        'Latitude': rand_lat,
        'Longitude': rand_lon,
        'Speed_limit': speed,
        'Light_Conditions': light,
        'Weather_Conditions': weather,
        'Road_Surface_Conditions': road_surface,
        'Urban_or_Rural_Area': area,
        'Hour': hour,
        'Month': month,
        'DayOfWeek': day,
        'IsWeekend': is_weekend,
        'Speed_Light_Inter': speed_light_inter
    }

    # ساخت دیتافریم و تبدیل اجباری به Float برای امنیت اجرای درخت تصمیم
    input_data = pd.DataFrame([raw_data], columns=features_list).astype(float)

    # اجرای مدل انتخابی
    model = models_dict[selected_model]
    probs = model.predict_proba(input_data)[0]
    
    # استخراج احتمالات بر اساس ایندکس ثابت مدل شما (0:Fatal, 1:Serious, 2:Slight)
    p_fatal = probs[0]
    p_serious = probs[1]
    p_slight = probs[2]

    # ==========================================
    # 5. منطق ارزیابی ریسک (Thresholding)
    # ==========================================
    # حل مشکل Imbalanced Data با تعریف آستانه حساسیت
    if p_fatal > 0.04:         # آستانه 4 درصد برای تصادف مرگبار
        severity_label = 'Fatal'
        pred_val = p_fatal
        alert_color = "red"
    elif p_serious > 0.15:     # آستانه 15 درصد برای جراحت جدی
        severity_label = 'Serious'
        pred_val = p_serious
        alert_color = "orange"
    else:                      # شرایط نرمال
        severity_label = 'Slight'
        pred_val = p_slight
        alert_color = "green"

    # ==========================================
    # 6. نمایش گرافیکی نتایج (داشبورد)
    # ==========================================
    col_chart, col_info = st.columns([2, 1])
    
    with col_chart:
        st.markdown(f"### 📊 {T['res_head']}")
        
        # رسم نمودار میله‌ای با Plotly
        fig = go.Figure(go.Bar(
            x=T['risk_levels'], 
            y=[p_fatal, p_serious, p_slight],
            marker_color=['#e74c3c', '#f1c40f', '#2ecc71'],
            text=[f"{p_fatal*100:.2f}%", f"{p_serious*100:.2f}%", f"{p_slight*100:.2f}%"], 
            textposition='auto',
            hovertemplate="<b>%{x}</b><br>Probability: %{y:.2%}<extra></extra>"
        ))
        
        fig.update_layout(
            height=380, 
            margin=dict(t=30, b=30, l=0, r=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(title="Probability", gridcolor='rgba(128,128,128,0.2)'),
            xaxis=dict(title="Severity Class")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_info:
        st.markdown(f"### 🎯 نتیجه نهایی")
        
        if severity_label == 'Fatal':
            st.error("🚨 هشدار: ریسک مرگبار (Fatal)")
            st.caption("سیستم هوشمند به دلیل عبور احتمال فوت از مرز بحرانی (۴٪)، این شرایط را به شدت پرخطر طبقه‌بندی کرده است.")
        elif severity_label == 'Serious':
            st.warning("⚠️ هشدار: جراحت جدی (Serious)")
            st.caption("احتمال بالای جراحات جدی. نیاز به اقدامات پیشگیرانه در این شرایط محیطی احساس می‌شود.")
        else:
            st.success("✅ وضعیت: کم‌خطر (Slight)")
            st.caption("بر اساس الگوهای ترافیکی، این شرایط در دسته تصادفات خسارتی و سطحی قرار می‌گیرد.")
            
        st.divider()
        st.markdown(f"**{T['metrics_title']}:**")
        st.metric(label="الگوریتم فعال", value=selected_model)
        
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: #7f8c8d; direction: {T['dir']};'>{T['footer']}</div>", unsafe_allow_html=True)