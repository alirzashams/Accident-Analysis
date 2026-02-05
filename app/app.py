import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import os

# ==========================================
# 1. Project Config & Styles
# ==========================================
st.set_page_config(
    page_title="Road Safety Analysis | Kharazmi Uni",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Text Resources (Persian/English) matching the thesis context
TEXTS = {
    'fa': {
        'dir': 'rtl',
        'font': 'B Nazanin, Tahoma',
        'header_title': "تحلیل و پیش‌بینی شدت تصادفات جاده‌ای",
        'header_subtitle': "پروژه پایانی کارشناسی | دانشگاه خوارزمی",
        'project_desc': """
        <div style="text-align: justify;">
        در این پژوهش، با بهره‌گیری از الگوریتم‌های یادگیری ماشین <b>(LightGBM & XGBoost)</b> و تحلیل داده‌های کلان ترافیکی بریتانیا، 
        یک چارچوب هوشمند برای تخمین شدت سوانح رانندگی توسعه داده شده است. 
        این سامانه قادر است بر اساس پارامترهای محیطی و زمانی، سطح ریسک حادثه را پیش‌بینی نماید.
        </div>
        """,
        'sb_title': "تنظیمات پارامترهای ورودی",
        'lbl_day': "روز هفته",
        'lbl_road': "نوع جاده",
        'lbl_speed': "حد مجاز سرعت (مایل/ساعت)",
        'lbl_light': "وضعیت روشنایی",
        'lbl_weather': "شرایط جوی",
        'lbl_area': "موقعیت مکانی",
        'lbl_hour': "ساعت وقوع حادثه",
        'lbl_comp': "فعال‌سازی مدل مقایسه‌ای (XGBoost)",
        
        # Options matching LabelEncoder mapping in notebook
        'days': ['شنبه', 'یکشنبه', 'دوشنبه', 'سه‌شنبه', 'چهارشنبه', 'پنج‌شنبه', 'جمعه'],
        'roads': {'شهری/فرعی': 1, 'جاده دوطرفه/اصلی': 2, 'بزرگراه': 3, 'نامشخص': 0},
        'lights': {'روز (روشن)': 1, 'تاریک (با چراغ)': 2, 'تاریک (بدون چراغ)': 3, 'نامشخص': 0},
        'weathers': {'صاف/آفتابی': 1, 'بارانی': 2, 'برفی': 3, 'مه‌آلود': 4, 'نامشخص': 0},
        'areas': {'شهری': 1, 'روستایی': 2},
        
        'res_main': "نتایج مدل اصلی (LightGBM)",
        'res_comp': "نتایج مدل مقایسه‌ای (XGBoost)",
        'risk_levels': ['خسارتی/سطحی', 'جرحی/جدی', 'فوت/مرگبار'],
        'risk_msgs': [
            "وضعیت کم‌خطر (Slight)",
            "وضعیت پرخطر (Serious) - نیاز به احتیاط",
            "وضعیت بحرانی (Fatal) - ریسک بسیار بالا"
        ],
        'rec_head': "راهکارهای پیشنهادی جهت کاهش ریسک",
        'rec_speed': "📉 <b>تحلیل حساسیت:</b> کاهش سرعت خودرو به <b>{0}</b>، احتمال وقوع حادثه مرگبار را <b>{1:.1f}٪</b> کاهش می‌دهد.",
        'rec_light': "💡 <b>پیشنهاد زیرساختی:</b> تأمین روشنایی در این محور، ریسک فوت را تا <b>{0:.1f}٪</b> تقلیل می‌دهد.",
        'footer': "دانشجو: علیرضا شمس | استاد راهنما: دکتر کیوان برنا | پاییز ۱۴۰۴"
    },
    'en': {
        'dir': 'ltr',
        'font': 'sans-serif',
        'header_title': "Road Accident Severity Prediction",
        'header_subtitle': "B.Sc. Final Project | Kharazmi University",
        'project_desc': """
        This project leverages machine learning algorithms <b>(LightGBM & XGBoost)</b> to analyze traffic accident patterns. 
        The system predicts accident severity based on environmental and temporal features extracted from the UK road safety dataset.
        """,
        'sb_title': "Input Parameters",
        'lbl_day': "Day of Week",
        'lbl_road': "Road Type",
        'lbl_speed': "Speed Limit (mph)",
        'lbl_light': "Light Conditions",
        'lbl_weather': "Weather Conditions",
        'lbl_area': "Area Type",
        'lbl_hour': "Time of Day",
        'lbl_comp': "Enable Model Comparison (XGBoost)",
        
        'days': ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        'roads': {'Urban/Single': 1, 'Dual Carriageway': 2, 'Motorway': 3, 'Unknown': 0},
        'lights': {'Daylight': 1, 'Darkness (Lit)': 2, 'Darkness (No Lights)': 3, 'Unknown': 0},
        'weathers': {'Fine': 1, 'Raining': 2, 'Snowing': 3, 'Fog': 4, 'Unknown': 0},
        'areas': {'Urban': 1, 'Rural': 2},
        
        'res_main': "Primary Model Analysis (LightGBM)",
        'res_comp': "Comparative Model (XGBoost)",
        'risk_levels': ['Slight', 'Serious', 'Fatal'],
        'risk_msgs': [
            "Low Risk (Slight)",
            "High Risk (Serious)",
            "Critical Risk (Fatal)"
        ],
        'rec_head': "Safety Improvement Recommendations",
        'rec_speed': "📉 <b>Sensitivity Analysis:</b> Reducing speed to <b>{0}</b> decreases fatal risk by <b>{1:.1f}%</b>.",
        'rec_light': "💡 <b>Infrastructure:</b> Installing street lights reduces fatal risk by <b>{0:.1f}%</b>.",
        'footer': "Student: Alireza Shams | Supervisor: Dr. Keyvan Borna | Fall 2025"
    }
}

# Language Selector
lang_opt = st.sidebar.radio("Language / زبان", ['فارسی', 'English'], horizontal=True)
L = 'fa' if lang_opt == 'فارسی' else 'en'
T = TEXTS[L]

# Dynamic CSS for RTL/LTR support and clean UI
st.markdown(f"""
<style>
    .main {{ direction: {T['dir']}; }}
    h1, h2, h3, p, div, span, label, .stMarkdown {{ 
        text-align: {'right' if L == 'fa' else 'left'} !important; 
        font-family: '{T['font']}', sans-serif !important; 
    }}
    .stSlider {{ direction: ltr !important; }}
    .stSlider label {{ direction: {T['dir']} !important; width: 100%; text-align: {'right' if L == 'fa' else 'left'} !important; }}
    .stSelectbox div[data-testid="stMarkdownContainer"] {{ direction: {T['dir']}; }}
    
    /* Custom Card Style for Results */
    .result-card {{
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Model Loader
# ==========================================
@st.cache_resource
def load_models():
    # Loading models generated by prediction.ipynb
    models = {}
    model_files = {'lgb': 'lgb_model.pkl', 'xgb': 'xgb_model.pkl'}
    
    for name, filename in model_files.items():
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    models[name] = pickle.load(f)
            except:
                pass
    return models

models = load_models()

# ==========================================
# 3. Main Interface & Inputs
# ==========================================
st.title(T['header_title'])
st.caption(T['header_subtitle'])
st.markdown(T['project_desc'], unsafe_allow_html=True)
st.divider()

st.sidebar.header(T['sb_title'])

def get_inputs():
    # Feature 1: Day of Week
    day_idx = st.sidebar.selectbox(T['lbl_day'], options=range(7), format_func=lambda x: T['days'][x])
    
    # Feature 2: Road Type
    road_key = st.sidebar.selectbox(T['lbl_road'], list(T['roads'].keys()))
    
    # Feature 3: Speed Limit
    speed = st.sidebar.slider(T['lbl_speed'], 10, 70, 30, step=10)
    
    # Feature 4: Light Conditions
    light_key = st.sidebar.selectbox(T['lbl_light'], list(T['lights'].keys()))
    
    # Feature 5: Weather Conditions
    weather_key = st.sidebar.selectbox(T['lbl_weather'], list(T['weathers'].keys()))
    
    # Feature 6: Area Type
    area_key = st.sidebar.radio(T['lbl_area'], list(T['areas'].keys()), horizontal=True)
    
    # Feature 7: Hour
    hour = st.sidebar.slider(T['lbl_hour'], 0, 23, 14)
    
    st.sidebar.markdown("---")
    compare_mode = st.sidebar.checkbox(T['lbl_comp'])

    # Dataframe construction matching notebook features exactly
    data = {
        'Day_of_Week': day_idx,
        'Road_Type': T['roads'][road_key],
        'Speed_limit': speed,
        'Light_Conditions': T['lights'][light_key],
        'Weather_Conditions': T['weathers'][weather_key],
        'Urban_or_Rural_Area': T['areas'][area_key],
        'Hour': hour
    }
    
    # Correct column order as per training
    cols = ['Day_of_Week', 'Road_Type', 'Speed_limit', 'Light_Conditions', 'Weather_Conditions', 'Urban_or_Rural_Area', 'Hour']
    return pd.DataFrame([data])[cols], compare_mode

input_df, show_compare = get_inputs()

# ==========================================
# 4. Prediction Logic & Visualization
# ==========================================
col_main, col_res = st.columns([2, 1])

if models.get('lgb'):
    # Main Prediction
    lgb_probs = models['lgb'].predict_proba(input_df)[0]
    lgb_pred = np.argmax(lgb_probs)
    
    with col_main:
        st.subheader(T['res_main'])
        
        # Professional Plotly Chart
        colors = ['#27ae60', '#f39c12', '#c0392b'] # Green, Orange, Red
        fig = go.Figure(data=[go.Bar(
            x=T['risk_levels'],
            y=lgb_probs,
            marker_color=colors,
            text=[f"{p*100:.1f}%" for p in lgb_probs],
            textposition='auto',
        )])
        fig.update_layout(
            yaxis_title="Probability",
            margin=dict(l=20, r=20, t=30, b=20),
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_res:
        # Result Card
        st.markdown(f"### {T['risk_levels'][lgb_pred]}")
        
        msg = T['risk_msgs'][lgb_pred]
        if lgb_pred == 2:
            st.error(msg)
        elif lgb_pred == 1:
            st.warning(msg)
        else:
            st.success(msg)
            
        st.metric("Fatal Prob.", f"{lgb_probs[2]*100:.1f}%")
        
        # Comparison Section
        if show_compare and models.get('xgb'):
            st.divider()
            st.markdown(f"**{T['res_comp']}**")
            xgb_pred = models['xgb'].predict(input_df)[0]
            st.info(f"XGBoost: {T['risk_levels'][xgb_pred]}")

    # ==========================================
    # 5. Smart Recommendations (Interactive)
    # ==========================================
    if lgb_pred > 0: # If Serious or Fatal
        st.markdown("---")
        st.subheader(T['rec_head'])
        
        # Speed Reduction Simulation
        curr_speed = input_df['Speed_limit'][0]
        if curr_speed > 20:
            sim_df = input_df.copy()
            sim_df['Speed_limit'] = curr_speed - 10
            new_prob = models['lgb'].predict_proba(sim_df)[0][2]
            diff = (lgb_probs[2] - new_prob) * 100
            if diff > 1.0:
                st.info(T['rec_speed'].format(curr_speed - 10, diff), icon="📉")

        # Infrastructure (Lighting) Simulation
        if input_df['Light_Conditions'][0] == 3: # Darkness No Lights
            sim_df_l = input_df.copy()
            sim_df_l['Light_Conditions'] = 2 # Lit
            sim_df_l['Speed_limit'] = curr_speed
            new_prob_l = models['lgb'].predict_proba(sim_df_l)[0][2]
            diff_l = (lgb_probs[2] - new_prob_l) * 100
            if diff_l > 1.0:
                st.success(T['rec_light'].format(diff_l), icon="💡")

else:
    st.error("Model files not detected. Please run the training notebook first.")

# ==========================================
# 6. Footer
# ==========================================
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #7f8c8d; font-size: 0.9em; direction: {T['dir']};">
    {T['footer']}
</div>
""", unsafe_allow_html=True)