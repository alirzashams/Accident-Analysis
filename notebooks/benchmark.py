import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

# ==========================================
# 1. تنظیمات و بارگذاری داده‌ها (هوشمند)
# ==========================================
# تلاش برای یافتن فایل دیتاست در مسیرهای مختلف
possible_paths = [
    'data/cleaned_accident_data.csv',
    'data/6accident_data.csv',
    'cleaned_accident_data.csv',
    '6accident_data.csv'
]

df = None
for path in possible_paths:
    if os.path.exists(path):
        print(f"✅ Dataset found: {path}")
        df = pd.read_csv(path)
        break

if df is None:
    print("❌ ERROR: Dataset not found! Please check file path.")
    exit()

# ==========================================
# 2. پیش‌پردازش و رفع مشکل حذف داده‌ها
# ==========================================
print("⚙️  Preprocessing data...")

target_col = 'Accident_Severity'

# 1. حذف ستون‌های کاملاً بی‌ربط یا دارای مقادیر خالی زیاد
cols_to_drop = [
    'Accident_Index', 'Date', 'Time', 'LSOA_of_Accident_Location', 
    'Location_Easting_OSGR', 'Location_Northing_OSGR', 
    'Junction_Detail', 'Junction_Control', 'Special_Conditions_at_Site' # این‌ها معمولا خالی‌اند
]
df = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)

# 2. به جای حذف ردیف‌ها، جاهای خالی را پر می‌کنیم! (نکته کلیدی)
# پر کردن مقادیر عددی با میانگین و متنی با 'Unknown'
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna('Unknown')
    else:
        df[col] = df[col].fillna(df[col].mean())

# حالا که پر کردیم، اگر باز هم نویزی بود حذف می‌کنیم (خیلی کم پیش می‌آید)
df = df.dropna()

print(f"📊 Data shape after cleaning: {df.shape}")

# جدا کردن X و y
if target_col not in df.columns:
    print(f"❌ Error: Target column '{target_col}' not found.")
    exit()

X = df.drop([target_col], axis=1)
y = df[target_col]

# انکود کردن هدف (تبدیل Fatal به 0, 1, 2)
le_y = LabelEncoder()
y = le_y.fit_transform(y.astype(str))

# انکود کردن ویژگی‌ها
le_X = LabelEncoder()
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype(str)
    X[col] = le_X.fit_transform(X[col])

# مقیاس‌دهی
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# تقسیم داده‌ها
# نکته: اگر دیتا خیلی بزرگ باشد، SVM خیلی طول می‌کشد. برای تست فقط 10 هزار تا برمی‌داریم.
# اگر سیستم قوی دارید، خط زیر را حذف کنید تا روی کل دیتا اجرا شود.
SAMPLE_SIZE = 10000 if len(df) > 10000 else len(df)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled[:SAMPLE_SIZE], y[:SAMPLE_SIZE], test_size=0.2, random_state=42
)

# ==========================================
# 3. تعریف و اجرای مدل‌ها
# ==========================================
models = {
    "SVM (Traditional)": SVC(kernel='rbf', max_iter=1000), 
    "XGBoost (Ensemble)": xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False),
    "LightGBM (Proposed)": lgb.LGBMClassifier(verbose=-1)
}

print("\n🚀 Training and Benchmarking Models...\n")
print(f"{'Model':<20} | {'Accuracy':<10} | {'F1-Score':<10} | {'Time (s)':<10}")
print("-" * 60)

for name, model in models.items():
    start = time.time()
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        duration = time.time() - start
        
        print(f"{name:<20} | {acc:.4f}     | {f1:.4f}     | {duration:.2f}")
    except Exception as e:
        print(f"{name:<20} | Error: {e}")

print("-" * 60)
print("✅ Benchmark Finished.")