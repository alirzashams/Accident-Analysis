
import kagglehub
import pandas as pd
import os
import shutil

# 1. Download dataset
print("⏳ Downloading huge dataset from Kaggle...")
path = kagglehub.dataset_download("tsiaras/uk-road-safety-accidents-and-vehicles")
print(f"✅ Download complete at: {path}")

# 2. Find the correct CSV file
csv_file = os.path.join(path, "Accident_Information.csv")
if not os.path.exists(csv_file):
    for f in os.listdir(path):
        if f.endswith('.csv') and 'Accident' in f:
            csv_file = os.path.join(path, f)
            break

# 3. Process and Save (To replace the old small file)
print("⚙️ Processing and replacing the old dataset...")

# فقط ستون‌های مورد نیاز را برمی‌داریم که حجم الکی بالا نرود
needed_columns = [
    'Accident_Severity', 'Date', 'Time', 'Latitude', 'Longitude', 
    'Speed_limit', 'Light_Conditions', 'Weather_Conditions', 
    'Road_Surface_Conditions', 'Urban_or_Rural_Area'
]

# خواندن فایل (ممکن است زمان‌بر باشد)
df = pd.read_csv(csv_file, low_memory=False)

# استانداردسازی نام ستون‌ها (اگر در دیتای جدید متفاوت بود)
# اینجا فرض می‌کنیم دیتای Kaggle استاندارد است، اما محض احتیاط:
df.rename(columns={'accident_severity': 'Accident_Severity'}, inplace=True) 

# انتخاب ستون‌ها و حذف ردیف‌های خالی
df = df[needed_columns].dropna()

# ذخیره در مسیر پروژه شما
output_path = 'data/cleaned_accident_data.csv'

# اطمینان از وجود پوشه data
os.makedirs('data', exist_ok=True)

df.to_csv(output_path, index=False)

print(f"✅ SUCCESS! The dataset has been updated.")
print(f"📍 Location: {output_path}")
print(f"📊 New Size: {len(df)} records (Huge Upgrade!)")