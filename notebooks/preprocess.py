import pandas as pd
import numpy as np
import os

# تنظیم مسیرها طبق ساختار جدید شما
# فرض: این فایل در پوشه notebooks اجرا می‌شود
INPUT_PATH = 'data/6accident_data.csv'
OUTPUT_DIR = '../data/'

# نام فایل نهایی (ترکیب پوشه و نام فایل)
OUTPUT_FILE = OUTPUT_DIR + 'cleaned_data.csv'

def preprocess_data():
    # 0. اطمینان از وجود پوشه خروجی
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    print("Loading raw data...")
    # بررسی اینکه فایل وجود دارد یا خیر
    if not os.path.exists(INPUT_PATH):
        print(f"Error: File not found at {INPUT_PATH}")
        # اگر فایل هنوز در پوشه raw نیست، مسیر قدیمی را چک کن
        alternative_path = '../data/6accident_data.csv'
        if os.path.exists(alternative_path):
            print(f"Found file at {alternative_path}, loading from there...")
            df = pd.read_csv(alternative_path)
        else:
            return
    else:
        df = pd.read_csv(INPUT_PATH)

    print(f"Original Rows: {len(df)}")

    # 1. حذف تکراری‌ها (از کد قبلی)
    df.drop_duplicates(inplace=True)

    # 2. اصلاح فرمت تاریخ (از کد قبلی)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

    # 3. حذف ردیف‌های بدون مختصات (کد جدید شما)
    df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

    # 4. فیلتر کردن مختصات بریتانیا (کد جدید شما)
    # محدوده تقریبی بریتانیا: عرض جغرافیایی 49 تا 61، طول جغرافیایی -10 تا 2
    uk_mask = (
        (df['Latitude'] > 49) & (df['Latitude'] < 61) &
        (df['Longitude'] > -10) & (df['Longitude'] < 2)
    )
    df_clean = df[uk_mask].copy()

    # گزارش حذفیات
    removed_count = len(df) - len(df_clean)
    print(f"Rows after UK filtering: {len(df_clean)}")
    print(f"❌ Removed {removed_count} rows (outliers outside UK)")

    # 5. ذخیره نهایی
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Cleaned data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    preprocess_data()