
import pandas as pd
import numpy as np
import os

# مسیر فایل ورودی و خروجی
input_file = 'data/6accident_data.csv'
output_file = 'data/cleaned_accident_data.csv'

# لیست مقادیری که باید به عنوان داده نامعتبر (NaN) در نظر گرفته شوند
missing_values = ["NA", "Data missing or out of range", "nan", ""]

def clean_accident_data():
    # بررسی وجود فایل در مسیر جاری
    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' was not found in the current directory.")
        return

    print("Loading data...")
    # خواندن فایل با شناسایی مقادیر نامعتبر
    df = pd.read_csv(input_file, na_values=missing_values)

    print(f"Original shape: {df.shape}")

    # 1. حذف ردیف‌های کاملاً تکراری
    df.drop_duplicates(inplace=True)
    print(f"Shape after removing duplicates: {df.shape}")

    # 2. مدیریت ستون‌های تاریخ و زمان
    # تبدیل ستون Date به فرمت datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    
    # 3. پر کردن یا حذف داده‌های خالی (اختیاری - بسته به نیاز پروژه)
    # در اینجا فقط ردیف‌هایی که تمام ستون‌هایشان خالی است را حذف می‌کنیم
    df.dropna(how='all', inplace=True)

    # مثال: حذف ردیف‌هایی که مختصات جغرافیایی ندارند (چون برای تحلیل مکان‌محور مهم هستند)
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

    # 4. اصلاح نوع داده ستون‌های عددی (در صورت وجود مقادیر متنی اشتباه)
    # تلاش برای تبدیل ستون‌های عددی که ممکن است اشتباهاً آبجکت خوانده شده باشند
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            continue

    print(f"Final shape: {df.shape}")
    
    # ذخیره فایل تمیز شده
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to '{output_file}'")

if __name__ == "__main__":
    clean_accident_data()