import pandas as pd
import os

# 디렉토리 내 모든 .xlsx 파일 리스트
excel_files = [f for f in os.listdir('.') if f.endswith('.xlsx')]

for file in excel_files:
    print(f"\n📂 File: {file}")
    try:
        df = pd.read_excel(file)
        if 'Genotoxicity' in df.columns:
            print(df['Genotoxicity'].value_counts())
            print(df['Genotoxicity'].value_counts(dropna=False, normalize=True))
        else:
            print("⚠️ 'Genotoxicity' column not found.")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
