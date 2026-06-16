import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("🔄 Membaca seluruh dataset asli (35.064 Baris)...")
df = pd.read_csv("PRSA_Data_Aotizhongxin_20130301-20170228.csv")

# 1. Sinkronisasi nama kolom operasional Kideco
df = df.rename(columns={
    'TEMP': 'Suhu Udara (°C)',
    'PRES': 'Tekanan Udara Area Pit (hPa)',
    'DEWP': 'Kelembapan Jalur Hauling (%)',
    'WSPM': 'Kecepatan Angin (m/s)',
    'RAIN': 'Curah Hujan (mm)',
    'CO': 'Kepadatan Dump Truck (Unit/Jam)',       
    'SO2': 'Indeks Aktivitas Blasting (Ppm)'       
})

fitur_sensor = [
    'Suhu Udara (°C)', 
    'Tekanan Udara Area Pit (hPa)',
    'Kelembapan Jalur Hauling (%)',
    'Kecepatan Angin (m/s)', 
    'Curah Hujan (mm)', 
    'Kepadatan Dump Truck (Unit/Jam)', 
    'Indeks Aktivitas Blasting (Ppm)'
]
target = 'PM2.5'

print("🛠️ Melakukan pembersihan data & imputasi missing value...")
df[fitur_sensor] = df[fitur_sensor].ffill().bfill()
df[target] = df[target].ffill().bfill()

# ======================================================================
# 🔥 KUNCI ROCKET AKURASI: FEATURE ENGINEERING (LAG & ROLLING FEATURES)
# ======================================================================
print("🚀 Membuat Fitur Historis Akumulasi (Lag Features)...")

# Membuat fitur kondisi debu 1 jam dan 2 jam yang lalu
df['Debu_1_Jam_Lalu'] = df[target].shift(1)
df['Debu_2_Jam_Lalu'] = df[target].shift(2)

# Membuat tren rata-rata suhu dan truk dalam 3 jam terakhir
df['Tren_Suhu_3Jam'] = df['Suhu Udara (°C)'].rolling(window=3).mean()
df['Tren_Truk_3Jam'] = df['Kepadatan Dump Truck (Unit/Jam)'].rolling(window=3).mean()

# Daftarkan semua fitur baru ke dalam daftar fitur pilihan utama
fitur_pilihan = fitur_sensor + ['Debu_1_Jam_Lalu', 'Debu_2_Jam_Lalu', 'Tren_Suhu_3Jam', 'Tren_Truk_3Jam']

# Hapus baris awal yang bernilai NaN akibat efek geser (.shift/.rolling)
df_clean = df[fitur_pilihan + [target]].dropna()
total_baris_utuh = len(df_clean)

print(f"📊 Total data siap latih dengan Fitur Akumulasi: {total_baris_utuh} baris!")

X = df_clean[fitur_pilihan]
y = df_clean[target]

# Split 80% Train, 20% Test secara urutan waktu
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print("🚀 Melatih XGBoost Regressor Tingkat Lanjut...")
model = xgb.XGBRegressor(
    objective='reg:squarederror', 
    random_state=42, 
    n_estimators=350,       
    max_depth=6,            
    learning_rate=0.05,     
    subsample=0.8,
    colsample_bytree=0.8
)

model.fit(X_train, y_train)

print("💾 Mengekspor model berkualitas tinggi...")
model.save_model("model_dust_track.json")

# Simpan data pendukung untuk Streamlit
X_train.to_csv("X_train_saved.csv", index=False)
X_test.to_csv("X_test_saved.csv", index=False)
y_test.to_csv("y_test_saved.csv", index=False)

# Hitung ulang metrik performa riil
y_pred = model.predict(X_test)
r2 = float(r2_score(y_test, y_pred))
mae = float(mean_absolute_error(y_test, y_pred))
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

metrics = {
    "total_data": total_baris_utuh,
    "r2": r2,
    "mae": mae,
    "rmse": rmse
}

with open("model_metrics.json", "w") as f:
    json.dump(metrics, f)

print(f"✨ PROCESS SELESAI DENGAN SUKSES!")
print(f"📈 Hasil Setelah Feature Engineering -> R² Score: {r2:.3f} | MAE: {mae:.2f} µg/m³ | RMSE: {rmse:.2f} µg/m³")