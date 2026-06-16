import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

print("📊 Membuka data hasil pengujian untuk pengecekan overfit...")
# Memuat data pengujian yang disimpan saat proses training sebelumnya
X_train = pd.read_csv("X_train_saved.csv")
X_test = pd.read_csv("X_test_saved.csv")
y_test = pd.read_csv("y_test_saved.csv").values.flatten()

# Memuat model matang
model = xgb.XGBRegressor()
model.load_model("model_dust_track.json")

# Ambil prediksi untuk data training dan data testing
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# --- MEMBUAT VISUALISASI ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# GRAFIK 1: SCATTER PLOT PREDIKSI VS AKTUAL (DATA TESTING)
axes[0].scatter(y_test, y_pred_test, alpha=0.3, color='#4ba3ff', label='Data Uji (Testing)')
# Garis diagonal sempurna (Kondisi tebakan 100% benar)
ideal_line = [min(y_test), max(y_test)]
axes[0].plot(ideal_line, ideal_line, color='#ff4b4b', linestyle='--', linewidth=2, label='Garis Sempurna (Ideal)')
axes[0].set_title(f"Grafik Akurasi Prediksi vs Aktual\n(R² Score: {r2_score(y_test, y_pred_test):.3f})", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Nilai PM2.5 Aktual Lapangan (µg/m³)")
axes[0].set_ylabel("Nilai PM2.5 Tebakan XGBoost (µg/m³)")
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend()

# GRAFIK 2: VALIDASI DISTRIBUSI EROR (MENGECEK OVERFIT)
eror_test = y_test - y_pred_test
sns.histplot(eror_test, kde=True, ax=axes[1], color='#ffaa00', bins=40)
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Eror = 0 (Tepat Sasaran)')
axes[1].set_title("Distribusi Margin Kesalahan (Residual Error)\nSemakin Rapat di Angka 0 = Makin Aman dari Overfit", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Selisih Meleset (µg/m³)")
axes[1].set_ylabel("Frekuensi Kejadian")
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend()

plt.tight_layout()
# Menyimpan hasil grafik agar bisa Anda lihat langsung di komputer
output_filename = "evaluasi_overfit_dust_track.png"
plt.savefig(output_filename, dpi=300)
plt.close()

print(f"🎉 GRAFIK BERHASIL DICETAK!")
print(f"📂 Silakan buka file gambar: '{output_filename}' di folder proyek Anda untuk melihat grafiknya.")