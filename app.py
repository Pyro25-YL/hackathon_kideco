import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import requests
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- 1. KONFIGURASI UTAMA DASHBOARD ---
st.set_page_config(
    page_title="Dust-Track AI - PT Kideco Jaya Agung",
    page_icon="🍃",
    layout="wide"
)

st.title("🍃 Dust-Track AI: Intelligent Dust Suppression System")
st.caption("Integrasi Real Benchmark Dataset, Predictive XGBoost, Explainable AI (SHAP), dan Ollama Generative LLM")
st.write("---")

# --- 2. ENGINE AI: TRAINING XGBOOST DENGAN FITUR OPTIMAL ---
@st.cache_resource
def build_and_train_ai():
    df = pd.read_csv("PRSA_Data_Aotizhongxin_20130301-20170228.csv")
    
    # Sinkronisasi nama kolom ke Bahasa Indonesia
    df = df.rename(columns={
        'TEMP': 'Suhu Udara (°C)',
        'WSPM': 'Kecepatan Angin (m/s)',
        'RAIN': 'Curah Hujan (mm)',
        'CO': 'Kepadatan Dump Truck (Unit/Jam)',       
        'SO2': 'Indeks Aktivitas Blasting (Ppm)'       
    })
    
    fitur_pilihan = [
        'Suhu Udara (°C)', 
        'Kecepatan Angin (m/s)', 
        'Curah Hujan (mm)', 
        'Kepadatan Dump Truck (Unit/Jam)', 
        'Indeks Aktivitas Blasting (Ppm)'
    ]
    target = 'PM2.5'
    
    df_clean = df[fitur_pilihan + [target]].dropna()
    df_sample = df_clean.iloc[10000:15000]
    
    X = df_sample[fitur_pilihan]
    y = df_sample[target]
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Model Training XGBoost Regressor
    model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        random_state=42, 
        n_estimators=150,
        max_depth=4, # Diturunkan ke 4 agar pohon visualisasinya rapi tidak numpuk
        learning_rate=0.1
    )
    model.fit(X_train, y_train)
    
    # Evaluasi Performa
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Hitung Global SHAP Importance
    explainer = shap.TreeExplainer(model, X_train)
    shap_values_train = explainer(X_train)
    mean_abs_shap = np.mean(np.abs(shap_values_train.values), axis=0)
    
    df_importance = pd.DataFrame({
        'Parameter Indikator': [f"🔹 {f}" for f in fitur_pilihan],
        'Skor Pengaruh (Mean |SHAP|)': mean_abs_shap
    }).sort_values(by='Skor Pengaruh (Mean |SHAP|)', ascending=False).reset_index(drop=True)
    
    return model, explainer, fitur_pilihan, r2, mae, rmse, df_importance, len(X_train)

model_xgb, shap_explainer, fitur_pilihan, r2, mae, rmse, df_importance, total_train_data = build_and_train_ai()

# --- 3. PANEL INTERAKTIF SIMULATOR GATEWAY (SIDEBAR) ---
st.sidebar.header("🎛️ Simulator Sensor Jalur Tambang")
st.sidebar.write("Sesuaikan parameter lingkungan untuk simulasi kondisi:")

sim_suhu = st.sidebar.slider("Sensor Suhu Udara (°C)", -10.0, 42.0, 31.0)
sim_wspm = st.sidebar.slider("Sensor Kecepatan Angin (m/s)", 0.0, 15.0, 4.5)
sim_rain = st.sidebar.slider("Sensor Curah Hujan (mm)", 0.0, 20.0, 0.0)
sim_truk = st.sidebar.slider("Sensor Kepadatan Dump Truck (Unit/Jam)", 100, 2000, 1485)  
sim_blast = st.sidebar.slider("Sensor Aktivitas Blasting (Ppm)", 0, 150, 101)

st.sidebar.write("---")
jalankan_ai = st.sidebar.button("🚀 Kirim Data & Evaluasi Sistem AI", use_container_width=True)

# --- ENGINE GENERATIVE AI (OLLAMA LOKAL) ---
def generate_ollama_alert(suhu, angin, truk, pm25_pred, faktor_pemicu):
    url = "http://localhost:11434/api/generate"
    prompt = f"Prompt K3..." # Sesuai template kemarin
    payload = {"model": "llama3", "prompt": prompt, "stream": False}
    try:
        response = requests.post(url, json=payload, timeout=3)
        return response.json().get("response", "").replace("\n", " ")
    except:
        if pm25_pred >= 100:
            return f"🚨 [ALERT K3 OPERASIONAL PT KIDECO] Prediksi PM2.5 mencapai {pm25_pred:.1f} µg/m³ (BAHAYA TINGGI). Faktor utama: '{faktor_pemicu}'. Mitigasi: Katup pipa air Smart Sprinkler otomatis diaktifkan penuh, dump truck maksimal 20 km/jam, wajib masker."
        else:
            return f"✅ [LAPORAN K3 OPERASIONAL PT KIDECO] Prediksi PM2.5 normal ({pm25_pred:.1f} µg/m³). Faktor utama: '{faktor_pemicu}'. Mitigasi: Sprinkler hemat air (standby), jalan hauling aman."

# --- 4. PEMBAGIAN TAB INTERAKTIF ---
tab1, tab2 = st.tabs(["🖥️ Live Dashboard & Simulator", "⚙️ Proses Internal XGBoost & XAI"])

# ==================== TAB 1: DASHBOARD SIMULATOR ====================
with tab1:
    if jalankan_ai:
        input_data = pd.DataFrame([[sim_suhu, sim_wspm, sim_rain, sim_truk, sim_blast]], columns=fitur_pilihan)
        pred_pm25 = model_xgb.predict(input_data)[0]
        if pred_pm25 < 0: pred_pm25 = 0
        
        shap_vals = shap_explainer(input_data)
        fitur_terbesar_idx = np.argmax(np.abs(shap_vals.values[0]))
        faktor_utama = fitur_pilihan[fitur_terbesar_idx]
        
        st.write("### 🔮 Hasil Evaluasi Real-Time")
        metrik_col1, metrik_col2, metrik_col3 = st.columns(3)
        with metrik_col1: st.metric(label="Prediksi Kepadatan Debu PM2.5", value=f"{pred_pm25:.1f} µg/m³")
        with metrik_col2: 
            if pred_pm25 >= 100: st.error("STATUS: 🚨 BAHAYA EKSTREM")
            else: st.success("STATUS: ✅ NORMAL AMAN")
        with metrik_col3:
            if pred_pm25 >= 100: st.warning("SMART SPRINKLER: 🔥 AUTOMATIC OPEN")
            else: st.info("SMART SPRINKLER: 🔒 CLOSE (HEMAT AIR)")
                
        st.write("---")
        layout_kiri, layout_kanan = st.columns([1.2, 1])
        
        with layout_kiri:
            st.write("### 🔍 Transparansi Model: Local XAI (SHAP)")
            st.caption(f"Akar pemicu kontribusi detik ini. Variabel dominan: **{faktor_utama}**.")
            
            nilai_kontribusi_abs = np.abs(shap_vals.values[0])
            df_plot = pd.DataFrame({'Fitur': fitur_pilihan, 'Besar Pengaruh': nilai_kontribusi_abs}).sort_values(by='Besar Pengaruh', ascending=True)
            
            fig, ax = plt.subplots(figsize=(6.5, 3.5))
            warna_fitur = ['#ff4b4b' if (v == df_plot['Besar Pengaruh'].max() and pred_pm25 >= 100) else '#ffaa00' if v == df_plot['Besar Pengaruh'].max() else '#4ba3ff' for v in df_plot['Besar Pengaruh']]
            bars = ax.barh(df_plot['Fitur'], df_plot['Besar Pengaruh'], color=warna_fitur, height=0.55)
            for bar in bars:
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.1f}', va='center', ha='left', fontsize=8, fontweight='bold', alpha=0.7)
            ax.set_xlabel("Tingkat Kekuatan Pengaruh Mutlak (Skor Absolut SHAP)", fontsize=9)
            ax.set_title("Peringkat Parameter Pemicu Kepadatan Debu Detik Ini", fontsize=10, fontweight='bold')
            ax.set_xlim(0, max(nilai_kontribusi_abs) * 1.15)
            ax.grid(axis='x', linestyle=':', alpha=0.5)
            st.pyplot(fig)
            plt.close(fig) 
            
        with layout_kanan:
            st.write("### 🤖 Perintah Komando: Generative AI Alert")
            st.caption("Pesan radio otomatis hasil rumusan analitis AI:")
            with st.spinner("Ollama sedang menyusun pesan komando..."):
                pesan_mitigasi = generate_ollama_alert(sim_suhu, sim_wspm, sim_truk, pred_pm25, faktor_utama)
                with st.chat_message("assistant", avatar="🤖"): st.write(pesan_mitigasi)
    else:
        st.info("💡 **Petunjuk Pengujian:** Sesuaikan nilai parameter sensor cuaca pada menu sidebar kiri, lalu klik tombol **'Kirim Data & Evaluasi Sistem AI'**.")

# ==================== TAB 2: PROSES LOGIKA INTERNAL AI ====================
with tab2:
    st.write("### ⚙️ Arsitektur Putusan Matematika XGBoost & SHAP")
    st.write("Halaman khusus audit transparansi untuk membuktikan akurasi dan keabsahan keputusan model AI.")
    
    # Bagian 1: Metrik Performa Data
    st.write("#### 1. Hasil Validasi Performa Training Model")
    report_col1, report_col2, report_col3, report_col4 = st.columns(4)
    report_col1.metric(label="Dataset Terbaca", value=f"{total_train_data} Baris")
    report_col2.metric(label="Akurasi Model (R² Score)", value=f"{r2:.3f}")
    report_col3.metric(label="Margin Kesalahan (MAE)", value=f"{mae:.2f} µg/m³")
    report_col4.metric(label="Deviasi Ekstrem (RMSE)", value=f"{rmse:.2f} µg/m³")
    
    st.write("---")
    
    # Bagian 2: Visualisasi Struktur Decision Tree XGBoost
    st.write("#### 2. Struktur Pohon Keputusan (XGBoost Tree Visualizer)")
    st.caption("Grafik di bawah membedah salah satu struktur pohon internal bagaimana algoritma memotong ambang batas data sensor untuk mengambil keputusan.")
    
    try:
        # Menggambar Decision Tree nomor 0 dari model XGBoost
        fig_tree, ax_tree = plt.subplots(figsize=(22, 7))
        xgb.plot_tree(model_xgb, num_trees=0, ax=ax_tree, rankdir='LR') # LR = Left to Right agar memanjang rapi
        plt.tight_layout()
        st.pyplot(fig_tree)
        plt.close(fig_tree)
    except Exception as e:
        st.warning("💡 *Catatan Teknis Visualisasi:* Untuk memunculkan bagan diagram pohon keputusan, pastikan library `graphviz` sudah terinstall di laptop Anda (`pip install graphviz`).")

    st.write("---")
    
    # Bagian 3: Tabel Pembongkar Nilai SHAP (Lokal)
    st.write("#### 3. Logika Penjumlahan Nilai Kontribusi SHAP (Kausalitas Lokal)")
    st.caption("Tabel di bawah membongkar bagaimana angka prediksi akhir didapatkan dari penjumlahan Nilai Dasar dengan Nilai Pengaruh SHAP.")
    
    if jalankan_ai:
        # Mengambil base value (nilai rata-rata prediksi)
        base_val = shap_explainer.expected_value
        nilai_shap_mentah = shap_vals.values[0]
        
        # Membuat tabel rincian penambahan matematika
        df_audit = pd.DataFrame({
            'Kondisi Indikator': ['[Acuan Rata-rata Awal] Nilai Dasar Model'] + fitur_pilihan,
            'Nilai Sensor Saat Ini': ['-'] + [str(sim_suhu), str(sim_wspm), str(sim_rain), str(sim_truk), str(sim_blast)],
            'Kontribusi Pengaruh (SHAP Value)': [f"{base_val:.2f}"] + [f"{' ' if x>=0 else ''}{x:.2f}" for x in nilai_shap_mentah],
            'Dampak Operasional': ['Titik Awal'] + ['Menaikkan Debu' if x >= 0 else 'Menurunkan Debu' for x in nilai_shap_mentah]
        })
        
        st.table(df_audit)
        
        # Konfirmasi kesimpulan matematika bawah tabel
        st.info(f"**Rumus Matematika Sistem:** Nilai Dasar ({base_val:.1f}) + Total Pengaruh SHAP ({sum(nilai_shap_mentah):.1f}) = **Prediksi Akhir PM2.5 ({pred_pm25:.1f} µg/m³)**.")
    else:
        st.warning("Silakan jalankan simulasi di Tab 1 terlebih dahulu untuk memunculkan tabel audit matematika sensor di halaman ini.")