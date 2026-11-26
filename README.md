# ⚡ Pokemon Legendary Prediction App (Streamlit & Machine Learning)

Aplikasi web interaktif berbasis **Streamlit** untuk memprediksi apakah suatu Pokemon berstatus **Legendary** atau **Biasa (Non-Legendary)** menggunakan algoritma **Decision Tree (Entropy)** dan **Random Forest Classifier**, dilengkapi visualisasi pohon keputusan, evaluasi metrik, uji diagnostik data leakage, serta simulator prediksi interaktif.

---

## 📁 Struktur File

```text
MatDis/
│
├── app.py               # Aplikasi utama Streamlit (UI & Model ML)
├── Pokemon.csv          # Dataset Pokemon (800 baris, 13 kolom)
├── requirements.txt     # Daftar pustaka Python yang dibutuhkan
├── .gitignore           # File pengabaian git (cache, venv, dll)
└── README.md            # Dokumentasi & panduan penggunaan
```

---

## 🚀 Cara Menjalankan Secara Lokal (Localhost)

### 1. Buka Terminal / PowerShell
Pastikan Anda berada di direktori proyek ini:
```bash
cd "c:\Users\helmi\OneDrive\Dokumen\MatDis"
```

### 2. Install Dependencies
Jalankan perintah berikut untuk menginstal pustaka yang diperlukan:
```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi Streamlit
```bash
streamlit run app.py
```
*Catatan: Jika perintah `streamlit` tidak langsung dikenali di PATH, gunakan:*
```bash
python -m streamlit run app.py
```
Aplikasi akan otomatis terbuka di browser Anda pada alamat: `http://localhost:8501`.

---

## ☁️ Panduan Upload / Deploy ke Streamlit Cloud (Gratis & Online)

Agar aplikasi dapat diakses secara publik oleh dosen, teman, atau reviewer melalui tautan web (misal: `https://pokemon-legendary-pred.streamlit.app`), ikuti langkah-langkah berikut:

### Langkah 1: Buat Repository di GitHub
1. Buka [github.com](https://github.com/) dan login ke akun GitHub Anda.
2. Klik tombol **"New"** untuk membuat repository baru.
3. Beri nama repository, misalnya: `pokemon-legendary-streamlit`.
4. Pilih **Public**, lalu klik **"Create repository"**.

### Langkah 2: Upload File ke GitHub
Anda bisa mengunggah file langsung lewat browser (tombol **Add file** -> **Upload files**) atau menggunakan git terminal:

```bash
# Inisialisasi git di folder lokal
git init
git add .
git commit -m "Initial commit Pokemon Streamlit App"
git branch -M main
git remote add origin https://github.com/USERNAME_ANDA/pokemon-legendary-streamlit.git
git push -u origin main
```
> **Penting**: Pastikan file `app.py`, `Pokemon.csv`, dan `requirements.txt` ter-upload ke root folder repository GitHub Anda.

### Langkah 3: Hubungkan ke Streamlit Community Cloud
1. Kunjungi [share.streamlit.io](https://share.streamlit.io/) dan login menggunakan akun **GitHub** Anda.
2. Klik tombol **"New app"** (atau **"Create app"**).
3. Isi form deployment:
   - **Repository**: Pilih repository GitHub yang baru Anda buat (`USERNAME/pokemon-legendary-streamlit`).
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Klik tombol **"Deploy!"**.

Streamlit Cloud akan otomatis menginstal paket dari `requirements.txt` dan menjalankan `app.py`. Dalam 1–2 menit, web app Anda akan aktif dan siap dibagikan!

---

## ✨ Fitur-Fitur Aplikasi

1. **📊 Eksplorasi Data (EDA)**:
   - Metrik ringkasan dataset (800 baris, 735 Biasa vs 65 Legendary).
   - Visualisasi ketidakseimbangan kelas (*class imbalance*).
   - Histogram perbandingan stat (Total, HP, Attack, Speed, dll).
   - Tabel pencarian Pokemon interaktif.

2. **🌳 Decision Tree (Pohon Keputusan)**:
   - Visualisasi diagram pohon keputusan kustom (dengan cabang Ya/Tidak, kotak proporsi kelas, dan warna pembeda).
   - Analisis overfitting pada variasi `max_depth` (1 hingga tanpa batas).
   - Evaluasi akurasi, confusion matrix, dan recall.
   - Pengeksporan teks aturan keputusan (*decision rules*).
   - Analisis fitur terpenting (*feature importances*).

3. **🌲 Random Forest Classifier**:
   - Model ensemble (100 estimators, `class_weight='balanced'`).
   - Laporan klasifikasi komprehensif (Precision, Recall, F1-Score).
   - Peringkat fitur paling berpengaruh.

4. **⚖️ Perbandingan Model (DT vs RF)**:
   - Tabel perbandingan metrik evaluasi kelas minoritas (Legendary).
   - Analisis mendalam mengenai pentingnya *Recall* pada deteksi data tidak seimbang.

5. **🔬 Uji Diagnostik Data Leakage**:
   - 9 titik pengecekan integritas data (overlap index train-test, kemurnian fitur, verifikasi turunan kolom Total, stratifikasi sampling, dll).
   - Analisis varian Pokemon dengan stat identik.

6. **🔮 Simulator Prediksi Interaktif**:
   - Pilih Pokemon yang sudah ada dari database (otomatis mengisi seluruh stat).
   - Atur stat kustom menggunakan slider & input angka (HP, Attack, Defense, Sp. Atk, Sp. Def, Speed, Generation, Type 1).
   - Prediksi simultan dari Decision Tree dan Random Forest lengkap dengan persentase probabilitas keyakinan model.
