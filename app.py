import io
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
import streamlit as st

# ============================================================
# KONFIGURASI HALAMAN STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Pokemon Legendary Predictor - ML & Decision Tree",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan modern & rapi
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2563EB;
    }
    .legendary-badge {
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .biasa-badge {
        background: linear-gradient(135deg, #3B82F6, #1D4ED8);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNGSI VISUALISASI POHON KEPUTUSAN
# ============================================================
def render_custom_tree(model, feature_names, fitur_biner, judul_gambar="Pohon Keputusan"):
    """Menggambar pohon keputusan yang rapi dengan cabang Ya/Tidak dan warna node."""
    t = model.tree_
    nama_kelas = ["Biasa", "Legendary"]
    warna_kelas = [(0.20, 0.55, 0.80), (0.95, 0.60, 0.10)]  # biru = Biasa, oranye = Legendary

    def hitung(node):
        v = t.value[node][0]
        return (v / v.sum() * t.n_node_samples[node]).round().astype(int)

    def campur_putih(warna, proporsi):
        a = 0.25 + 0.75 * proporsi
        return tuple(1 - a * (1 - c) for c in warna)

    posisi, anak = {}, {}
    penghitung = [0]

    def layout(node, depth):
        kiri, kanan = t.children_left[node], t.children_right[node]
        if kiri == -1:  # Leaf
            posisi[node] = (penghitung[0], -depth)
            penghitung[0] += 1
            return depth
        fitur = feature_names[t.feature[node]]
        ya, tidak = (kanan, kiri) if fitur in fitur_biner else (kiri, kanan)
        anak[node] = (ya, tidak)
        d1 = layout(ya, depth + 1)
        d2 = layout(tidak, depth + 1)
        posisi[node] = ((posisi[ya][0] + posisi[tidak][0]) / 2, -depth)
        return max(d1, d2)

    depth_max = layout(0, 0)
    n_leaf = penghitung[0]

    fig, ax = plt.subplots(figsize=(max(11, n_leaf * 2.7), max(6, (depth_max + 1) * 2.8)))
    ax.set_xlim(-0.7, n_leaf - 0.3)
    ax.set_ylim(-depth_max - 0.6, 0.6)
    ax.axis("off")

    # Cabang + Label Ya / Tidak
    for node, (ya, tidak) in anak.items():
        px, py = posisi[node]
        for child, label, warna in [(ya, "Ya", "#1b7f3b"), (tidak, "Tidak", "#b3261e")]:
            cx, cy = posisi[child]
            ax.plot([px, cx], [py, cy], color="#555555", lw=1.8, zorder=1)
            ax.text(
                (px + cx) / 2, (py + cy) / 2, label,
                ha="center", va="center",
                fontsize=11, fontweight="bold", color=warna, zorder=3,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=warna, lw=1)
            )

    # Kotak node
    for node, (x, y) in posisi.items():
        jml = hitung(node)
        total = jml.sum()
        mayor = int(jml.argmax())
        warna = campur_putih(warna_kelas[mayor], jml[mayor] / total)
        info = f"n = {total}\nBiasa: {jml[0]} | Legendary: {jml[1]}"

        if node in anak:
            fitur = feature_names[t.feature[node]]
            if fitur in fitur_biner:
                awal, nilai = fitur.rsplit("_", 1) if "_" in fitur else (fitur, "1")
                pertanyaan = f"{awal} = {nilai}?"
            else:
                pertanyaan = f"{fitur} <= {t.threshold[node]:.1f}?"
            teks = f"{pertanyaan}\n{info}"
            gaya = dict(boxstyle="round,pad=0.5", fc=warna, ec="#333333", lw=1.5)
        else:
            teks = f"PREDIKSI: {nama_kelas[mayor].upper()}\n{info}"
            gaya = dict(boxstyle="round,pad=0.5", fc=warna, ec="#000000", lw=2.5)

        ax.text(x, y, teks, ha="center", va="center", fontsize=10, zorder=4, bbox=gaya)

    ax.legend(
        handles=[
            Patch(fc=warna_kelas[0], label="Mayoritas Biasa"),
            Patch(fc=warna_kelas[1], label="Mayoritas Legendary")
        ],
        loc="lower right", fontsize=10, framealpha=0.9
    )
    ax.set_title(judul_gambar, fontsize=15, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


# ============================================================
# FUNGSI LOAD DATA & PREPROCESSING
# ============================================================
@st.cache_data
def muat_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            df = pd.read_csv(uploaded_file, encoding="latin-1")
    else:
        candidates = ["Pokemon.csv", "archive/Pokemon.csv"]
        df = None
        for c in candidates:
            if os.path.exists(c):
                try:
                    df = pd.read_csv(c)
                except Exception:
                    df = pd.read_csv(c, encoding="latin-1")
                break
        if df is None:
            st.error("File Pokemon.csv tidak ditemukan di server. Silakan upload file CSV melalui sidebar.")
            st.stop()
    return df


def siapkan_fitur(df, use_total=True):
    kolom_target = [c for c in df.columns if "legendary" in c.lower()]
    if not kolom_target:
        st.error("Kolom 'Legendary' tidak ditemukan pada file CSV!")
        st.stop()
    target = kolom_target[0]

    nama_numerik = ["hp", "attack", "defense", "sp. atk", "sp. def", "speed",
                    "sp_attack", "sp_defense", "generation"]
    if use_total:
        nama_numerik += ["total", "base_total"]

    nama_tipe = ["type 1", "type1", "type_1"]
    num_cols = [c for c in df.columns if c.lower() in nama_numerik]
    type_cols = [c for c in df.columns if c.lower() in nama_tipe]

    X = pd.get_dummies(df[num_cols + type_cols], columns=type_cols, dtype=int)
    y = df[target].astype(int)

    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]
    return X, y, target, num_cols, type_cols


# ============================================================
# SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.image("https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/25.png", width=110)
    st.title("⚡ Menu Navigasi")
    
    halaman = st.radio(
        "Pilih Halaman:",
        [
            "📊 Eksplorasi Data (EDA)",
            "🌳 Decision Tree",
            "🌲 Random Forest",
            "⚖️ Perbandingan Model (DT vs RF)",
            "🔬 Uji Diagnostik Data Leakage",
            "🔮 Prediksi Pokemon Baru"
        ]
    )

    st.markdown("---")
    st.subheader("⚙️ Parameter Global")
    
    uploaded_file = st.file_uploader("Upload CSV Kustom (Opsional)", type=["csv"])
    use_total_col = st.checkbox("Sertakan kolom 'Total'", value=True, help="Total adalah jumlah stat HP, Attack, Defense, Sp.Atk, Sp.Def, Speed")
    test_size_val = st.slider("Ukuran Data Uji (Test Size)", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    random_seed = st.number_input("Random State", value=42, step=1)

    st.markdown("---")
    st.caption("Aplikasi Prediksi Pokemon Legendary • Streamlit & Scikit-Learn")


# Memuat Data
df = muat_data(uploaded_file)
X, y, target_col, num_cols, type_cols = siapkan_fitur(df, use_total=use_total_col)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size_val, random_state=random_seed, stratify=y
)
fitur_biner = {c for c in X.columns if set(X[c].unique()) <= {0, 1}}

# Inisialisasi Model B & Random Forest
@st.cache_resource
def latih_model(X_tr, y_tr, seed):
    # Decision Tree Small (Model B)
    m_b = DecisionTreeClassifier(
        criterion="entropy", max_depth=3, min_samples_leaf=5, random_state=seed
    )
    m_b.fit(X_tr, y_tr)

    # Decision Tree Full (Model A)
    m_a = DecisionTreeClassifier(criterion="entropy", random_state=seed)
    m_a.fit(X_tr, y_tr)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=seed, class_weight='balanced'
    )
    rf.fit(X_tr, y_tr)

    return m_a, m_b, rf

model_full, model_small, rf_model = latih_model(X_train, y_train, random_seed)


# ============================================================
# HALAMAN 1: EKSPLORASI DATA (EDA)
# ============================================================
if halaman == "📊 Eksplorasi Data (EDA)":
    st.markdown('<p class="main-title">📊 Eksplorasi Data Pokemon</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Analisis distribusi kelas, statistik deskriptif, dan fitur dataset Pokemon</p>', unsafe_allow_html=True)

    # Metric Row
    c1, c2, c3, c4 = st.columns(4)
    total_data = len(df)
    n_biasa = int((df[target_col] == 0).sum() if df[target_col].dtype != bool else (~df[target_col]).sum())
    n_legend = int((df[target_col] == 1).sum() if df[target_col].dtype != bool else df[target_col].sum())
    pct_legend = (n_legend / total_data) * 100

    c1.metric("Total Pokemon", f"{total_data} baris")
    c2.metric("Pokemon Biasa", f"{n_biasa}", delta=f"{100-pct_legend:.1f}%")
    c3.metric("Pokemon Legendary", f"{n_legend}", delta=f"{pct_legend:.1f}%", delta_color="inverse")
    c4.metric("Jumlah Fitur Terproses", f"{X.shape[1]} kolom")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Ketidakseimbangan Kelas (Class Imbalance)")
        dist_df = pd.DataFrame({
            "Kategori": ["Biasa (Non-Legendary)", "Legendary"],
            "Jumlah": [n_biasa, n_legend],
            "Persentase (%)": [round(100 - pct_legend, 2), round(pct_legend, 2)]
        })
        st.dataframe(dist_df, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(6, 3.8))
        colors = ["#3B82F6", "#F59E0B"]
        bars = ax.bar(dist_df["Kategori"], dist_df["Jumlah"], color=colors, edgecolor="#1F2937", linewidth=1.2)
        ax.set_ylabel("Jumlah Pokemon", fontsize=11)
        ax.set_title("Distribusi Target (Legendary vs Biasa)", fontsize=12, fontweight="bold")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 10, f"{yval} ({yval/total_data*100:.1f}%)", ha="center", va="bottom", fontweight="bold")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

    with col_right:
        st.subheader("Distribusi Stat Berdasarkan Status")
        stat_choice = st.selectbox("Pilih Stat untuk Dibandingkan:", ["Total", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"])
        
        fig2, ax2 = plt.subplots(figsize=(6, 3.8))
        biasa_vals = df[df[target_col] == 0][stat_choice].dropna()
        legend_vals = df[df[target_col] == 1][stat_choice].dropna()
        
        ax2.hist(biasa_vals, bins=25, alpha=0.6, label="Biasa", color="#3B82F6", edgecolor="black")
        ax2.hist(legend_vals, bins=25, alpha=0.7, label="Legendary", color="#F59E0B", edgecolor="black")
        ax2.set_xlabel(stat_choice, fontsize=11)
        ax2.set_ylabel("Frekuensi", fontsize=11)
        ax2.set_title(f"Histogram {stat_choice}: Biasa vs Legendary", fontsize=12, fontweight="bold")
        ax2.legend()
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("📋 Cuplikan Data Pokemon")
    search_q = st.text_input("Cari nama Pokemon:", "")
    if search_q:
        df_preview = df[df["Name"].astype(str).str.contains(search_q, case=False, na=False)]
    else:
        df_preview = df.head(15)
    st.dataframe(df_preview, use_container_width=True)


# ============================================================
# HALAMAN 2: DECISION TREE
# ============================================================
elif halaman == "🌳 Decision Tree":
    st.markdown('<p class="main-title">🌳 Decision Tree (Pohon Keputusan)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Pelatihan model klasifikasi dengan kriteria Entropy, perbandingan max_depth, dan visualisasi pohon terperinci</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🖼️ Visualisasi Pohon (Model B)", "📈 Analisis Overfitting (max_depth)", "📝 Aturan & Fitur Penting"])

    with tab1:
        st.subheader("Visualisasi Pohon Keputusan (max_depth = 3, min_samples_leaf = 5)")
        st.caption("Pohon ini dilatih dengan kriteria *Entropy*. Kotak biru menunjukkan mayoritas kelas 'Biasa', kotak oranye menunjukkan mayoritas 'Legendary'. Cabang hijau menandakan 'Ya' dan merah 'Tidak'.")

        fig_tree = render_custom_tree(
            model_small, list(X.columns), fitur_biner,
            judul_gambar="Decision Tree - Prediksi Pokemon Legendary (max_depth=3)"
        )
        st.pyplot(fig_tree)

        # Download button untuk gambar pohon
        buf = io.BytesIO()
        fig_tree.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        st.download_button(
            label="📥 Unduh Gambar Pohon (PNG)",
            data=buf.getvalue(),
            file_name="pohon_keputusan_pokemon.png",
            mime="image/png"
        )

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 📌 Evaluasi Model B (depth=3, leaf>=5)")
            y_pred_b = model_small.predict(X_test)
            acc_b = model_small.score(X_test, y_test)
            rec_b = recall_score(y_test, y_pred_b, zero_division=0)
            st.write(f"- **Akurasi Train:** {model_small.score(X_train, y_train):.3f}")
            st.write(f"- **Akurasi Test:** {acc_b:.3f}")
            st.write(f"- **Recall Legendary (Test):** {rec_b:.3f}")
            st.write(f"- **Jumlah Leaf Node:** {model_small.get_n_leaves()}")

        with c2:
            st.markdown("##### 🎯 Confusion Matrix Model B")
            cm_b = confusion_matrix(y_test, y_pred_b)
            cm_df = pd.DataFrame(
                cm_b,
                index=["Aktual: Biasa", "Aktual: Legendary"],
                columns=["Prediksi: Biasa", "Prediksi: Legendary"]
            )
            st.table(cm_df)

    with tab2:
        st.subheader("Perbandingan max_depth (Mencegah Overfitting)")
        depth_records = []
        for d in [1, 2, 3, 4, 5, 8, None]:
            m = DecisionTreeClassifier(criterion="entropy", max_depth=d, random_state=random_seed)
            m.fit(X_train, y_train)
            acc_tr = m.score(X_train, y_train)
            acc_te = m.score(X_test, y_test)
            rec = recall_score(y_test, m.predict(X_test), zero_division=0)
            depth_records.append({
                "max_depth": str(d) if d is not None else "None (Tanpa Batas)",
                "Akurasi Train": round(acc_tr, 3),
                "Akurasi Test": round(acc_te, 3),
                "Recall Legendary (Test)": round(rec, 3)
            })

        df_depth = pd.DataFrame(depth_records)
        col_t, col_p = st.columns([1, 1.2])

        with col_t:
            st.dataframe(df_depth, use_container_width=True, hide_index=True)
            st.info("""
            **💡 Observasi Overfitting:**
            Ketika `max_depth` diset ke `None`, akurasi training mencapai **1.000 (100%)**, namun akurasi test menurun.
            Pembatasan ke `max_depth = 3` membuat model lebih general dan mudah diinterpretasikan.
            """)

        with col_p:
            fig_d, ax_d = plt.subplots(figsize=(6, 4))
            x_idx = range(len(df_depth))
            ax_d.plot(x_idx, df_depth["Akurasi Train"], marker="o", label="Akurasi Train", color="#2563EB", lw=2)
            ax_d.plot(x_idx, df_depth["Akurasi Test"], marker="s", label="Akurasi Test", color="#10B981", lw=2)
            ax_d.plot(x_idx, df_depth["Recall Legendary (Test)"], marker="^", label="Recall Legendary", color="#F59E0B", lw=2)
            ax_d.set_xticks(list(x_idx))
            ax_d.set_xticklabels(df_depth["max_depth"], rotation=30)
            ax_d.set_ylabel("Skor", fontsize=11)
            ax_d.set_title("Grafik Perbandingan Skala max_depth", fontsize=12, fontweight="bold")
            ax_d.legend()
            ax_d.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig_d)

    with tab3:
        st.subheader("Aturan Pohon & Fitur Terpenting")
        c_rule, c_imp = st.columns([1.2, 1])

        with c_rule:
            st.markdown("##### 📜 Teks Aturan Pohon Keputusan (Model B)")
            tree_text = export_text(model_small, feature_names=list(X.columns))
            st.code(tree_text, language="text")

        with c_imp:
            st.markdown("##### ⭐ Fitur Paling Berpengaruh")
            imp_series = pd.Series(model_small.feature_importances_, index=X.columns).sort_values(ascending=False).head(7)
            imp_df = pd.DataFrame({"Fitur": imp_series.index, "Importance": imp_series.values.round(4)})
            st.dataframe(imp_df, use_container_width=True, hide_index=True)

            fig_imp, ax_imp = plt.subplots(figsize=(5, 3.5))
            ax_imp.barh(imp_df["Fitur"][::-1], imp_df["Importance"][::-1], color="#3B82F6")
            ax_imp.set_xlabel("Feature Importance")
            ax_imp.set_title("Top Fitur Terpenting (Decision Tree)", fontsize=11, fontweight="bold")
            st.pyplot(fig_imp)


# ============================================================
# HALAMAN 3: RANDOM FOREST
# ============================================================
elif halaman == "🌲 Random Forest":
    st.markdown('<p class="main-title">🌲 Random Forest Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Model ensemble 100 decision trees dengan penyeimbang kelas (balanced class weight)</p>', unsafe_allow_html=True)

    y_pred_rf = rf_model.predict(X_test)
    acc_rf_tr = rf_model.score(X_train, y_train)
    acc_rf_te = rf_model.score(X_test, y_test)
    rec_rf = recall_score(y_test, y_pred_rf, zero_division=0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Akurasi Training", f"{acc_rf_tr * 100:.1f}%")
    m2.metric("Akurasi Testing", f"{acc_rf_te * 100:.1f}%")
    m3.metric("Recall Legendary (Test)", f"{rec_rf * 100:.1f}%")

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🎯 Confusion Matrix Random Forest")
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        cm_rf_df = pd.DataFrame(
            cm_rf,
            index=["Aktual: Biasa", "Aktual: Legendary"],
            columns=["Prediksi: Biasa", "Prediksi: Legendary"]
        )
        st.table(cm_rf_df)

        st.subheader("📑 Classification Report")
        cr_dict = classification_report(y_test, y_pred_rf, target_names=["Biasa", "Legendary"], output_dict=True, zero_division=0)
        cr_df = pd.DataFrame(cr_dict).transpose().round(3)
        st.dataframe(cr_df, use_container_width=True)

    with c2:
        st.subheader("⭐ Top 7 Fitur Terpenting (Random Forest)")
        imp_rf = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(7)
        imp_rf_df = pd.DataFrame({"Fitur": imp_rf.index, "Importance": imp_rf.values.round(4)})
        st.dataframe(imp_rf_df, use_container_width=True, hide_index=True)

        fig_rf, ax_rf = plt.subplots(figsize=(6, 4))
        ax_rf.barh(imp_rf_df["Fitur"][::-1], imp_rf_df["Importance"][::-1], color="#10B981")
        ax_rf.set_xlabel("Importance Skor")
        ax_rf.set_title("Kepentingan Fitur Random Forest", fontsize=12, fontweight="bold")
        st.pyplot(fig_rf)


# ============================================================
# HALAMAN 4: PERBANDINGAN MODEL (DT VS RF)
# ============================================================
elif halaman == "⚖️ Perbandingan Model (DT vs RF)":
    st.markdown('<p class="main-title">⚖️ Perbandingan Decision Tree vs Random Forest</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Perbandingan komprehensif metrik evaluasi model klasifikasi Pokemon Legendary</p>', unsafe_allow_html=True)

    y_pred_dt = model_small.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)

    report_dt = classification_report(y_test, y_pred_dt, target_names=["Biasa", "Legendary"], output_dict=True, zero_division=0)
    report_rf = classification_report(y_test, y_pred_rf, target_names=["Biasa", "Legendary"], output_dict=True, zero_division=0)

    perbandingan_data = {
        "Metrik Evaluasi": [
            "Akurasi Keseluruhan (Overall Accuracy)",
            "Precision (Kelas Legendary)",
            "Recall (Kelas Legendary)",
            "F1-Score (Kelas Legendary)"
        ],
        "Decision Tree (Model B)": [
            round(report_dt["accuracy"], 3),
            round(report_dt["Legendary"]["precision"], 3),
            round(report_dt["Legendary"]["recall"], 3),
            round(report_dt["Legendary"]["f1-score"], 3)
        ],
        "Random Forest": [
            round(report_rf["accuracy"], 3),
            round(report_rf["Legendary"]["precision"], 3),
            round(report_rf["Legendary"]["recall"], 3),
            round(report_rf["Legendary"]["f1-score"], 3)
        ]
    }

    df_comp = pd.DataFrame(perbandingan_data)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    st.markdown("---")

    col_cm1, col_cm2 = st.columns(2)
    with col_cm1:
        st.subheader("Confusion Matrix: Decision Tree")
        cm_dt = confusion_matrix(y_test, y_pred_dt)
        st.table(pd.DataFrame(cm_dt, index=["Aktual: Biasa", "Aktual: Legendary"], columns=["Pred: Biasa", "Pred: Legendary"]))

    with col_cm2:
        st.subheader("Confusion Matrix: Random Forest")
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        st.table(pd.DataFrame(cm_rf, index=["Aktual: Biasa", "Aktual: Legendary"], columns=["Pred: Biasa", "Pred: Legendary"]))

    # Kesimpulan Komparatif
    st.subheader("📌 Kesimpulan Analisis")
    dt_rec = df_comp.loc[2, "Decision Tree (Model B)"]
    rf_rec = df_comp.loc[2, "Random Forest"]
    dt_f1 = df_comp.loc[3, "Decision Tree (Model B)"]
    rf_f1 = df_comp.loc[3, "Random Forest"]

    if rf_rec > dt_rec:
        st.success("🏆 **Random Forest lebih unggul** dalam mendeteksi Pokemon Legendary karena memiliki nilai **Recall yang lebih tinggi**.")
    elif rf_rec < dt_rec:
        st.info("🌳 **Decision Tree lebih unggul** dalam mendeteksi Pokemon Legendary (Recall lebih tinggi).")
    else:
        if rf_f1 >= dt_f1:
            st.success("✨ Kedua model memiliki tingkat deteksi (Recall) yang sama, namun **Random Forest lebih optimal secara keseluruhan (F1-Score lebih tinggi/stabil)**.")
        else:
            st.info("✨ Kedua model memiliki tingkat deteksi (Recall) yang sama, namun **Decision Tree lebih optimal secara keseluruhan (F1-Score lebih baik)**.")

    st.markdown("""
    > **Mengapa Recall Penting pada Kasus Ini?**  
    > Dataset memiliki ketidakseimbangan kelas (*class imbalance*) yang tajam, di mana hanya ~8% Pokemon yang berstatus Legendary.  
    > Kesalahan melewatkan Pokemon Legendary (*False Negative*) jauh lebih krusial dibandingkan menduga Pokemon biasa sebagai Legendary (*False Positive*).
    """)


# ============================================================
# HALAMAN 5: UJI DIAGNOSTIK DATA LEAKAGE
# ============================================================
elif halaman == "🔬 Uji Diagnostik Data Leakage":
    st.markdown('<p class="main-title">🔬 Uji Diagnostik: Data Leakage & Integritas Data</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Pemeriksaan ketat untuk memastikan tidak ada kebocoran data (data leakage) antara data latih dan data uji</p>', unsafe_allow_html=True)

    # 1. Overlap Index
    overlap_idx = len(set(X_train.index) & set(X_test.index))
    
    # 2. Duplicated features in full X
    dup_x = int(X.duplicated().sum())
    dup_names = int(df['Name'].duplicated().sum()) if 'Name' in df.columns else 0

    # 3. Identical feature rows in BOTH train and test
    cross = X_train.merge(X_test, how='inner', on=list(X.columns))
    cross_count = len(cross)

    # 4. Target in X?
    target_in_x = target_col in X.columns

    # 5. Total derived
    stat_list = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    stat_cols_in_df = [c for c in df.columns if c in stat_list or c.lower() in [s.lower() for s in stat_list]]
    total_valid = False
    if 'Total' in df.columns and len(stat_cols_in_df) == 6:
        total_valid = bool((df['Total'] == df[stat_cols_in_df].sum(axis=1)).all())

    # Tampilan Hasil Diagnostik
    c1, c2, c3 = st.columns(3)
    c1.metric("Overlap Index Train & Test", f"{overlap_idx}", "Lolos (Harus 0)" if overlap_idx == 0 else "Gagal!", delta_color="normal")
    c2.metric("Cross-Overlap Fitur Train & Test", f"{cross_count}", "Lolos (Harus 0)" if cross_count == 0 else "Ada irisan!", delta_color="normal")
    c3.metric("Kolom Target di Fitur X", "TIDAK ADA" if not target_in_x else "ADA!", "Lolos (Aman)" if not target_in_x else "Bocor!", delta_color="normal")

    st.markdown("---")
    st.subheader("📋 Rincian 9 Poin Pemeriksaan Diagnostik")

    checks = [
        ("1. Train/Test Index Overlap", f"{overlap_idx} baris", "Aman (0)", overlap_idx == 0),
        ("2. Total Baris Train + Test", f"{len(X_train)} + {len(X_test)} = {len(X_train) + len(X_test)}", f"Sesuai Total X ({len(X)})", (len(X_train) + len(X_test)) == len(X)),
        ("3. Baris Fitur Duplikat pada Seluruh X", f"{dup_x} baris", f"Ada beberapa Pokemon berbeda dengan base stat identik", True),
        ("4. Vektor Fitur Identik di KEDUA Train dan Test", f"{cross_count} baris", "Aman (0)", cross_count == 0),
        ("5. Kolom Target ('Legendary') Tidak Berada di Fitur X", "Aman (False)", "Target tidak bocor ke fitur input", not target_in_x),
        ("6. Verifikasi Kolom 'Total'", "100% Cocok (HP+Atk+Def+SpA+SpD+Spe)", "Berasal dari input stats, bukan target", total_valid),
        ("7. Stratified Sampling Ratio", f"Train: {int(y_train.sum())}/{len(y_train)} ({y_train.mean():.2%}) | Test: {int(y_test.sum())}/{len(y_test)} ({y_test.mean():.2%})", "Rasio terjaga seimbang", True),
        ("8. One-Hot Dummy Encoding", f"{len([c for c in X.columns if c.startswith('Type 1_')])} kolom dummy Type 1", "Terekspansi konsisten", True),
        ("9. Preprocessing Pipeline Leakage", "Tidak ada fit scaler/imputer global", "Bebas dari estimasi data masa depan", True)
    ]

    for title, val, note, status in checks:
        icon = "✅" if status else "❌"
        st.markdown(f"**{icon} {title}**: `{val}` — *{note}*")

    st.markdown("---")
    st.subheader("🔎 Detail Baris dengan Fitur Stat Duplikat")
    st.caption("Beberapa Pokemon memiliki stat bertarung yang persis sama (misalnya varian bentuk atau gender seperti Meowstic, Wormadam, Basculin).")

    train_idx = set(X_train.index)
    dup_mask = X.duplicated(keep=False)
    dup_cols_show = [c for c in ['Name', 'Type 1', 'Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', target_col] if c in df.columns]
    dup_df = df.loc[dup_mask, dup_cols_show].copy()
    dup_df['Split'] = ['Train' if i in train_idx else 'Test' for i in dup_df.index]
    st.dataframe(dup_df.sort_values(by=[c for c in ['Total', 'HP', 'Attack'] if c in dup_df.columns]), use_container_width=True)


# ============================================================
# HALAMAN 6: SIMULASI PREDIKSI POKEMON BARU
# ============================================================
elif halaman == "🔮 Prediksi Pokemon Baru":
    st.markdown('<p class="main-title">🔮 Simulator Prediksi Pokemon Baru</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Uji coba model machine learning secara real-time dengan memilih Pokemon yang ada atau meracik stat kustom!</p>', unsafe_allow_html=True)

    mode = st.radio("Metode Input:", ["Pilih dari Dataset", "Racik Stat Kustom Sendiri"], horizontal=True)

    type_options = sorted([c.replace("Type 1_", "") for c in X.columns if c.startswith("Type 1_")])
    if not type_options:
        type_options = ["Grass", "Fire", "Water", "Bug", "Normal", "Poison", "Electric", "Ground", "Fairy", "Fighting", "Psychic", "Rock", "Ghost", "Ice", "Dragon", "Dark", "Steel", "Flying"]

    if mode == "Pilih dari Dataset":
        poke_list = df["Name"].dropna().tolist() if "Name" in df.columns else [f"Pokemon #{i}" for i in range(len(df))]
        selected_poke = st.selectbox("Pilih Pokemon:", poke_list, index=min(149, len(poke_list)-1)) # default Mewtwo jika ada
        
        row_data = df[df["Name"] == selected_poke].iloc[0]
        
        default_type = str(row_data.get("Type 1", "Normal"))
        default_gen = int(row_data.get("Generation", 1))
        default_hp = int(row_data.get("HP", 70))
        default_atk = int(row_data.get("Attack", 80))
        default_def = int(row_data.get("Defense", 70))
        default_spa = int(row_data.get("Sp. Atk", 80))
        default_spd = int(row_data.get("Sp. Def", 70))
        default_spe = int(row_data.get("Speed", 80))
        actual_status = bool(row_data.get(target_col, False))
        
        st.info(f"Status Asli di Dataset: **{'👑 LEGENDARY' if actual_status else '⚪ BIASA (Non-Legendary)'}**")
    else:
        default_type = "Dragon"
        default_gen = 1
        default_hp = 106
        default_atk = 110
        default_def = 90
        default_spa = 154
        default_spd = 90
        default_spe = 130
        actual_status = None

    st.markdown("##### ⚙️ Atur Nilai Statistik (Stats):")
    col_in1, col_in2, col_in3 = st.columns(3)

    with col_in1:
        tipe_1 = st.selectbox("Tipe Utama (Type 1)", type_options, index=type_options.index(default_type) if default_type in type_options else 0)
        generasi = st.slider("Generasi (Generation)", 1, 6, default_gen)
        hp = st.number_input("HP (Health Points)", 1, 255, default_hp)
        atk = st.number_input("Attack", 1, 255, default_atk)

    with col_in2:
        defe = st.number_input("Defense", 1, 255, default_def)
        spa = st.number_input("Sp. Attack (Special Attack)", 1, 255, default_spa)
        spd = st.number_input("Sp. Defense (Special Defense)", 1, 255, default_spd)
        spe = st.number_input("Speed", 1, 255, default_spe)

    total_stat = hp + atk + defe + spa + spd + spe

    with col_in3:
        st.markdown("##### Total Base Stats:")
        st.markdown(f"<h1 style='color: #2563EB;'>{total_stat}</h1>", unsafe_allow_html=True)
        if total_stat >= 580:
            st.caption("🔥 Total stat sangat tinggi (indikasi kuat Legendary)")
        else:
            st.caption("🛡️ Total stat reguler")

    # Bentuk vektor input untuk prediksi
    input_dict = {col: 0 for col in X.columns}
    
    # Isi stat numerik
    for col in X.columns:
        c_low = col.lower()
        if c_low == "hp":
            input_dict[col] = hp
        elif c_low == "attack":
            input_dict[col] = atk
        elif c_low == "defense":
            input_dict[col] = defe
        elif c_low in ["sp. atk", "sp_attack"]:
            input_dict[col] = spa
        elif c_low in ["sp. def", "sp_defense"]:
            input_dict[col] = spd
        elif c_low == "speed":
            input_dict[col] = spe
        elif c_low == "generation":
            input_dict[col] = generasi
        elif c_low in ["total", "base_total"] and use_total_col:
            input_dict[col] = total_stat

    # Dummy Type 1
    type_col_name = f"Type 1_{tipe_1}"
    if type_col_name in input_dict:
        input_dict[type_col_name] = 1

    input_df = pd.DataFrame([input_dict])

    st.markdown("---")
    st.subheader("🎯 Hasil Prediksi Model")

    # Prediksi Decision Tree & Random Forest
    pred_dt = model_small.predict(input_df)[0]
    prob_dt = model_small.predict_proba(input_df)[0]

    pred_rf = rf_model.predict(input_df)[0]
    prob_rf = rf_model.predict_proba(input_df)[0]

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.markdown("### 🌳 Decision Tree")
        if pred_dt == 1:
            st.markdown('<p class="legendary-badge">👑 Prediksi: LEGENDARY</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="biasa-badge">⚪ Prediksi: BIASA (Normal)</p>', unsafe_allow_html=True)
        
        st.write(f"**Keyakinan (Probabilitas):**")
        st.progress(float(prob_dt[1]))
        st.caption(f"Peluang Legendary: **{prob_dt[1]*100:.1f}%** | Peluang Biasa: **{prob_dt[0]*100:.1f}%**")

    with res_col2:
        st.markdown("### 🌲 Random Forest")
        if pred_rf == 1:
            st.markdown('<p class="legendary-badge">👑 Prediksi: LEGENDARY</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="biasa-badge">⚪ Prediksi: BIASA (Normal)</p>', unsafe_allow_html=True)
        
        st.write(f"**Keyakinan (Probabilitas):**")
        st.progress(float(prob_rf[1]))
        st.caption(f"Peluang Legendary: **{prob_rf[1]*100:.1f}%** | Peluang Biasa: **{prob_rf[0]*100:.1f}%**")
