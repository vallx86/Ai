import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Konfigurasi Halaman
st.set_page_config(page_title="Analisis K-Means Kecanduan Gadget", layout="wide")

st.title("📊 Analisis Clustering Kecanduan Gadget Mahasiswa")
st.markdown("""
Aplikasi ini mengelompokkan mahasiswa berdasarkan tingkat kecanduan gadget menggunakan algoritma **K-Means**.
Data diambil dari fitur **p1 sampai p8**.
""")

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    # Menggunakan sep=';' karena file Anda menggunakan titik koma
    data = pd.read_csv('Data_ponsel_valid.csv', sep=';')
    return data

try:
    df = load_data()
    
    # Sidebar: Pengaturan Parameter
    st.sidebar.header("Konfigurasi Algoritma")
    k_value = st.sidebar.slider("Pilih Jumlah Cluster (k)", min_value=2, max_value=5, value=3)
    
    # Memilih fitur untuk clustering (p1 - p8)
    features = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
    X = df[features]

    # 1. Preprocessing: Standarisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Proses K-Means
    kmeans = KMeans(n_clusters=k_value, init='k-means++', random_state=42)
    df['Cluster_Result'] = kmeans.fit_predict(X_scaled)

    # Layout Utama
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Data & Hasil Cluster")
        st.dataframe(df, height=400)
        
        # Download Button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Hasil (.csv)", data=csv, file_name="hasil_clustering.csv", mime="text/csv")

    with col2:
        st.subheader("Visualisasi Sebaran (PCA)")
        # Karena kita punya 8 fitur, kita gunakan PCA untuk mereduksi jadi 2D agar bisa diplot
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
        pca_df['Cluster'] = df['Cluster_Result']

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Cluster', palette='bright', s=100, ax=ax)
        plt.title(f"Visualisasi Cluster (k={k_value})")
        st.pyplot(fig)

    # 3. Analisis Karakteristik Cluster
    st.divider()
    st.subheader(" Analisis Karakteristik Tiap Cluster")
    
    # Menghitung rata-rata tiap fitur per cluster
    cluster_analysis = df.groupby('Cluster_Result')[features].mean()
    
    # Menampilkan tabel rata-rata
    st.write("Tabel di bawah menunjukkan nilai rata-rata tiap parameter (p1-p8) pada setiap cluster:")
    st.table(cluster_analysis)

    # Penjelasan singkat
    st.info("""
    **Cara Membaca:**
    - Cari cluster dengan nilai rata-rata paling tinggi di sebagian besar kolom (p1-p8). Itu biasanya mengindikasikan kelompok dengan **Tingkat Kecanduan Tinggi**.
    - Bandingkan hasil `Cluster_Result` dengan kolom `Tingkat_kecanduan` asli untuk melihat konsistensi data.
    """)

except FileNotFoundError:
    st.error("File 'Data_ponsel_valid.csv' tidak ditemukan. Pastikan file berada di folder yang sama dengan app.py.")
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")