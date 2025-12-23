import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Prediksi Produksi Padi (Random Forest)",
    page_icon="🌾",
    layout="wide"
)


# -------------------------
# Load CSS
# -------------------------
def load_css(path="assets/style.css"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# -------------------------
# Helpers
# -------------------------
DEFAULT_DATA_PATH = "data/Data_Tanaman_Padi_Sumatera_version_1.csv"
TARGET_COL_DEFAULT = "Produksi"


def read_dataset(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if os.path.exists(DEFAULT_DATA_PATH):
        return pd.read_csv(DEFAULT_DATA_PATH)
    # fallback: if user running in environment where file is mounted elsewhere
    # (for your case in this chat environment it's /mnt/data/... but local user won't have it)
    mounted = "/mnt/data/Data_Tanaman_Padi_Sumatera_version_1.csv"
    if os.path.exists(mounted):
        return pd.read_csv(mounted)
    return None


def build_model(n_estimators, max_depth, min_samples_split, min_samples_leaf, random_state):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth if max_depth > 0 else None,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1
    )
    return model


def make_pipeline(X: pd.DataFrame, model):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])
    return pipe, num_cols, cat_cols


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# -------------------------
# UI - Header
# -------------------------
st.markdown(
    """
    <div class="card">
      <h1>🌾 Prediksi Produksi Padi (Random Forest)</h1>
      <div class="sub">
        Dataset: Padi Sumatera • Target: <b>Produksi</b> • Model: <b>RandomForestRegressor</b>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")


# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")
    uploaded = st.file_uploader("Upload CSV (opsional)", type=["csv"])

    st.markdown("### Model (Random Forest)")
    n_estimators = st.slider("n_estimators", 50, 800, 300, 50)
    max_depth = st.slider("max_depth (0 = None)", 0, 50, 0, 1)
    min_samples_split = st.slider("min_samples_split", 2, 20, 2, 1)
    min_samples_leaf = st.slider("min_samples_leaf", 1, 20, 1, 1)
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("random_state", value=42, step=1)

    st.markdown("---")
    st.markdown("### Output")
    save_model = st.toggle("Simpan model (.joblib)", value=True)
    model_name = st.text_input("Nama file model", value="model_random_forest_padi.joblib")


# -------------------------
# Load data
# -------------------------
df = read_dataset(uploaded)

if df is None:
    st.error("Dataset belum ditemukan. Upload CSV atau taruh file di folder data/ sesuai struktur.")
    st.stop()

# Basic cleaning of column names (optional)
df.columns = [c.strip() for c in df.columns]

if TARGET_COL_DEFAULT not in df.columns:
    st.error(f"Kolom target '{TARGET_COL_DEFAULT}' tidak ditemukan. Kolom yang ada: {list(df.columns)}")
    st.stop()

# Show preview
colA, colB = st.columns([1.4, 1.0], gap="large")
with colA:
    st.markdown('<div class="card"><h3> Preview Data</h3></div>', unsafe_allow_html=True)
    st.dataframe(df.head(12), use_container_width=True)

with colB:
    st.markdown('<div class="card"><h3>🔎 Ringkasan</h3></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="pills">
          <div class="pill"><div class="k">Jumlah Baris</div><div class="v">{df.shape[0]}</div></div>
          <div class="pill"><div class="k">Jumlah Kolom</div><div class="v">{df.shape[1]}</div></div>
          <div class="pill"><div class="k">Target</div><div class="v">{TARGET_COL_DEFAULT}</div></div>
        </div>
        <div class="small-note" style="margin-top:10px;">
          Tip: Model memakai <b>Provinsi</b> sebagai fitur kategorikal (OneHotEncoder).
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")


# -------------------------
# Train / Evaluate
# -------------------------
X = df.drop(columns=[TARGET_COL_DEFAULT])
y = df[TARGET_COL_DEFAULT].astype(float)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=float(test_size),
    random_state=int(random_state)
)

model = build_model(
    n_estimators=int(n_estimators),
    max_depth=int(max_depth),
    min_samples_split=int(min_samples_split),
    min_samples_leaf=int(min_samples_leaf),
    random_state=int(random_state),
)

pipe, num_cols, cat_cols = make_pipeline(X, model)

train_btn = st.button("Latih & Evaluasi Model", use_container_width=True)

if train_btn:
    with st.spinner("Melatih model..."):
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

    mae = float(mean_absolute_error(y_test, preds))
    _rmse = rmse(y_test, preds)
    r2 = float(r2_score(y_test, preds))

    st.markdown(
        """
        <div class="card">
          <h3> Hasil Evaluasi</h3>
          <div class="sub">Metrik dihitung pada data test.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div class="pills">
          <div class="pill"><div class="k">MAE</div><div class="v">{mae:,.2f}</div></div>
          <div class="pill"><div class="k">RMSE</div><div class="v">{_rmse:,.2f}</div></div>
          <div class="pill"><div class="k">R²</div><div class="v">{r2:,.4f}</div></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # Feature importance (after preprocessing -> get feature names)
    try:
        preprocess = pipe.named_steps["preprocess"]
        model_fitted = pipe.named_steps["model"]

        # Get feature names
        feature_names = []
        # numeric
        feature_names.extend(num_cols)

        # categorical (onehot)
        if len(cat_cols) > 0:
            ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
            ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
            feature_names.extend(ohe_names)

        importances = model_fitted.feature_importances_
        fi = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi = fi.sort_values("importance", ascending=False).head(20)

        st.markdown('<div class="card"><h3>🏷️ Feature Importance (Top 20)</h3></div>', unsafe_allow_html=True)

        fig = plt.figure()
        plt.barh(fi["feature"][::-1], fi["importance"][::-1])
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Gagal menampilkan feature importance: {e}")

    st.write("")

    # Save model
    if save_model:
        try:
            joblib.dump(pipe, model_name)
            st.success(f"Model disimpan sebagai: {model_name}")
            with open(model_name, "rb") as f:
                st.download_button(
                    "⬇️ Download model",
                    data=f,
                    file_name=model_name,
                    mime="application/octet-stream",
                    use_container_width=True
                )
        except Exception as e:
            st.warning(f"Gagal menyimpan model: {e}")

    # Store trained model to session for prediction UI
    st.session_state["trained_pipe"] = pipe
    st.session_state["X_columns"] = X.columns.tolist()
    st.session_state["num_cols"] = num_cols
    st.session_state["cat_cols"] = cat_cols


# -------------------------
# Prediction UI
# -------------------------
st.write("")
st.markdown('<div class="card"><h3>🧪 Prediksi Produksi (Input Manual)</h3><div class="sub">Latih model dulu, lalu isi input untuk prediksi.</div></div>', unsafe_allow_html=True)

pipe_trained = st.session_state.get("trained_pipe", None)

if pipe_trained is None:
    st.info("Klik **Latih & Evaluasi Model** terlebih dahulu agar fitur prediksi aktif.")
else:
    # Build input form based on columns
    cols = st.session_state["X_columns"]
    cat_cols = st.session_state["cat_cols"]
    num_cols = st.session_state["num_cols"]

    # defaults based on median / most_frequent from dataset
    defaults = {}
    for c in num_cols:
        defaults[c] = float(df[c].median()) if pd.api.types.is_numeric_dtype(df[c]) else 0.0
    for c in cat_cols:
        defaults[c] = df[c].mode().iloc[0] if not df[c].mode().empty else ""

    with st.form("pred_form"):
        c1, c2, c3 = st.columns(3, gap="large")

        inputs = {}
        for i, col in enumerate(cols):
            target_col = [c1, c2, c3][i % 3]
            with target_col:
                if col in cat_cols:
                    options = sorted(df[col].dropna().astype(str).unique().tolist())
                    if str(defaults[col]) not in options and len(options) > 0:
                        defaults[col] = options[0]
                    inputs[col] = st.selectbox(col, options=options, index=options.index(str(defaults[col])) if str(defaults[col]) in options else 0)
                else:
                    inputs[col] = st.number_input(col, value=float(defaults[col]))

        submitted = st.form_submit_button("🔮 Prediksi", use_container_width=True)

    if submitted:
        X_new = pd.DataFrame([inputs])
        pred = float(pipe_trained.predict(X_new)[0])
        st.markdown(
            f"""
            <div class="card">
              <h2>✅ Hasil Prediksi Produksi</h2>
              <div class="sub">Perkiraan produksi (sesuai skala data kamu):</div>
              <div style="font-size:2rem;font-weight:900;margin-top:6px;">
                {pred:,.2f}
              </div>
              <div class="small-note" style="margin-top:8px;">
                Catatan: Angka ini mengikuti satuan pada kolom <b>Produksi</b> di dataset.
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
