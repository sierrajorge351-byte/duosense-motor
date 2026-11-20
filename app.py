from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ml_core import CLASSES, FEATURES, FS, build_dataset, cluster_unsupervised, elbow_curve, filter_signal, train_supervised, validate_feature_frame

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

st.set_page_config(page_title="DuoSense | Diagnóstico multimodal", page_icon="⚙️", layout="wide")
st.markdown("""
<style>
  .block-container {max-width: 1320px; padding-top: 1.7rem; padding-bottom: 3rem;}
  [data-testid="stAppViewContainer"] {background: linear-gradient(180deg,#f4f8fb 0%,#ffffff 330px);}
  [data-testid="stSidebar"] {background: #10283f;}
  [data-testid="stSidebar"] * {color: #eef6fa !important;}
  [data-testid="stSidebar"] [data-baseweb="select"] * {color:#10283f !important;}
  [data-testid="stSidebar"] input {color:#10283f !important;}
  .ds-hero {padding: 1.35rem 1.55rem; border-radius: 18px; color:white;
    background: linear-gradient(120deg,#10283f,#176b78); margin: .15rem 0 1.1rem 0;
    box-shadow: 0 12px 28px rgba(16,40,63,.14);}
  .ds-hero h1 {color:white; margin:0 0 .25rem 0; font-size:2.25rem;}
  .ds-hero p {color:#d9edf0; margin:0; font-size:1.02rem;}
  .ds-kicker {font-size:.76rem; letter-spacing:.11em; text-transform:uppercase; color:#9be3dc; font-weight:700;}
  div[data-testid="stMetric"] {background:white; border:1px solid #e3ebf0; border-radius:14px; padding: .75rem 1rem; box-shadow:0 3px 12px rgba(16,40,63,.045);}
  div[data-testid="stMetricLabel"] {color:#496171;}
  .ds-card {background:#fff; border:1px solid #e3ebf0; border-radius:14px; padding:1rem 1.1rem; height:100%;}
  .ds-card h4 {margin:.15rem 0 .45rem 0; color:#10283f;}
  .ds-card p {margin:0; color:#496171; font-size:.94rem;}
  div[data-testid="stTabs"] button {font-weight:650;}
  div[data-testid="stAlert"] {border-radius:12px;}
</style>
<div class="ds-hero"><div class="ds-kicker">Proyecto final · Machine Learning</div>
<h1>DuoSense</h1><p>Explora la salud de un rodamiento con vibración y corriente del motor.</p></div>
""", unsafe_allow_html=True)
st.caption("Prototipo académico · datos experimentales Paderborn · aplicación local en tu computador")

with st.sidebar:
    st.header("Ajustes de la prueba")
    trees = st.slider("Árboles del clasificador", 100, 500, 250, step=50)
    state = st.selectbox("Estado de la señal de ejemplo", CLASSES)
    channel = st.radio("Modalidad para comparar", ["Vibración", "Corriente del motor"])
    uploaded_csvs = st.file_uploader(
        "Cargar uno o varios CSV de características",
        type=["csv"],
        accept_multiple_files=True,
        key="feature_csvs",
        help="Cada archivo debe incluir las diez características de DuoSense, 'estado' y 'recording'. Se requieren al menos cuatro archivos de medición por clase.",
    )
    st.caption("Cada CSV se conserva como conjunto separado. El original Paderborn permanece disponible.")

@st.cache_data(show_spinner="Cargando mediciones reales y características multimodales…")
def cached_data():
    return build_dataset()

@st.cache_resource(show_spinner="Entrenando el modelo supervisado…")
def cached_supervised(frame, n_estimators):
    return train_supervised(frame, n_estimators)

@st.cache_data(show_spinner="Agrupando los registros sin usar sus etiquetas…")
def cached_clusters(frame):
    return cluster_unsupervised(frame, len(CLASSES))

@st.cache_data(show_spinner="Calculando la curva del codo…")
def cached_elbow(frame):
    return elbow_curve(frame, max_k=8)

try:
    data, examples = cached_data()
except Exception as exc:
    st.error(f"No se pudieron cargar los datos Paderborn incluidos. Comprueba que estén los archivos de la carpeta data. Detalle: {exc}")
    st.stop()

data_sources = {"Paderborn incluido": data}
used_names = set(data_sources)
invalid_uploads = []
for uploaded_csv in uploaded_csvs:
    try:
        uploaded_frame = validate_feature_frame(pd.read_csv(uploaded_csv))
        source_name = uploaded_csv.name
        duplicate_number = 2
        while source_name in used_names:
            source_name = f"{uploaded_csv.name} ({duplicate_number})"
            duplicate_number += 1
        used_names.add(source_name)
        data_sources[source_name] = uploaded_frame
    except Exception as exc:
        invalid_uploads.append(f"{uploaded_csv.name}: {exc}")

with st.sidebar:
    if invalid_uploads:
        for message in invalid_uploads:
            st.warning(f"Archivo omitido: {message}")
    data_source = st.selectbox(
        "Conjunto para revisar",
        list(data_sources),
        help="Cambiar de conjunto actualiza solo la vista y los resultados de ese archivo.",
    )
    st.caption(f"Disponibles en esta sesión: {len(data_sources)} conjuntos válidos.")

data = data_sources[data_source]
st.session_state["duosense_data_source"] = data_source

overview = st.columns(4)
overview[0].metric("Ventanas cargadas", f"{len(data):,}")
overview[1].metric("Modalidades", "2")
overview[2].metric("Estados", f"{data['estado'].nunique()}")
overview[3].metric("Registros", f"{data['recording'].nunique()}")

section = st.radio(
    "Elige qué quieres explorar",
    ["Inicio", "Clasificar", "Agrupar", "Señal antes/después", "Probar y guardar"],
    horizontal=True,
    label_visibility="collapsed",
)

if section == "Inicio":
    st.markdown("### ¿Qué hace esta aplicación?")
    st.write("Mira dos señales del mismo motor: cuánto vibra y cuánta corriente consume. Resume pequeños fragmentos de ambas y permite probar dos formas de encontrar patrones.")
    st.caption(f"Datos usados en esta ejecución: {data_source} · {len(data)} ventanas · {data['recording'].nunique()} archivos de medición.")
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="ds-card"><h4>① Medimos</h4><p>Vibración y corriente: dos pistas distintas sobre el mismo motor.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="ds-card"><h4>② Comparamos</h4><p>Un método aprende ejemplos conocidos; otro busca grupos por su cuenta.</p></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="ds-card"><h4>③ Interpretamos</h4><p>Revisamos métricas y señales, sin tratar el resultado como diagnóstico industrial.</p></div>', unsafe_allow_html=True)
    st.markdown("#### Estados incluidos")
    st.write("Saludable · Daño real de pista interior")
    st.info("**Límite importante:** son mediciones reales, pero hay un solo rodamiento físico por estado. Las métricas describen este conjunto; no demuestran que el modelo funcione con otros motores.")
    with st.expander("¿Qué significan las columnas de datos?"):
        st.write("Cada fila resume un fragmento de señal. RMS indica su nivel; curtosis y factor de cresta ayudan a describir picos; frecuencia dominante y bandas de energía resumen su contenido. Se combinan cinco valores de vibración y cinco de corriente.")
        st.dataframe(data.groupby("estado")[FEATURES].mean().round(3), use_container_width=True)

elif section == "Clasificar":
    st.markdown("### Clasificar: aprender con ejemplos conocidos")
    st.write("**Random Forest** aprende de fragmentos marcados como saludables o dañados y luego intenta clasificar fragmentos reservados para la prueba.")
    st.caption("La prueba usa archivos de medición separados del entrenamiento. Sin embargo, ambos grupos aún incluyen los mismos dos rodamientos físicos.")
    try:
        model, test_pack, metrics = cached_supervised(data, trees)
    except Exception as exc:
        st.error(f"No se pudo entrenar el clasificador: {exc}")
        st.stop()
    m1, m2, m3 = st.columns(3)
    m1.metric("Exactitud", f"{metrics['accuracy']:.1%}")
    m2.metric("F1 macro", f"{metrics['f1_macro']:.3f}")
    m3.metric("Prueba", f"{metrics['test_size']} ventanas")
    cm = metrics["matrix"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=15, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set(xlabel="Predicción", ylabel="Etiqueta real", title="Matriz de confusión")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    st.pyplot(fig, clear_figure=True)
    report_rows = {name: metrics["report"][name] for name in CLASSES}
    st.dataframe(pd.DataFrame(report_rows).T[["precision", "recall", "f1-score", "support"]].round(3), use_container_width=True)

elif section == "Agrupar":
    st.markdown("### Agrupar: buscar patrones sin respuestas")
    st.write("**K-means** intenta separar los fragmentos en grupos sin conocer las etiquetas. Primero estandarizamos las diez características para que las de mayor escala no dominen el cálculo.")
    try:
        clusters, projection, unsup = cached_clusters(data)
        k_values, inertias = cached_elbow(data)
    except Exception as exc:
        st.error(f"No se pudieron agrupar los datos: {exc}")
        st.stop()
    u1, u2, u3 = st.columns(3)
    u1.metric("Silhouette", f"{unsup['silhouette']:.3f}", help="Cohesión y separación interna de los grupos; no requiere etiquetas.")
    u2.metric("ARI de comparación", f"{unsup['ari_external']:.3f}", help="Comparación posterior entre grupos y clases conocidas; no intervino en el agrupamiento.")
    explained = sum(unsup["pca_explained_variance"])
    u3.metric("Varianza mostrada por PCA", f"{explained:.1%}", help="Porcentaje de variación de las diez características representado en estos dos ejes.")
    st.caption("Se muestran dos grupos porque el caso compara dos estados. La curva del codo ayuda a explorar K; no decide automáticamente el número de grupos.")
    chart_left, chart_right = st.columns(2)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    scatter = ax.scatter(projection[:, 0], projection[:, 1], c=clusters, cmap="viridis", s=18, alpha=0.75)
    ax.set(title="Grupos proyectados con PCA", xlabel="Componente principal 1", ylabel="Componente principal 2")
    ax.grid(alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="Grupo K-means")
    chart_left.pyplot(fig, clear_figure=True)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(k_values, inertias, marker="o", color="#0f766e", linewidth=2)
    ax.axvline(len(CLASSES), color="#d97706", linestyle="--", linewidth=1.5, label="K usado = 2")
    ax.set(title="Método del codo · búsqueda exploratoria", xlabel="Número de grupos (K)", ylabel="Inercia dentro de los grupos")
    ax.set_xticks(k_values)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    chart_right.pyplot(fig, clear_figure=True)
    st.caption(f"Los componentes de PCA representan {unsup['pca_explained_variance'][0]:.1%} y {unsup['pca_explained_variance'][1]:.1%} de la variación; juntos, {explained:.1%}.")
    composition = pd.crosstab(data["estado"], pd.Series(clusters, name="Grupo"))
    st.markdown("#### ¿Qué estados terminaron en cada grupo?")
    st.dataframe(composition, use_container_width=True)
    st.info("El ARI se calcula al final, al comparar grupos con las etiquetas conocidas. Las etiquetas no participan en K-means. El codo es una guía visual, no una prueba de que los grupos sean clases reales.")

elif section == "Señal antes/después":
    st.markdown("### Ver la señal original y la filtrada")
    st.write("El filtro reduce componentes por encima de 5 kHz para facilitar la inspección visual de la traza.")
    example = examples[state]
    key = "vibration" if channel == "Vibración" else "current"
    raw = example[key]
    processed = filter_signal(raw, fs=FS, cutoff=5000)
    t = example["t"]
    stop = int(0.03 * FS)
    before, after = st.columns(2)
    with before:
        st.markdown("**Antes · señal cruda**")
        fig, ax = plt.subplots(figsize=(6, 3.3))
        ax.plot(t[:stop], raw[:stop], color="#d97706", lw=0.9)
        ax.set(xlabel="Tiempo (s)", ylabel="Amplitud", title=channel)
        ax.grid(alpha=0.2)
        st.pyplot(fig, clear_figure=True)
    with after:
        st.markdown("**Después · filtro pasa-bajas, 5 kHz**")
        fig, ax = plt.subplots(figsize=(6, 3.3))
        ax.plot(t[:stop], processed[:stop], color="#0f766e", lw=0.9)
        ax.set(xlabel="Tiempo (s)", ylabel="Amplitud", title="Señal filtrada")
        ax.grid(alpha=0.2)
        st.pyplot(fig, clear_figure=True)
    st.caption("Las gráficas muestran trazas reales del conjunto Paderborn incluido, muestreadas a 64 kHz; cargarlas en la tabla de características no reemplaza estas trazas. El filtro es visual: el modelo se entrena con características de las señales crudas.")

elif section == "Probar y guardar":
    st.markdown("### Probar una ventana y guardar el resultado")
    st.write("Elige un fragmento reservado para ver qué clase predice el modelo. El botón de abajo guarda el modelo y un resumen de métricas en la carpeta `artifacts`.")
    try:
        model, test_pack, metrics = cached_supervised(data, trees)
    except Exception as exc:
        st.error(f"No se pudo entrenar el modelo para la prueba: {exc}")
        st.stop()
    x_test, y_test, y_pred, test_indices = test_pack
    selected = st.selectbox("Ventana de prueba", list(range(min(30, len(x_test)))), format_func=lambda i: f"Ventana {i+1}")
    row = x_test.iloc[[selected]]
    predicted = model.predict(row)[0]
    actual = y_test.iloc[selected]
    p1, p2 = st.columns(2)
    p1.metric("Clasificación del modelo", predicted)
    p2.metric("Etiqueta de referencia", actual)
    with st.expander("Características de las dos modalidades"):
        st.dataframe(row.T.rename(columns={row.index[0]: "Valor"}), use_container_width=True)
    st.write("La clasificación es exploratoria: los datos contienen un rodamiento por estado y no representan validación independiente entre equipos.")
    if st.button("Guardar modelo y métricas", type="primary"):
        import joblib
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        model_path = ARTIFACTS / f"random_forest_{stamp}.joblib"
        metrics_path = ARTIFACTS / f"resultados_{stamp}.json"
        try:
            joblib.dump(model, model_path)
            payload = {"source": data_source, "classes": CLASSES, "features": FEATURES,
                       "n_estimators": trees, "accuracy": metrics["accuracy"], "f1_macro": metrics["f1_macro"],
                       "confusion_matrix": metrics["matrix"].tolist(),
                       "group_split_recordings": {"train": metrics["train_recordings"], "test": metrics["test_recordings"]}, "created_utc": stamp}
            metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            st.success(f"Modelo y resultados guardados en {ARTIFACTS}")
        except Exception as exc:
            st.error(f"No se pudieron guardar los artefactos. Detalle: {exc}")

st.divider()
st.caption("DuoSense · prototipo académico con mediciones reales de Paderborn · no es instrumento de monitoreo")

# The Windows launcher waits for this marker before opening the browser, so it
# does not show a blank page while Streamlit is still running the first script.
try:
    (ROOT / ".duosense_ready").write_text("ready", encoding="utf-8")
except OSError:
    pass

