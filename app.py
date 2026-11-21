from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml_core import (ARTIFACT_DIR, DATA_DIR, FAULTS_ES, evaluate_supervised,
                     evaluate_unsupervised, feature_columns, load_builtin,
                     read_uploaded_csv, save_experiment, load_panel_cases,
                     panel_feature_columns, panel_case_table,
                     evaluate_panels_supervised, evaluate_panels_unsupervised,
                     save_panel_experiment)

st.set_page_config(page_title="ElectroPV | Diagnóstico inteligente", page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background:linear-gradient(180deg,#f4faf8 0%,#f7f9fc 42%,#fff 100%);}
h1,h2,h3 {font-family:'Manrope',sans-serif!important;color:#123a35;}
.hero {padding:1.5rem 1.8rem;border-radius:24px;background:linear-gradient(118deg,#123a35,#126d60 65%,#38a983);color:white;margin-bottom:1.1rem;}
.hero h1 {color:white!important;font-size:2.35rem;margin:0 0 .35rem 0;}
.hero p {color:#ddf4ed;font-size:1.05rem;margin:0;max-width:850px;}
.pill {display:inline-block;border:1px solid #cde7de;color:#14634e;background:#effaf5;border-radius:999px;padding:.32rem .75rem;font-size:.82rem;margin:.15rem .3rem .15rem 0;}
.callout {border-left:4px solid #16a47e;background:#eff8f4;padding:.85rem 1rem;border-radius:8px;color:#27473e;}
[data-testid="stMetric"] {background:white;border:1px solid #e2ece8;padding:1rem;border-radius:16px;box-shadow:0 4px 16px #14382d0a;}
section[data-testid="stSidebar"] {background:#f0f7f3;}
</style>
""", unsafe_allow_html=True)

FAULT_NAMES = {k: v for k, v in FAULTS_ES.items()}
LABEL_OPTIONS = [f"{k} · {v}" for k,v in FAULT_NAMES.items()]

@st.cache_data(show_spinner="Leyendo los 16 experimentos…")
def get_data():
    return load_builtin()

@st.cache_data(show_spinner="Cargando las pruebas de seis paneles…")
def get_panel_data():
    return load_panel_cases()

def dataset_zip() -> bytes:
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(DATA_DIR.glob("F[0-7][LM].csv")):
            z.write(p,arcname=p.name)
    return buf.getvalue()

def label_names(values):
    return [f"{v} · {FAULT_NAMES.get(int(v), 'Sin etiqueta')}" for v in values]

df_all=get_data()
st.sidebar.markdown("## ⚡ ElectroPV")
st.sidebar.caption("Diagnóstico de fallas en un sistema solar conectado a la red")
page=st.sidebar.radio("Ir a",["Inicio","Probar 6 paneles","Evaluar modelos","Probar mis CSV","Entender el proyecto"],label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown("**Datos disponibles**")
if page=="Probar 6 paneles":
    st.sidebar.write("6 paneles físicos · 18 pruebas")
    st.sidebar.caption("Fuente: Mismatching PV Modules · Mendeley Data · CC BY 4.0")
else:
    st.sidebar.write("16 experimentos · 8 estados · 2 modos")
    st.sidebar.caption("Fuente: GPVS-Faults · Mendeley Data · CC BY 4.0")

if page=="Inicio":
    st.markdown("<div class='hero'><h1>ElectroPV</h1><p>Una aplicación para explorar señales eléctricas reales y comparar dos formas de encontrar fallas en sistemas fotovoltaicos.</p></div>",unsafe_allow_html=True)
    for text in ["Datos de laboratorio reales","16 CSV listos para probar","Corriente + voltaje","Modelos supervisados y K-means"]:
        st.markdown(f"<span class='pill'>✦ {text}</span>",unsafe_allow_html=True)
    st.write("")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Experimentos", "16", "8 situaciones × 2 modos")
    c2.metric("Señales eléctricas", "13", "Corriente, voltaje y frecuencia")
    c3.metric("Clases", "8", "Normal + 7 escenarios de falla")
    c4.metric("Ventana de análisis", "1.024", "muestras por fragmento")
    st.markdown("### ¿Qué hace esta aplicación?")
    a,b=st.columns(2)
    with a:
        st.markdown("#### 1 · Aprende con respuestas conocidas")
        st.write("Los modelos supervisados reciben fragmentos etiquetados como normal o con un tipo de falla. Después se prueban en registros del otro modo de funcionamiento, que no se usaron para entrenar.")
        st.markdown("#### 2 · Busca grupos sin respuestas")
        st.write("K-means organiza fragmentos parecidos sin mirar sus etiquetas durante el agrupamiento. Al final se comparan los grupos con las respuestas conocidas para evaluar qué tan relacionados quedaron.")
    with b:
        st.markdown("#### 3 · Compara señales")
        st.write("Puedes analizar corriente, voltaje o ambos juntos. Así ves si combinar modalidades ayuda a reconocer las fallas en estas pruebas.")
        st.markdown("#### 4 · Prueba CSV propios")
        st.write("Carga uno o varios CSV. Cada archivo se procesa de forma independiente y la aplicación muestra la predicción por fragmento y el resumen del caso.")
    st.info("Para las pruebas con paneles físicos diferentes, abre **Probar 6 paneles**. Los 16 CSV de GPVS se conservan como un conjunto adicional de demostración.")
    st.download_button("Descargar los 16 CSV de prueba",data=dataset_zip(),file_name="ElectroPV_CSV_reales_16_experimentos.zip",mime="application/zip",use_container_width=True)
    st.caption("Los archivos descargables contienen características calculadas a partir de las señales reales. Para descargar las señales originales completas, visita la fuente Mendeley citada en la guía.")

elif page=="Probar 6 paneles":
    st.title("Pruebas con seis paneles fotovoltaicos")
    st.write("Aquí puedes probar **cada panel físico por separado**. Hay seis muestras físicas, de dos modelos distintos. Cada una tiene tres condiciones: limpia, sucia y sombreada. Son 18 archivos CSV independientes.")
    panels=get_panel_data()
    c1,c2,c3=st.columns(3)
    c1.metric("Paneles físicos", "6")
    c2.metric("Pruebas panel + condición", "18")
    c3.metric("Fragmentos medidos", f"{len(panels):,}")
    modality=st.selectbox("Qué mediciones usar",["Eléctricas + térmicas","Eléctricas","Térmica"],help="Eléctricas: corriente y voltaje; térmica: temperatura de la superficie; ambas: usa las dos fuentes.")
    cols=panel_feature_columns(modality)
    cases=panel_case_table(panels,cols)
    st.markdown("### Los 18 archivos listos para probar")
    table=cases[["panel_id","modelo","condicion"]].drop_duplicates().rename(columns={"panel_id":"Panel físico","modelo":"Modelo de panel","condicion":"Condición"})
    st.dataframe(table,use_container_width=True,hide_index=True)
    panel_opts=sorted(panels["panel_id"].unique())
    chosen=st.selectbox("Elige un panel",panel_opts)
    cond=st.selectbox("Elige la prueba",["Limpio","Sucio","Sombreado"])
    one=panels[(panels.panel_id==chosen)&(panels.condicion==cond)]
    if not one.empty:
        st.download_button("Descargar solo esta prueba CSV",one.to_csv(index=False).encode("utf-8-sig"),file_name=f"{chosen}_{cond}.csv",mime="text/csv")
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        from pathlib import Path
        for p in sorted((Path(__file__).parent/"data"/"panel_cases").glob("*.csv")):
            z.write(p,arcname=p.name)
    st.download_button("Descargar los 18 CSV de los seis paneles",buf.getvalue(),file_name="ElectroPV_6_paneles_18_pruebas.zip",mime="application/zip",type="primary")
    st.caption("Los CSV anteriores de GPVS siguen disponibles en Inicio. Esos 16 archivos corresponden a otro conjunto experimental, no a 16 paneles.")
    st.markdown("### Comparación supervisada y no supervisada")
    st.write("La prueba supervisada deja fuera un panel físico completo, entrena con los otros cinco y predice sus tres condiciones. Repite esto para cada panel. K-means agrupa las 18 pruebas por similitud sin usar sus nombres durante el agrupamiento.")
    if st.button("Ejecutar comparación para los seis paneles",type="primary"):
        try:
            with st.spinner("Comparando paneles; puede tardar unos segundos…"):
                sup=evaluate_panels_supervised(panels,modality)
                uns=evaluate_panels_unsupervised(panels,modality)
                meta={"fecha":datetime.now().isoformat(timespec="seconds"),"dataset":"Mismatching PV Modules","paneles":6,"casos":18,"modalidad":modality,"kmeans":{"k":3,"silhouette":uns["silhouette"],"ari":uns["ari"]}}
                save_panel_experiment(meta,sup["tabla"])
                st.session_state["panel_eval"]=(sup,uns)
        except Exception as e:
            st.error(f"No se pudo ejecutar la comparación: {e}")
    if "panel_eval" in st.session_state:
        sup,uns=st.session_state["panel_eval"]
        st.markdown("#### Resultados de modelos supervisados")
        st.caption("Cada fila evaluada representa un panel físico y una condición. Son 18 casos; no se mezclan fragmentos del mismo panel entre entrenamiento y prueba.")
        st.dataframe(sup["tabla"].style.format({"Exactitud":"{:.1%}","Exactitud balanceada":"{:.1%}","F1 macro":"{:.1%}"}),use_container_width=True,hide_index=True)
        pred=sup["predictions"].rename(columns={"panel_id":"Panel físico","modelo":"Modelo","real":"Condición real","predicho":"Predicción","ventanas":"Fragmentos"})
        st.dataframe(pred,use_container_width=True,hide_index=True)
        st.markdown("#### K-means: grupos por similitud")
        st.write(f"Silhouette: **{uns['silhouette']:.3f}** · ARI: **{uns['ari']:.3f}**. Son medidas de agrupamiento; no son porcentajes de diagnóstico correcto.")
        fig=px.scatter(uns["plot"],x="Componente 1",y="Componente 2",color="Grupo K-means",symbol="condicion",hover_data=["panel_id","modelo","condicion"],title="18 pruebas de paneles resumidas en dos dimensiones")
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(uns["mapping"],use_container_width=True,hide_index=True)
        st.info("Datos reales publicados por Mendeley Data: [Experimental data for Mismatching Faults of Photovoltaic Modules](https://data.mendeley.com/datasets/xjs42j8dtf/1), CC BY 4.0. Son seis muestras de panel, no seis plantas completas.")

elif page=="Evaluar modelos":
    st.title("Evaluar modelos")
    st.write("Se entrena con un modo de operación y se prueba con el otro. Así los fragmentos del mismo archivo no quedan repartidos entre entrenamiento y prueba.")
    data=df_all.copy()
    c1,c2,c3=st.columns([1,1,1.2])
    train_mode=c1.selectbox("Entrenar con",["L","M"],format_func=lambda x: "L · Potencia limitada (IPPT)" if x=="L" else "M · Potencia máxima (MPPT)")
    modality=c2.selectbox("Señales para el modelo",["Corriente + voltaje","Corriente","Voltaje"])
    selected=c3.multiselect("Experimentos incluidos",sorted(data["origen"].unique()),default=sorted(data["origen"].unique()),help="Deja los 16 seleccionados para comparar todos los estados.")
    data=data[data["origen"].isin(selected)]
    if len(selected)<16:
        st.warning("Para evaluar las ocho clases, incluye los registros de ambos modos (L y M) de cada escenario F0–F7.")
    test_mode="M" if train_mode=="L" else "L"
    st.markdown(f"**Entrenamiento:** archivos terminados en `{train_mode}` &nbsp; · &nbsp; **Prueba:** archivos terminados en `{test_mode}`",unsafe_allow_html=True)
    if st.button("Ejecutar evaluación completa",type="primary",use_container_width=True):
        with st.spinner("Entrenando y evaluando los modelos. Con estos datos suele tardar unos segundos…"):
            try:
                res=evaluate_supervised(data,modality,train_mode)
                uns=evaluate_unsupervised(data,modality,train_mode,8)
                st.session_state["eval"]={"supervised":res,"unsup":uns,"train_mode":train_mode,"modality":modality}
                meta={"fecha":datetime.now().isoformat(timespec="seconds"),"entrenamiento":train_mode,"prueba":test_mode,"modalidad":modality,"archivos":selected,
                      "kmeans":{"k":8,"silhouette":float(uns["silhouette"]),"ari":float(uns["ari"]) if uns["ari"] is not None else None,"varianza_pca_2d":float(uns["pca_variance"])}}
                saved=save_experiment(meta,res)
                st.session_state["saved_name"]=saved.name
            except Exception as e: st.error(f"No se pudo ejecutar la evaluación: {e}")
    if "eval" in st.session_state:
        ev=st.session_state["eval"];res=ev["supervised"];uns=ev["unsup"]
        st.markdown("---\n### Resultados supervisados")
        st.caption("Los tres modelos reciben exactamente el mismo entrenamiento y prueba. El mejor se identifica por F1 macro.")
        st.dataframe(res["tabla"].style.format({"Exactitud":"{:.1%}","Exactitud balanceada":"{:.1%}","F1 macro":"{:.1%}"}).highlight_max(axis=0,color="#d8f3e7"),use_container_width=True,hide_index=True)
        k1,k2,k3,k4=st.columns(4)
        best_row=res["tabla"].set_index("Modelo").loc[res["best"]]
        k1.metric("Modelo destacado",res["best"])
        k2.metric("Exactitud",f"{best_row['Exactitud']:.1%}")
        k3.metric("F1 macro",f"{best_row['F1 macro']:.1%}")
        k4.metric("Ventanas de prueba",f"{len(res['test']):,}")
        st.markdown("#### Matriz de confusión del modelo destacado")
        labels=res["labels"]; cm=pd.DataFrame(res["confusion"],index=label_names(labels),columns=label_names(labels))
        fig=px.imshow(cm,text_auto=True,color_continuous_scale="Teal",aspect="auto",labels=dict(x="Predicción",y="Respuesta conocida",color="Ventanas"))
        fig.update_layout(height=520,margin=dict(l=8,r=8,t=24,b=8),xaxis_tickangle=-35)
        st.plotly_chart(fig,use_container_width=True)
        with st.expander("¿Cómo se lee esta matriz?"):
            st.write("Las filas son la condición real conocida del experimento y las columnas son lo que predijo el modelo. Los números de la diagonal son aciertos; los que quedan fuera muestran confusiones entre estados.")
        st.markdown("### Resultados no supervisados · K-means")
        u1,u2,u3=st.columns(3)
        u1.metric("Grupos buscados",8,"uno por cada escenario esperado")
        u2.metric("Silhouette",f"{uns['silhouette']:.3f}",help="Entre -1 y 1. Resume si los grupos se ven compactos y separados; no es exactitud.")
        u3.metric("ARI frente a etiquetas",f"{uns['ari']:.3f}" if uns["ari"] is not None else "N/D",help="Comparación posterior entre grupos y etiquetas conocidas. Las etiquetas no se usaron para formar los grupos.")
        f2=px.scatter(uns["plot"],x="Componente 1",y="Componente 2",color="Grupo K-means",symbol="Etiqueta conocida",hover_data=["Experimento","Fase"],title="Fragmentos agrupados (PCA solo para visualizar)")
        f2.update_layout(height=500)
        st.plotly_chart(f2,use_container_width=True)
        st.caption(f"Los dos ejes del gráfico conservan {uns['pca_variance']:.1%} de la variación de características; es un resumen 2D. Silhouette no es porcentaje de aciertos; ARI solo compara los grupos con las etiquetas después del ajuste.")
        st.markdown("#### Grupo y clase más frecuente después de comparar")
        st.dataframe(uns["mapping"],use_container_width=True)
        c1,c2=st.columns(2)
        if st.session_state.get("saved_name"):
            p=ARTIFACT_DIR / st.session_state["saved_name"]
            if p.exists(): c1.download_button("Descargar resumen de este experimento",p.read_bytes(),file_name=p.name,mime="application/json")
        c2.download_button("Descargar predicciones detalladas",data=pd.DataFrame({"archivo":res["test"]["origen"].to_numpy(),"ventana":res["test"]["ventana"].to_numpy(),"real":res["y_test"].map(FAULTS_ES).to_numpy(),"predicho":[FAULTS_ES.get(int(y),"Desconocido") for y in res["pred"]]}).to_csv(index=False).encode("utf-8-sig"),file_name="ElectroPV_predicciones.csv",mime="text/csv")
    st.markdown("---")
    st.markdown("#### ¿Qué significa L y M?")
    st.write("L es el modo de potencia limitada (IPPT); M es el seguimiento del punto de máxima potencia (MPPT). Se evalúa cruzando modos para que la prueba use registros distintos a los del entrenamiento.")

elif page=="Probar mis CSV":
    st.title("Probar mis CSV")
    st.write("Carga uno o varios archivos. Cada archivo se analiza por separado; subir un caso nuevo no modifica los resultados anteriores ni los CSV incluidos.")
    files=st.file_uploader("Selecciona archivos CSV",type=["csv"],accept_multiple_files=True,help="Acepta CSV de características descargados desde Inicio o CSV crudos del dataset GPVS-Faults con sus 14 columnas originales.")
    modality=st.selectbox("Usar señales",["Corriente + voltaje","Corriente","Voltaje"],key="inference_modality")
    train_mode=st.radio("Entrenar el modelo base con",["L","M"],horizontal=True,format_func=lambda x:"L · potencia limitada" if x=="L" else "M · potencia máxima")
    if files and st.button("Analizar archivos",type="primary"):
        with st.spinner("Preparando el modelo y analizando cada archivo…"):
            full=get_data(); train=full[full["modo"]==train_mode]; cols=feature_columns(modality)
            from sklearn.ensemble import RandomForestClassifier
            model=RandomForestClassifier(n_estimators=180,max_features="sqrt",class_weight="balanced_subsample",random_state=42,n_jobs=-1)
            model.fit(train[cols].fillna(0),train["etiqueta"].astype(int))
            all_results=[]
            for f in files:
                try:
                    pdata,kind=read_uploaded_csv(f)
                    pred=model.predict(pdata[cols].replace([float("inf"),-float("inf")],float("nan")).fillna(0)).astype(int)
                    pdata["prediccion"]=[FAULTS_ES.get(int(y),"Estado no identificado") for y in pred]
                    counts=pd.Series(pdata["prediccion"]).value_counts(normalize=True).mul(100).round(1)
                    label=counts.index[0]
                    all_results.append({"nombre":f.name,"tipo":kind,"n":len(pdata),"label":label,"porcentajes":counts,"data":pdata})
                except Exception as e:
                    st.error(f"{f.name}: {e}")
            st.session_state["uploads_results"]=all_results
    for result in st.session_state.get("uploads_results",[]):
        with st.container(border=True):
            st.markdown(f"### {result['nombre']}")
            st.caption(f"Entrada: {result['tipo']} · {result['n']} fragmentos analizados")
            st.success(f"Estado más frecuente: **{result['label']}**")
            chart=result["porcentajes"].rename_axis("Estado estimado").reset_index(name="Porcentaje")
            st.bar_chart(chart.set_index("Estado estimado"),y="Porcentaje",color="#159274")
            st.caption("La barra resume las predicciones por fragmento de este archivo. Es una salida del modelo, no una certificación del estado físico del equipo.")
            st.download_button(f"Descargar predicciones de {result['nombre']}",result["data"].to_csv(index=False).encode("utf-8-sig"),file_name=f"predicciones_{result['nombre']}",mime="text/csv",key=f"download_{result['nombre']}")

elif page=="Entender el proyecto":
    st.title("Entender el proyecto")
    st.markdown("### En palabras sencillas")
    st.write("El sistema solar produce corriente continua en los paneles. Un inversor la convierte en corriente alterna para conectarse a la red. Sensores registran cómo cambian corrientes y voltajes. Cuando se introduce una falla, algunas señales cambian. ElectroPV convierte pequeñas partes de esas señales en números y estudia esos patrones.")
    with st.expander("¿Qué es multimodal aquí?",expanded=True):
        st.write("Multimodal significa usar más de un tipo de medición. En este caso usamos **corriente** y **voltaje**. Corriente sola y voltaje solo son pruebas comparables; al combinarlas el modelo recibe ambas fuentes de información.")
    with st.expander("¿Qué hace el método supervisado?"):
        st.write("Aprende usando ejemplos con respuesta conocida. Por ejemplo, los datos F2 después de introducir la falla se etiquetan como falla del inversor. Luego el modelo predice la clase de fragmentos de otros registros. Se usan un árbol, Random Forest y regresión logística.")
    with st.expander("¿Qué hace K-means?"):
        st.write("K-means recibe los números, pero no los nombres de las fallas. Separa los fragmentos en grupos por semejanza. Solo después se compara qué etiquetas conocidas hay en cada grupo. Por eso silhouette no es exactitud y ARI es una medida de concordancia posterior.")
    with st.expander("¿Cómo se preparan los datos?"):
        st.write("Se excluye la columna Time como predictor, se trabaja con las 13 señales eléctricas, se cortan en ventanas consecutivas de 1.024 muestras y por señal se calculan media, RMS, desviación estándar, curtosis y factor de cresta. Resultan 65 características por ventana. Para F1–F7, las ventanas anteriores al punto medio se consideran normales y las posteriores conservan la etiqueta de la falla del archivo. F0 es normal durante todo el registro.")
    with st.expander("¿Cómo se evita una prueba engañosa?"):
        st.write("No se mezclan ventanas del mismo CSV entre entrenamiento y prueba. Se entrena con todos los registros de un modo (L o M) y se prueba con los del otro. Es una prueba más exigente de cambio de modo, pero los datos son de una misma bancada experimental: no demuestra por sí sola que el modelo funcione en cualquier planta real.")
    with st.expander("¿Qué miden las métricas?"):
        st.markdown("- **Exactitud:** fracción total de ventanas clasificadas correctamente.\n- **Exactitud balanceada:** promedia el acierto por clase para que ninguna clase pese solo por tener más datos.\n- **F1 macro:** combina precisión y recuperación y luego promedia las clases por igual.\n- **Silhouette:** resume qué tan compactos y separados quedaron los grupos; puede ser negativo o llegar a 1.\n- **ARI:** compara después los grupos encontrados contra las etiquetas conocidas; 1 indica coincidencia, 0 parecido al azar y valores negativos peor que azar.")
    st.markdown("### Estados del conjunto")
    st.dataframe(pd.DataFrame([{"Código":k,"Estado":v} for k,v in FAULT_NAMES.items()]),hide_index=True,use_container_width=True)
    st.warning("El punto de falla se aproxima usando la mitad del archivo porque el conjunto indica que la falla se introduce a mitad de cada experimento. Para un CSV nuevo desconocido, las predicciones son exploratorias; la aplicación no conoce la verdad del archivo.")

