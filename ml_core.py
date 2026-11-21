"""Procesamiento y modelos para ElectroPV (GPVS-Faults)."""
from __future__ import annotations

import io
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score,
                             balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, silhouette_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "csv_pruebas"
PANEL_DATA_DIR = ROOT / "data" / "panel_cases"
ARTIFACT_DIR = ROOT / "artifacts"
WINDOW = 1024
CHANNELS = ["Ipv", "Vpv", "Vdc", "ia", "ib", "ic", "va", "vb", "vc", "Iabc", "If", "Vabc", "Vf"]
CURRENT = {"Ipv", "ia", "ib", "ic", "Iabc", "If"}
VOLTAGE = {"Vpv", "Vdc", "va", "vb", "vc", "Vabc", "Vf"}
FAULTS = {0: "Normal", 1: "Falla en el arreglo fotovoltaico", 2: "Falla en el inversor",
          3: "Anomalía de red", 4: "Falla del sensor de realimentación",
          5: "Falla de conexión a red", 6: "Falla del controlador MPPT",
          7: "Falla del convertidor elevador (boost)"}
FAULTS_ES = {0: "Normal", 1: "Arreglo fotovoltaico", 2: "Inversor", 3: "Red eléctrica",
             4: "Sensor de realimentación", 5: "Conexión a red", 6: "Control MPPT", 7: "Convertidor boost"}
PANEL_CONDITIONS = ["Limpio", "Sucio", "Sombreado"]


def feature_columns(modality: str) -> list[str]:
    allowed = CURRENT if modality == "Corriente" else VOLTAGE if modality == "Voltaje" else CURRENT | VOLTAGE
    return [c for c in feature_names() if c.split("_")[0] in allowed]


def feature_names() -> list[str]:
    return [f"{ch}_{stat}" for ch in CHANNELS for stat in ("media", "rms", "desv", "curtosis", "factor_cresta")]


def load_builtin() -> pd.DataFrame:
    frames = []
    for file in sorted(DATA_DIR.glob("F[0-7][LM].csv")):
        part = pd.read_csv(file)
        part["origen"] = file.stem
        frames.append(part)
    if not frames:
        raise FileNotFoundError("No se encontraron los 16 CSV de demostración.")
    return pd.concat(frames, ignore_index=True)


def load_panel_cases() -> pd.DataFrame:
    files=sorted(PANEL_DATA_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No se encontraron los CSV de los seis paneles.")
    return pd.concat([pd.read_csv(p) for p in files],ignore_index=True)


def panel_feature_columns(modality: str) -> list[str]:
    stats=("media","rms","desv","min","max","curtosis")
    electrical=[f"{s}_{t}" for s in ("corriente","voltaje","potencia") for t in stats]+["irradiancia","temperatura_modulo"]
    thermal=["temp_media","temp_desv","temp_max","temp_p95"]
    cols=electrical+thermal
    if modality=="Eléctricas": return electrical
    if modality=="Térmica": return thermal
    return cols


def panel_case_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.groupby(["panel_id","modelo","condicion"],as_index=False)[cols].mean()


def evaluate_panels_supervised(df: pd.DataFrame, modality: str) -> dict:
    from sklearn.base import clone
    cols=panel_feature_columns(modality)
    if not cols: raise ValueError("No hay características para esa modalidad.")
    models={
        "Árbol de decisión":DecisionTreeClassifier(max_depth=10,class_weight="balanced",random_state=42),
        "Random Forest":RandomForestClassifier(n_estimators=180,max_features="sqrt",class_weight="balanced_subsample",random_state=42,n_jobs=-1),
        "Regresión logística":make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42))
    }
    scores=[]; details={}; labels=PANEL_CONDITIONS
    for name,base in models.items():
        predictions=[]
        for held_panel in sorted(df["panel_id"].unique()):
            train=df[df["panel_id"]!=held_panel]; test=df[df["panel_id"]==held_panel]
            model=clone(base); model.fit(train[cols].replace([np.inf,-np.inf],np.nan).fillna(0),train["condicion"])
            test=test.copy(); test["_pred"]=model.predict(test[cols].replace([np.inf,-np.inf],np.nan).fillna(0))
            for (pid,mod,cond),grp in test.groupby(["panel_id","modelo","condicion"]):
                guess=grp["_pred"].mode().iloc[0]
                predictions.append({"panel_id":pid,"modelo":mod,"real":cond,"predicho":guess,"ventanas":len(grp)})
        pred=pd.DataFrame(predictions)
        scores.append({"Modelo":name,"Exactitud":accuracy_score(pred["real"],pred["predicho"]),
                       "Exactitud balanceada":balanced_accuracy_score(pred["real"],pred["predicho"]),
                       "F1 macro":f1_score(pred["real"],pred["predicho"],labels=labels,average="macro",zero_division=0)})
        details[name]=pred
    table=pd.DataFrame(scores); best=table.sort_values("F1 macro",ascending=False).iloc[0]["Modelo"]
    bestpred=details[best]
    cm=confusion_matrix(bestpred["real"],bestpred["predicho"],labels=labels)
    return {"tabla":table,"best":best,"predictions":bestpred,"confusion":cm,"labels":labels,"cols":cols}


def evaluate_panels_unsupervised(df: pd.DataFrame, modality: str) -> dict:
    cols=panel_feature_columns(modality)
    cases=panel_case_table(df,cols)
    x=cases[cols].replace([np.inf,-np.inf],np.nan).fillna(0)
    scaler=StandardScaler(); xs=scaler.fit_transform(x)
    k=3; km=KMeans(n_clusters=k,random_state=42,n_init=10); cluster=km.fit_predict(xs)
    sil=silhouette_score(xs,cluster) if len(np.unique(cluster))>1 else float("nan")
    ari=adjusted_rand_score(cases["condicion"],cluster)
    xy=PCA(n_components=2,random_state=42).fit_transform(xs)
    plot=cases[["panel_id","modelo","condicion"]].copy()
    plot["Componente 1"]=xy[:,0]; plot["Componente 2"]=xy[:,1]; plot["Grupo K-means"]=cluster.astype(str)
    mapping=plot.groupby("Grupo K-means")["condicion"].agg(lambda s:s.mode().iloc[0]).rename("Condición más común después de comparar").reset_index()
    variance=float(PCA(n_components=2,random_state=42).fit(xs).explained_variance_ratio_.sum())
    return {"cases":cases,"plot":plot,"mapping":mapping,"silhouette":float(sil),"ari":float(ari),"pca_variance":variance,"model":km,"scaler":scaler}


def save_panel_experiment(metadata: dict, metrics: pd.DataFrame) -> Path:
    import json
    ARTIFACT_DIR.mkdir(parents=True,exist_ok=True)
    stamp=time.strftime("%Y%m%d_%H%M%S")
    payload={**metadata,"metricas_supervisadas":metrics.to_dict(orient="records")}
    out=ARTIFACT_DIR/f"paneles_{stamp}.json"
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    return out


def raw_to_features(raw: pd.DataFrame, filename: str = "archivo.csv") -> pd.DataFrame:
    """Convierte señales crudas al mismo formato de ventanas usado por los ejemplos."""
    aliases = {str(c).strip().lower(): c for c in raw.columns}
    missing = [c for c in ["Time", *CHANNELS] if c.lower() not in aliases]
    if missing:
        raise ValueError("El CSV crudo debe tener columnas Time, Ipv, Vpv, Vdc, ia, ib, ic, va, vb, vc, Iabc, If, Vabc y Vf.")
    values = raw[[aliases[c.lower()] for c in CHANNELS]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < WINDOW:
        raise ValueError(f"Se requieren al menos {WINDOW} filas válidas; se recibieron {len(values)}.")
    # Las filas iniciales se consideran fase previa. En los datos GPVS la avería
    # se inyecta aproximadamente a mitad del registro; en un archivo nuevo no
    # se asigna etiqueta real, solo se estiman ventanas para inferencia.
    out = []
    for start in range(0, len(values) - WINDOW + 1, WINDOW):
        x = values[start:start + WINDOW]
        feats = []
        for j in range(len(CHANNELS)):
            s = x[:, j]; mean = float(s.mean()); rms = float(np.sqrt(np.mean(s * s)))
            sd = float(s.std()); kurt = float(np.mean(((s - mean) / sd) ** 4)) if sd > 1e-12 else 0.0
            crest = float(np.max(np.abs(s)) / rms) if rms > 1e-12 else 0.0
            feats.extend([mean, rms, sd, kurt, crest])
        out.append(feats)
    result = pd.DataFrame(out, columns=feature_names())
    result["archivo_origen"] = Path(filename).stem
    result["modo"] = "Subido"
    result["ventana"] = np.arange(len(result))
    result["fase"] = "Sin etiqueta conocida"
    result["etiqueta"] = -1
    result["origen"] = Path(filename).stem
    return result


def read_uploaded_csv(uploaded) -> tuple[pd.DataFrame, str]:
    data = uploaded.getvalue()
    name = uploaded.name
    df = pd.read_csv(io.BytesIO(data))
    if set(feature_names()).issubset(df.columns):
        df["origen"] = name.rsplit(".", 1)[0]
        return df, "características"
    return raw_to_features(df, name), "señales crudas procesadas"


def get_split(df: pd.DataFrame, train_mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_mode = "M" if train_mode == "L" else "L"
    train = df[(df["modo"] == train_mode) & (df["etiqueta"] >= 0)].copy()
    test = df[(df["modo"] == test_mode) & (df["etiqueta"] >= 0)].copy()
    return train, test


def evaluate_supervised(df: pd.DataFrame, modality: str, train_mode: str) -> dict:
    train, test = get_split(df, train_mode)
    cols = feature_columns(modality)
    if train.empty or test.empty:
        raise ValueError("Faltan registros para separar los dos modos de operación.")
    x_train = train[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    x_test = test[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train["etiqueta"].astype(int); y_test = test["etiqueta"].astype(int)
    available = sorted(y_train.unique())
    models = {
        "Árbol de decisión": DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=180, max_features="sqrt", class_weight="balanced_subsample", random_state=42, n_jobs=-1),
        "Regresión logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42))
    }
    scores = []; fitted = {}; predictions = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test).astype(int)
        predictions[name] = pred
        fitted[name] = model
        scores.append({"Modelo": name, "Exactitud": accuracy_score(y_test, pred),
                       "Exactitud balanceada": balanced_accuracy_score(y_test, pred),
                       "F1 macro": f1_score(y_test, pred, average="macro", zero_division=0)})
    best = max(scores, key=lambda x: x["F1 macro"])
    name = best["Modelo"]; pred = predictions[name]
    labels = sorted(set(y_test) | set(available))
    report = classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0)
    return {"tabla": pd.DataFrame(scores), "train": train, "test": test, "cols": cols,
            "y_test": y_test, "pred": pred, "best": name, "model": fitted[name],
            "labels": labels, "confusion": confusion_matrix(y_test, pred, labels=labels), "report": report}


def evaluate_unsupervised(df: pd.DataFrame, modality: str, train_mode: str, k: int = 8) -> dict:
    _, test = get_split(df, train_mode)
    cols = feature_columns(modality)
    x = test[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = StandardScaler(); xs = scaler.fit_transform(x)
    k = max(2, min(int(k), len(test) - 1))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = km.fit_predict(xs)
    silhouette = silhouette_score(xs, clusters, sample_size=min(1500, len(xs)), random_state=42) if len(np.unique(clusters)) > 1 else float("nan")
    known = test["etiqueta"].astype(int).to_numpy()
    ari = adjusted_rand_score(known, clusters) if (known >= 0).any() else None
    pca = PCA(n_components=2, random_state=42); xy = pca.fit_transform(xs)
    plot = pd.DataFrame({"Componente 1":xy[:,0],"Componente 2":xy[:,1],"Grupo K-means":clusters.astype(str),
                         "Etiqueta conocida":test["etiqueta"].map(FAULTS_ES).fillna("Sin etiqueta").to_numpy(),
                         "Experimento":test["origen"].to_numpy(),"Fase":test["fase"].to_numpy()})
    mapping = plot.assign(_label=known).groupby("Grupo K-means")[["_label"]].agg(lambda s: int(pd.Series(s).mode().iloc[0]))
    mapping["Grupo se parece más a"] = mapping["_label"].map(FAULTS_ES)
    return {"test":test,"clusters":clusters,"silhouette":silhouette,"ari":ari,"plot":plot,
            "pca_variance":float(pca.explained_variance_ratio_.sum()),"mapping":mapping.drop(columns="_label"),"model":km,"scaler":scaler}


def save_experiment(metadata: dict, result: dict) -> Path:
    import json
    import joblib
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    payload = dict(metadata)
    payload["métricas"] = result["tabla"].to_dict(orient="records")
    out = ARTIFACT_DIR / f"experimento_{stamp}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(result["model"], ARTIFACT_DIR / f"modelo_{stamp}.joblib")
    return out

