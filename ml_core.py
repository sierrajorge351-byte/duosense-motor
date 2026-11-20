"""Multimodal motor-fault lab: synchronized vibration and current modalities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, adjusted_rand_score, classification_report,
                             confusion_matrix, f1_score, silhouette_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from pathlib import Path

FS = 64000
SAMPLES = 16384
CLASSES = ["Saludable", "Daño real de rodamiento"]
FEATURES = [
    "vib_rms", "vib_kurtosis", "vib_crest", "vib_dom_hz", "vib_hf_ratio",
    "current_rms", "current_kurtosis", "current_crest", "current_dom_hz", "current_sideband_ratio",
]


def validate_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate user-supplied feature CSV data before model fitting."""
    required = FEATURES + ["estado", "recording"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Faltan columnas requeridas: " + ", ".join(missing))
    result = frame.copy()
    for column in FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[required].isna().any().any() or not np.isfinite(result[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Hay valores vacíos, infinitos o no numéricos en las características o etiquetas.")
    result["estado"] = result["estado"].astype(str)
    result["recording"] = result["recording"].astype(str)
    if set(result["estado"]) != set(CLASSES):
        raise ValueError("'estado' debe contener exactamente las clases: " + " y ".join(CLASSES))
    groups_per_class = result.groupby("estado")["recording"].nunique()
    if groups_per_class.min() < 4:
        raise ValueError("Incluye al menos cuatro archivos de medición (recording) por clase.")
    return result


def _features(signal: np.ndarray, prefix: str, fs: int) -> dict[str, float]:
    x = np.asarray(signal, dtype=float)
    x = x - np.mean(x)
    rms = float(np.sqrt(np.mean(x*x)))
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1/fs)
    power = spectrum**2
    dominant = float(freqs[np.argmax(spectrum[1:]) + 1]) if len(spectrum) > 1 else 0.0
    result = {
        f"{prefix}_rms": rms,
        f"{prefix}_kurtosis": float(kurtosis(x, fisher=False, bias=False)),
        f"{prefix}_crest": float(np.max(np.abs(x)) / (rms + 1e-12)),
        f"{prefix}_dom_hz": dominant,
    }
    if prefix == "vib":
        result["vib_hf_ratio"] = float(power[freqs >= 1000].sum() / (power.sum() + 1e-12))
    else:
        # Sideband energy away from the 50 Hz supply frequency in this German test setup.
        side = ((freqs >= 10) & (freqs <= 45)) | ((freqs >= 55) & (freqs <= 100))
        result["current_sideband_ratio"] = float(power[side].sum() / (power.sum() + 1e-12))
    return result


def build_dataset(data_dir: Path | None = None):
    """Load the compact feature extract and example traces from Paderborn real data."""
    data_dir = data_dir or Path(__file__).resolve().parent / "data"
    frame = pd.read_csv(data_dir / "paderborn_real_features.csv")
    pack = np.load(data_dir / "paderborn_real_examples.npz")
    examples = {}
    for label in CLASSES:
        examples[label] = {
            "vibration": pack[f"{label}_vibration"],
            "current": pack[f"{label}_current"],
            "t": pack[f"{label}_t"],
        }
    return frame, examples


def train_supervised(data: pd.DataFrame, trees: int = 250):
    x, y = data[FEATURES], data["estado"]
    splitter = GroupShuffleSplit(n_splits=100, test_size=0.25, random_state=42)
    train_idx = test_idx = None
    for candidate_train, candidate_test in splitter.split(x, y, groups=data["recording"]):
        if set(y.iloc[candidate_train]) == set(CLASSES) and set(y.iloc[candidate_test]) == set(CLASSES):
            train_idx, test_idx = candidate_train, candidate_test
            break
    if train_idx is None or test_idx is None:
        raise ValueError("No se pudo separar entrenamiento y prueba por archivos manteniendo ambas clases.")
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    idx_test = data.index[test_idx]
    # One worker keeps the local app responsive on laptops with many logical cores.
    model = RandomForestClassifier(n_estimators=trees, class_weight="balanced", random_state=42,
                                   min_samples_leaf=2, n_jobs=1)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    return model, (x_test, y_test, predicted, idx_test), {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "f1_macro": float(f1_score(y_test, predicted, average="macro")),
        "report": classification_report(y_test, predicted, labels=CLASSES, output_dict=True, zero_division=0),
        "matrix": confusion_matrix(y_test, predicted, labels=CLASSES),
        "train_size": int(len(x_train)), "test_size": int(len(x_test)),
        "train_recordings": int(data.iloc[train_idx]["recording"].nunique()),
        "test_recordings": int(data.iloc[test_idx]["recording"].nunique()),
    }


def cluster_unsupervised(data: pd.DataFrame, k: int = 2):
    """Fit clustering using sensor features only; labels are consulted only for external evaluation."""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(data[FEATURES])
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    cluster = model.fit_predict(x_scaled)
    pca = PCA(n_components=2, random_state=42)
    projection = pca.fit_transform(x_scaled)
    return cluster, projection, {
        "silhouette": float(silhouette_score(x_scaled, cluster)),
        "ari_external": float(adjusted_rand_score(data["estado"], cluster)),
        "counts": pd.Series(cluster).value_counts().sort_index().to_dict(),
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
    }


def elbow_curve(data: pd.DataFrame, max_k: int = 8):
    """Return exploratory K-means inertia values after feature standardization."""
    x_scaled = StandardScaler().fit_transform(data[FEATURES])
    upper = max(1, min(int(max_k), len(data) - 1))
    values = []
    for k in range(1, upper + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=200)
        model.fit(x_scaled)
        values.append(float(model.inertia_))
    return list(range(1, upper + 1)), values


def filter_signal(values: np.ndarray, fs: int = FS, cutoff: float = 5000.0) -> np.ndarray:
    sos = butter(4, cutoff, btype="lowpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(values, dtype=float))

