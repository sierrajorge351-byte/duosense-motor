"""Create compact, panel-specific CSV examples from the ITM Medellin data."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "panels_source" / "Experimental data for Mismatching Faults of Photovoltaic Modules" / "RawData"
OUT = ROOT / "data" / "panel_cases"
WINDOW = 4096
OUT.mkdir(parents=True, exist_ok=True)

def thermal_features(path: Path) -> dict:
    try:
        t = pd.read_csv(path, encoding="utf-16", skiprows=4, header=0)
        a = t.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        a = a[np.isfinite(a)]
    except Exception:
        a = np.array([], dtype=float)
    if not len(a):
        return {"temp_media": np.nan, "temp_desv": np.nan, "temp_max": np.nan, "temp_p95": np.nan}
    return {"temp_media": float(a.mean()), "temp_desv": float(a.std()),
            "temp_max": float(a.max()), "temp_p95": float(np.percentile(a, 95))}

def stats(x: np.ndarray, prefix: str) -> dict:
    x = np.asarray(x, dtype=float)
    mean = float(x.mean()); sd = float(x.std()); rms = float(np.sqrt(np.mean(x*x)))
    kurt = float(np.mean(((x-mean)/sd)**4)) if sd > 1e-12 else 0.0
    return {f"{prefix}_media":mean, f"{prefix}_rms":rms, f"{prefix}_desv":sd,
            f"{prefix}_min":float(x.min()), f"{prefix}_max":float(x.max()),
            f"{prefix}_curtosis":kurt}

electrical = list(DATA.glob("*/ElectricalData/*.mat"))
all_rows=[]
for mat_path in sorted(electrical):
    model = mat_path.parents[1].name
    stem = mat_path.stem
    m = re.match(r"(Panel[A-C]|Panel[1-3])_(Clean|Dirt|Shadow)$", stem, re.I)
    if not m: continue
    panel, raw_cond = m.groups()
    cond = {"clean":"Limpio", "dirt":"Sucio", "shadow":"Sombreado"}[raw_cond.lower()]
    panel_id = f"{model}_{panel}"
    d=loadmat(mat_path)
    current=np.ravel(d["Ipanel"]).astype(float)
    voltage=np.ravel(d["Vpanel"]).astype(float)
    power=current*voltage
    thermal_path=mat_path.parents[1]/"ThermalData"/(stem+".csv")
    th=thermal_features(thermal_path)
    irr=float(np.ravel(d.get("IRR",[np.nan]))[0])
    module_temp=float(np.ravel(d.get("T",[np.nan]))[0])
    records=[]
    n=min(len(current),len(voltage))
    for start in range(0,n-WINDOW+1,WINDOW):
        sl=slice(start,start+WINDOW)
        rec={"panel_id":panel_id,"modelo":model,"condicion":cond,"ventana":len(records),
             "irradiancia":irr,"temperatura_modulo":module_temp,**th}
        rec.update(stats(current[sl],"corriente")); rec.update(stats(voltage[sl],"voltaje")); rec.update(stats(power[sl],"potencia"))
        records.append(rec)
    frame=pd.DataFrame(records)
    safe=panel_id.replace("-","_")
    label={"Limpio":"limpio","Sucio":"sucio","Sombreado":"sombreado"}[cond]
    out=OUT/f"{safe}_{label}.csv"
    frame.to_csv(out,index=False)
    all_rows.extend(records)
    print(f"{out.name}: {len(frame)} ventanas · {len(current):,} muestras · thermal={'sí' if np.isfinite(th['temp_media']) else 'no'}")

summary=pd.DataFrame(all_rows)
summary.to_csv(ROOT/"data"/"panel_cases_all.csv",index=False)
print(f"TOTAL CSV: {summary['panel_id'].nunique()} paneles; {summary.groupby(['panel_id','condicion']).ngroups} panel-condición; {len(summary):,} ventanas")

