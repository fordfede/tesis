"""# Job 4 - Modelo Completo: Variables Macro + RWA ↔ Estrés Financiero"""

# ════════════════════════════════════════════════════════════
# 1. Configuración
# ════════════════════════════════════════════════════════════

DEBUG: bool = True
SAVE_DATA: bool = True

# ════════════════════════════════════════════════════════════
# 2. Imports
# ════════════════════════════════════════════════════════════

import contextlib
import datetime
import io
import os
import time
import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR  # CAMBIO I5-1 (Job 8)
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller  # CAMBIO I5-6 (Job 8)
from statsmodels.tsa.stattools import grangercausalitytests

from google.colab import drive
drive.mount('/content/drive')

# ════════════════════════════════════════════════════════════
# 3. Paths y Constantes
# ════════════════════════════════════════════════════════════

# CAMBIO H2-5 (Job 8): ver nota en Job 0 (mismo patrón).
BASE_DIR: str = os.environ.get(
    "TESIS_BASE_DIR", "/content/drive/MyDrive/Tesis/expansion"
)
DIR_PROCESSED: str = os.path.join(BASE_DIR, "data", "processed")
DIR_REPORTES: str = os.path.join(
    BASE_DIR, "outputs", "reportes"
)
DIR_MODELOS: str = os.path.join(BASE_DIR, "outputs", "modelos")

ARCHIVO_LARGAS_EXT: str = os.path.join(
    DIR_PROCESSED, "series_largas_ext.csv"
)
ARCHIVO_COMPLETAS: str = os.path.join(
    DIR_PROCESSED, "series_completas.csv"
)
ARCHIVO_GRANGER: str = os.path.join(
    DIR_REPORTES, "granger_completo.csv"
)
# CAMBIO I5-3 (Job 8): tabla de apoyo por rezago (fuera de BH).
ARCHIVO_GRANGER_REZAGO: str = os.path.join(
    DIR_REPORTES, "granger_por_rezago.csv"
)
ARCHIVO_MET: str = os.path.join(
    DIR_REPORTES, "metricas_completo.csv"
)
ARCHIVO_TOP: str = os.path.join(
    DIR_REPORTES, "top_señales.csv"
)

# ── FRED ────────────────────────────────────────────────────

FRED_OBS: str = (
    "https://api.stlouisfed.org/fred/series/observations"
)
TIMEOUT: int = 30
PAUSA: float = 1.5
HISTORIA_INICIO: str = "2021-01-01"
HOY: str = datetime.date.today().isoformat()

# CAMBIO H1-1 (Job 8): se elimina la clave hardcodeada. Se lee
# de userdata/variable de entorno y se corta con error claro si
# no está configurada (antes quedaba un valor sustituto fijo).
try:
    from google.colab import userdata as _ud  # type: ignore
    FRED_API_KEY: str = _ud.get("FRED_API_KEY") or ""
except Exception:
    FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
# CAMBIO B1-3 (Bloque 1): en modo snapshot no se descarga FRED,
# así que la clave no es obligatoria (MODO_DATOS se define después).
if not FRED_API_KEY and os.environ.get(
    "TESIS_MODO_DATOS", "descarga"
) != "snapshot":
    raise RuntimeError(
        "[DEBUG][ALERTA] FRED_API_KEY no configurada: definila "
        "en google.colab.userdata o en la variable de entorno "
        "FRED_API_KEY antes de correr este Job."
    )

# Series FRED a descargar: {series_id: nombre_columna}
FRED_SERIES: dict[str, str] = {
    "DFF": "macro_fed_funds",
    "SOFR": "macro_sofr",
    "DGS1MO": "macro_tsy_1m",
    "DGS3MO": "macro_tsy_3m",
    "DGS2": "macro_tsy_2y",
    "DGS10": "macro_tsy_10y",
    "T10Y2Y": "macro_spread_10y2y",
    "T10YIE": "macro_infl_exp_10y",
    "RRPONTSYD": "macro_rrp_overnight",
    "NFCI": "macro_nfci",
    "BAMLH0A0HYM2": "macro_spread_hy",
    "BAMLC0A0CM": "macro_spread_ig",
    "INDPRO": "macro_indpro",
    "UMCSENT": "macro_umcsent",
}

# ── Columnas existentes en series_largas_ext ────────────────

COL_USDY: str = "rwa_usdy_supply"
COLS_RWA: list[str] = [
    "rwa_red_stellar",
    "rwa_red_ripple",
    "rwa_red_algorand",
    "rwa_tipo_treasurybills",
    "rwa_tipo_commodities",
    "rwa_tipo_privatecredit",
    "rwa_tipo_stocksetfs",
    "rwa_total_global",
]
COLS_RWA_ALL: list[str] = [COL_USDY] + COLS_RWA

COL_CISS: str = "macro_euro_ciss"
COL_STLFSI: str = "macro_eeuu_stlfsi"

# ── Transformaciones por columna ────────────────────────────
# "log_diff", "diff", "nivel"
TRANSFORM: dict[str, str] = {}

for _c in COLS_RWA_ALL:
    TRANSFORM[_c] = "log_diff"

TRANSFORM["macro_fed_funds"] = "diff"
TRANSFORM["macro_sofr"] = "diff"
TRANSFORM["macro_tsy_1m"] = "diff"
TRANSFORM["macro_tsy_3m"] = "diff"
TRANSFORM["macro_tsy_2y"] = "diff"
TRANSFORM["macro_tsy_10y"] = "diff"
TRANSFORM["macro_spread_10y2y"] = "diff"
TRANSFORM["macro_infl_exp_10y"] = "diff"
TRANSFORM["macro_rrp_overnight"] = "log_diff"
TRANSFORM["macro_nfci"] = "nivel"
TRANSFORM["macro_spread_hy"] = "diff"
TRANSFORM["macro_spread_ig"] = "diff"
TRANSFORM["macro_indpro"] = "log_diff"
TRANSFORM["macro_umcsent"] = "diff"
TRANSFORM[COL_CISS] = "diff"
TRANSFORM[COL_STLFSI] = "nivel"

# ── Features macro (para predecir RWA) ──────────────────────
COLS_MACRO_FEAT: list[str] = [
    "macro_fed_funds", "macro_sofr",
    "macro_tsy_1m", "macro_tsy_3m",
    "macro_tsy_2y", "macro_tsy_10y",
    "macro_spread_10y2y", "macro_infl_exp_10y",
    "macro_rrp_overnight", "macro_nfci",
    "macro_spread_hy", "macro_spread_ig",
    "macro_indpro", "macro_umcsent",
    COL_CISS, COL_STLFSI,
]

# CAMBIO C4-1 (Job 8): frecuencia nativa de cada columna macro.
# NFCI y STLFSI4 son semanales; INDPRO y UMCSENT son mensuales.
# Hoy se ffillean a diario y se tratan como si fueran diarias en
# todas las pistas, lo que en las que se diferencian (INDPRO,
# UMCSENT) produce casi todos ceros (el valor no cambia dentro
# del mes). Las columnas RWA/onchain no están acá: son diarias
# por construcción y frecuencia_admite_pista() las admite igual.
FRECUENCIA_NATIVA: dict[str, str] = {
    "macro_fed_funds": "D",
    "macro_sofr": "D",
    "macro_tsy_1m": "D",
    "macro_tsy_3m": "D",
    "macro_tsy_2y": "D",
    "macro_tsy_10y": "D",
    "macro_spread_10y2y": "D",
    "macro_infl_exp_10y": "D",
    "macro_rrp_overnight": "D",
    "macro_nfci": "W",
    "macro_spread_hy": "D",
    "macro_spread_ig": "D",
    "macro_indpro": "M",
    "macro_umcsent": "M",
    COL_CISS: "D",
    COL_STLFSI: "W",
}
_RANGO_FRECUENCIA: dict[str, int] = {"D": 0, "W": 1, "M": 2}
_PISTA_A_FRECUENCIA: dict[str, str] = {"A": "D", "B": "W", "M": "M"}

# Pista M (mensual): opcional, apagada por defecto. Única pista
# donde participan columnas de frecuencia nativa mensual.
INCLUIR_PISTA_MENSUAL: bool = False
LAGS_PISTA_M: int = 6

# CAMBIO I7-1 (Job 8): cola estancada (ver construir_dataset_unificado).
MAX_DIAS_ESTANCADO: int = 7
ARCHIVO_COLAS_ESTANCADAS: str = os.path.join(
    DIR_REPORTES, "colas_estancadas.csv"
)

# ── Granger ─────────────────────────────────────────────────
K_GRANGER_D: int = 14
K_GRANGER_S: int = 8
ALFA_BH: float = 0.05

# ── Modelo predictivo ──────────────────────────────────────
K_FEAT_A: int = 7
K_FEAT_B: int = 4
# CAMBIO H4 (documentación, sin cambio de comportamiento):
# Jobs 9/10 usan un único AR_LAGS_CONTROL=2 para ambas
# pistas, en vez de un valor por pista. No se unificó entre
# Jobs: el usuario confirmó dejarlo así y documentarlo.
AR_P_A: int = 3
AR_P_B: int = 2

# CAMBIO C3-1 (Job 8): grilla de Ridge (reemplaza ALPHAS).
# Lasso ya no usa grilla fija (alphas=None) — ver modelo_lasso.
RIDGE_ALPHAS: np.ndarray = np.logspace(-6, 3, 30)
UMBRAL_MEJORA_MATERIAL: float = 5.0
TOP_N_SEÑALES: int = 5

NOTA_RWA: str = (
    "RWA de adopción incipiente: log-dif dominada por "
    "crecimiento estructural, no por respuesta a estrés."
)

# ── CAMBIO B1-1 (Bloque 1): congelamiento de datos ─────────────
# MODO_DATOS="descarga" (default): los Jobs 1-4 descargan y, al
# guardar, copian sus CSV a DIR_SNAPSHOT y actualizan manifest.json.
# MODO_DATOS="snapshot": los Jobs 1-3 y los pasos 1-2 del Job 4 no
# descargan; restauran los CSV desde el snapshot verificando SHA-256.
# Los Jobs 5-14 verifican el hash de sus entradas en ambos modos.
# Bloque compartido: mismo texto en los Jobs 1-14.
import hashlib
import json
import shutil

MODO_DATOS: str = os.environ.get("TESIS_MODO_DATOS", "descarga")
if MODO_DATOS not in ("descarga", "snapshot"):
    raise ValueError(
        f"[DEBUG][ERROR] TESIS_MODO_DATOS inválido: {MODO_DATOS!r} "
        "(usar 'descarga' o 'snapshot')."
    )
BASE_DIR_SNAPSHOT: str = os.environ.get(
    "TESIS_BASE_DIR", "/content/drive/MyDrive/Tesis/expansion"
)
DIR_SNAPSHOT: str = os.environ.get(
    "TESIS_DIR_SNAPSHOT",
    os.path.join(BASE_DIR_SNAPSHOT, "data", "snapshot"),
)
ARCHIVO_MANIFEST: str = os.path.join(DIR_SNAPSHOT, "manifest.json")
print(f"[DEBUG] MODO_DATOS={MODO_DATOS} | snapshot: {DIR_SNAPSHOT}")


def sha256_archivo(ruta: str) -> str:
    """
    SHA-256 de un archivo, leído por bloques.

    CAMBIO B1-1 (Bloque 1).

    Args:
        ruta: Ruta del archivo.

    Returns:
        Hash hexadecimal.
    """
    _h = hashlib.sha256()
    with open(ruta, "rb") as _f:
        for _bloque in iter(lambda: _f.read(1 << 20), b""):
            _h.update(_bloque)
    return _h.hexdigest()


def leer_manifest() -> dict:
    """
    Lee manifest.json del snapshot ({"archivos": {}} si no existe).

    CAMBIO B1-1 (Bloque 1).

    Returns:
        Manifest como dict, con clave "archivos" por nombre de CSV.
    """
    if not os.path.isfile(ARCHIVO_MANIFEST):
        return {"archivos": {}}
    try:
        with open(ARCHIVO_MANIFEST, encoding="utf-8") as _f:
            _manifest = json.load(_f)
    except (OSError, ValueError) as e:
        raise RuntimeError(
            f"[DEBUG][ERROR] manifest ilegible: {ARCHIVO_MANIFEST}: {e}"
        ) from e
    _manifest.setdefault("archivos", {})
    return _manifest


def congelar_snapshot(rutas: list[str], job: str) -> None:
    """
    Copia CSV ya guardados a DIR_SNAPSHOT y registra en
    manifest.json: archivo, filas, columnas, fechas mín/máx y
    SHA-256. Si el archivo ya estaba congelado con otro hash, se
    reemplaza y se avisa con [DEBUG][ALERTA].

    CAMBIO B1-1 (Bloque 1).

    Args:
        rutas: CSV escritos en esta corrida.
        job: Job productor (ej. "Job 1").
    """
    os.makedirs(DIR_SNAPSHOT, exist_ok=True)
    _manifest = leer_manifest()
    for _ruta in rutas:
        _nombre = os.path.basename(_ruta)
        if not os.path.isfile(_ruta):
            print(
                f"[DEBUG][ALERTA] snapshot: no existe {_ruta}, "
                "no se congela"
            )
            continue
        try:
            _df = pd.read_csv(_ruta)
        except (pd.errors.EmptyDataError, ValueError, OSError) as e:
            print(f"[DEBUG][ALERTA] snapshot: no se pudo leer {_ruta}: {e}")
            _df = pd.DataFrame()
        _fecha_min: str | None = None
        _fecha_max: str | None = None
        if "fecha" in _df.columns:
            _fechas = pd.to_datetime(_df["fecha"], errors="coerce").dropna()
            if not _fechas.empty:
                _fecha_min = _fechas.min().date().isoformat()
                _fecha_max = _fechas.max().date().isoformat()
        _sha = sha256_archivo(_ruta)
        _previo = _manifest["archivos"].get(_nombre, {}).get("sha256")
        if _previo and _previo != _sha:
            print(
                f"[DEBUG][ALERTA] snapshot: {_nombre} reemplaza la "
                f"versión congelada ({_previo[:12]}... → {_sha[:12]}...)"
            )
        shutil.copy2(_ruta, os.path.join(DIR_SNAPSHOT, _nombre))
        _manifest["archivos"][_nombre] = {
            "job": job,
            "ruta_relativa": os.path.relpath(_ruta, BASE_DIR_SNAPSHOT),
            "filas": int(len(_df)),
            "columnas": [str(c) for c in _df.columns],
            "fecha_min": _fecha_min,
            "fecha_max": _fecha_max,
            "sha256": _sha,
            "congelado": datetime.datetime.now().isoformat(
                timespec="seconds"
            ),
        }
        print(
            f"[DEBUG] snapshot: congelado {_nombre} | {len(_df)} filas | "
            f"{_fecha_min} → {_fecha_max} | sha256 {_sha[:12]}..."
        )
    _tmp = ARCHIVO_MANIFEST + ".tmp"
    with open(_tmp, "w", encoding="utf-8") as _f:
        json.dump(_manifest, _f, indent=2, ensure_ascii=False)
    os.replace(_tmp, ARCHIVO_MANIFEST)


def restaurar_desde_snapshot(
    job: str, obligatorios: list[str]
) -> list[str]:
    """
    Modo snapshot: copia desde DIR_SNAPSHOT a su ruta original
    todos los CSV que el manifest atribuye a `job`, verificando
    antes el SHA-256. Aborta nombrando el archivo si falta en el
    manifest o en la carpeta, o si el hash no coincide.

    CAMBIO B1-1 (Bloque 1).

    Args:
        job: Job productor (ej. "Job 1").
        obligatorios: Rutas que deben estar sí o sí en el snapshot.

    Returns:
        Rutas restauradas.
    """
    _entradas = {
        _n: _e for _n, _e in leer_manifest()["archivos"].items()
        if _e.get("job") == job
    }
    for _ruta in obligatorios:
        if os.path.basename(_ruta) not in _entradas:
            raise FileNotFoundError(
                f"[DEBUG][ERROR] {job}: {os.path.basename(_ruta)} no "
                f"figura en {ARCHIVO_MANIFEST}. Correr el {job} con "
                "TESIS_MODO_DATOS=descarga para congelarlo."
            )
    if not _entradas:
        raise FileNotFoundError(
            f"[DEBUG][ERROR] {job}: ningún archivo del {job} en "
            f"{ARCHIVO_MANIFEST}."
        )
    _restaurados: list[str] = []
    for _nombre, _entrada in sorted(_entradas.items()):
        _origen = os.path.join(DIR_SNAPSHOT, _nombre)
        if not os.path.isfile(_origen):
            raise FileNotFoundError(
                f"[DEBUG][ERROR] {job}: falta {_origen} en el snapshot."
            )
        _sha = sha256_archivo(_origen)
        if _sha != _entrada.get("sha256"):
            raise RuntimeError(
                f"[DEBUG][ERROR] {job}: el hash de {_nombre} no "
                f"coincide con el manifest ({_sha[:12]}... vs "
                f"{str(_entrada.get('sha256'))[:12]}...)."
            )
        _destino = os.path.join(BASE_DIR_SNAPSHOT, _entrada["ruta_relativa"])
        os.makedirs(os.path.dirname(_destino), exist_ok=True)
        shutil.copy2(_origen, _destino)
        _restaurados.append(_destino)
        print(
            f"[DEBUG] snapshot: restaurado {_nombre} → {_destino} "
            f"(sha256 {_sha[:12]}... OK)"
        )
    return _restaurados


# ════════════════════════════════════════════════════════════
# 4. Funciones — Descarga FRED
# ════════════════════════════════════════════════════════════


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = TIMEOUT,
) -> Any:
    # CAMBIO B1-2 (Bloque 1): en modo snapshot no hay red.
    if MODO_DATOS == "snapshot":
        raise RuntimeError(
            "[DEBUG][ERROR] MODO_DATOS=snapshot: llamada de red "
            f"bloqueada ({url[:55]})."
        )
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        if DEBUG:
            print(f"[DEBUG] HTTP {r.status_code}: {url[:55]}")
        return None
    except requests.Timeout:
        if DEBUG:
            print(f"[DEBUG] timeout: {url[:55]}")
        return None
    except requests.RequestException as e:
        if DEBUG:
            print(f"[DEBUG] request error: {type(e).__name__}")  # CAMBIO B1-2: sin URL/clave
        return None


def adquirir_fred(
    series_id: str,
    inicio: str = HISTORIA_INICIO,
) -> pd.DataFrame:
    if not FRED_API_KEY:
        if DEBUG:
            print("[DEBUG] FRED: FRED_API_KEY no configurada")
        return pd.DataFrame()

    params: dict[str, Any] = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": inicio,
    }
    data = get_json(FRED_OBS, params)
    if data is None:
        if DEBUG:
            print(f"[DEBUG] FRED {series_id}: request fallida")
        return pd.DataFrame()

    obs: list[dict] = data.get("observations", [])
    if not obs:
        if DEBUG:
            print(f"[DEBUG] FRED {series_id}: sin observaciones")
        return pd.DataFrame()

    filas: list[dict] = []
    for o in obs:
        val = o.get("value", ".")
        try:
            fval = float(val) if val not in (".", "") else None
        except (ValueError, TypeError):
            fval = None
        filas.append({
            "fecha": o.get("date", ""),
            "value": fval,
        })

    df = pd.DataFrame(filas)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.set_index("fecha").sort_index()
    if DEBUG:
        n_ok = int(df["value"].notna().sum())
        print(
            f"[DEBUG] FRED {series_id}: {n_ok} obs "
            f"({df.index.min().date()} → "
            f"{df.index.max().date()})"
        )
    return df


def descargar_todas_fred() -> dict[str, pd.Series]:
    """
    Descarga todas las series FRED definidas en FRED_SERIES.
    Cada serie falla aislada; no tumba el resto.
    """
    resultado: dict[str, pd.Series] = {}
    for series_id, col_name in FRED_SERIES.items():
        try:
            df = adquirir_fred(series_id)
            if not df.empty and df["value"].notna().any():
                resultado[col_name] = df["value"].rename(
                    col_name
                )
                if DEBUG:
                    print(f"  → {col_name}: OK")
            else:
                if DEBUG:
                    print(f"  → {col_name}: sin datos")
        except Exception as exc:
            if DEBUG:
                print(
                    f"  → {col_name}: FALLÓ ({str(exc)[:50]})"
                )
        time.sleep(PAUSA)
    return resultado


def validar_fred_completo(series_fred: dict[str, pd.Series]) -> None:
    """CAMBIO E2E-04-01: no publicar series_completas parcial."""
    faltantes = [
        f"{serie_id} ({columna})"
        for serie_id, columna in FRED_SERIES.items()
        if columna not in series_fred
        or series_fred[columna].dropna().empty
    ]
    if faltantes:
        raise RuntimeError(
            "[DEBUG][ERROR] Job 4: descargas FRED incompletas: "
            + ", ".join(faltantes)
            + ". No se genera series_completas.csv en esta ejecucion."
        )


# ════════════════════════════════════════════════════════════
# 5. Funciones — Carga, unión y transformación
# ════════════════════════════════════════════════════════════


def cargar_series(ruta: str) -> pd.DataFrame:
    df = pd.read_csv(ruta, parse_dates=["fecha"])
    df = df.set_index("fecha").sort_index()
    if DEBUG:
        print(f"[DEBUG] {os.path.basename(ruta)}: {df.shape}")
    return df


def validar_entrada_largas_ext(ruta: str) -> pd.DataFrame:
    """CAMBIO E2E-04-01: validar salida del Job 3 antes de FRED."""
    if not os.path.isfile(ruta):
        raise FileNotFoundError(
            f"[DEBUG][ERROR] Job 4: falta {ruta}. "
            "Ejecutar el Job 3 para generar series_largas_ext.csv."
        )
    try:
        df = cargar_series(ruta)
    except (pd.errors.EmptyDataError, KeyError, ValueError) as exc:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 4: entrada ilegible o sin "
            f"columna 'fecha': {ruta}. Revisar el Job 3."
        ) from exc
    if df.empty:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 4: entrada vacia: {ruta}. "
            "Revisar la salida del Job 3."
        )
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 4: fechas invalidas en {ruta}. "
            "Revisar la salida del Job 3."
        )
    if df.index.has_duplicates:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 4: fechas duplicadas en {ruta}. "
            "No se puede alinear a una grilla diaria."
        )
    return df


def detectar_cola_estancada(
    serie: pd.Series,
    max_dias: int = MAX_DIAS_ESTANCADO,
) -> pd.Series:
    """
    Detecta, al final de una serie diaria ya alineada, una cola
    de valor idéntico repetido por más de `max_dias` días
    consecutivos — típicamente producida por ffill cuando la
    fuente no se actualizó, no por un valor real estancado.

    CAMBIO I7-1 (Job 8).

    Args:
        serie: Serie diaria (puede tener NaN).
        max_dias: Días consecutivos repetidos para considerar
            la cola "estancada".

    Returns:
        Máscara booleana (mismo índice que `serie`): True en los
        días de la cola estancada detectada al final.
    """
    mascara = pd.Series(False, index=serie.index)
    valores = serie.to_numpy()
    n = len(valores)
    i = n - 1
    while i >= 0 and pd.isna(valores[i]):
        i -= 1
    if i < 0:
        return mascara
    ultimo = valores[i]
    j = i
    while j >= 0 and valores[j] == ultimo:
        j -= 1
    largo = i - j
    if largo > max_dias:
        mascara.iloc[j + 1:i + 1] = True
    return mascara


def construir_dataset_unificado(
    df_base: pd.DataFrame,
    series_fred: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Une series_largas_ext con las series FRED descargadas.
    Alinea a grilla diaria con ffill (sin interpolación).

    CAMBIO I7-1 (Job 8): el ffill de la grilla diaria se aplica
    a todas las columnas, incluidas las RWA. Si una columna RWA
    no se actualizó en la fuente, ffill repite su último valor
    real indefinidamente — una cola "estancada" que no es dato
    real. Se detecta y se vuelve a NaN (no se rellena), y se
    registra en ARCHIVO_COLAS_ESTANCADAS.
    """
    df = df_base.copy()

    for col_name, serie in series_fred.items():
        if col_name in df.columns:
            if DEBUG:
                print(
                    f"[DEBUG] {col_name} ya existe en base, "
                    "se sobreescribe con FRED"
                )
        df = df.join(
            serie.to_frame(col_name), how="outer"
        )

    df = df.asfreq("D")
    df = df.ffill()

    rwa_cols = [c for c in COLS_RWA_ALL if c in df.columns]
    if rwa_cols:
        mask_all_nan = df[rwa_cols].isna().all(axis=1)
        first_valid = (~mask_all_nan).idxmax()
        df = df.loc[first_valid:].copy()

    filas_colas: list[dict[str, Any]] = []
    for col in rwa_cols:
        mascara = detectar_cola_estancada(df[col])
        n_afectados = int(mascara.sum())
        if n_afectados > 0:
            df.loc[mascara, col] = np.nan
            if DEBUG:
                print(
                    f"[DEBUG] {col}: cola estancada de "
                    f"{n_afectados} días → vuelve a NaN "
                    "(no se rellena)"
                )
        filas_colas.append({
            "columna": col,
            "n_dias_estancados": n_afectados,
            "fecha_extraccion": HOY,
        })
    if filas_colas and SAVE_DATA:
        os.makedirs(DIR_REPORTES, exist_ok=True)
        pd.DataFrame(filas_colas).to_csv(
            ARCHIVO_COLAS_ESTANCADAS, index=False
        )
        if DEBUG:
            print(f"[DEBUG] guardado: {ARCHIVO_COLAS_ESTANCADAS}")

    return df


def reportar_cobertura(df: pd.DataFrame) -> pd.DataFrame:
    filas: list[dict] = []
    for col in df.columns:
        serie = df[col]
        n_total = len(serie)
        n_nulo = int(serie.isna().sum())
        n_val = n_total - n_nulo
        fechas_val = serie.dropna()
        if len(fechas_val) > 0:
            min_f = fechas_val.index.min().date().isoformat()
            max_f = fechas_val.index.max().date().isoformat()
        else:
            min_f, max_f = "s/d", "s/d"
        filas.append({
            "columna": col,
            "n_total": n_total,
            "n_con_valor": n_val,
            "n_nulo": n_nulo,
            "pct_cobertura": round(
                n_val / n_total * 100, 1
            ) if n_total > 0 else 0.0,
            "min_fecha": min_f,
            "max_fecha": max_f,
        })
    return pd.DataFrame(filas)


def _log_diff(serie: pd.Series) -> pd.Series:
    return np.log(serie.where(serie > 0)).diff()


def transformar_columna(
    serie: pd.Series, tipo: str,
) -> pd.Series:
    if tipo == "log_diff":
        return _log_diff(serie)
    elif tipo == "diff":
        return serie.diff()
    elif tipo == "nivel":
        return serie.copy()
    else:
        return serie.diff()


def frecuencia_admite_pista(col: str, pista: str) -> bool:
    """
    True si la frecuencia nativa de `col` es igual o más fina
    que la de `pista` (A→diaria, B→semanal, M→mensual).

    CAMBIO C4-2 (Job 8). Las columnas que no están en
    FRECUENCIA_NATIVA (RWA/onchain) son diarias por construcción
    y siempre califican.

    Args:
        col: Nombre de la columna.
        pista: "A", "B" o "M".

    Returns:
        True si `col` puede participar de `pista`.
    """
    nativa = FRECUENCIA_NATIVA.get(col)
    if nativa is None:
        return True
    freq_pista = _PISTA_A_FRECUENCIA.get(pista, "D")
    return _RANGO_FRECUENCIA[nativa] <= _RANGO_FRECUENCIA[freq_pista]


def transformar_pista_a(
    df: pd.DataFrame,
    cols: list[str],
) -> dict[str, pd.Series]:
    """
    Pista A (diaria): transforma cada columna según TRANSFORM.

    CAMBIO C4-2 (Job 8): solo participan columnas con frecuencia
    nativa diaria (FRECUENCIA_NATIVA). NFCI, STLFSI4 (semanales)
    e INDPRO, UMCSENT (mensuales) quedan excluidas: antes se
    ffilleaban a diario y se diferenciaban a diario.
    """
    resultado: dict[str, pd.Series] = {}
    for col in cols:
        if col not in df.columns:
            continue
        if not frecuencia_admite_pista(col, "A"):
            if DEBUG:
                print(
                    f"[DEBUG][ALERTA] Pista A: {col} excluida "
                    "(frecuencia nativa "
                    f"{FRECUENCIA_NATIVA.get(col)}, no diaria)"
                )
            continue
        tipo = TRANSFORM.get(col, "diff")
        s = transformar_columna(df[col], tipo).dropna()
        if len(s) > 10:
            resultado[col] = s
    return resultado


def transformar_pista_b(
    df: pd.DataFrame,
    cols: list[str],
) -> dict[str, pd.Series]:
    """
    Pista B (semanal): resample W.last() → transformar.

    CAMBIO C4-2 (Job 8): solo participan columnas con frecuencia
    nativa diaria o semanal. INDPRO y UMCSENT (mensuales) quedan
    excluidas de esta pista (ver Pista M / INCLUIR_PISTA_MENSUAL).
    NFCI y STLFSI4 entran con su valor semanal "last" (TRANSFORM
    ya las tiene como "nivel", sin diferenciar).
    """
    resultado: dict[str, pd.Series] = {}
    for col in cols:
        if col not in df.columns:
            continue
        if not frecuencia_admite_pista(col, "B"):
            if DEBUG:
                print(
                    f"[DEBUG][ALERTA] Pista B: {col} excluida "
                    "(frecuencia nativa "
                    f"{FRECUENCIA_NATIVA.get(col)}, no semanal)"
                )
            continue
        serie_sem = df[col].resample("W").last()
        tipo = TRANSFORM.get(col, "diff")
        s = transformar_columna(serie_sem, tipo).dropna()
        if len(s) > 10:
            resultado[col] = s
    return resultado


def transformar_pista_m(
    df: pd.DataFrame,
    cols: list[str],
    n_lags: int = LAGS_PISTA_M,
) -> dict[str, pd.Series]:
    """
    Pista M (mensual, OPCIONAL): resample ME.last() →
    transformar → rezagos 1 a `n_lags`.

    CAMBIO C4-3 (Job 8): única pista donde participan columnas
    de frecuencia nativa mensual (INDPRO, UMCSENT). Apagada por
    defecto (INCLUIR_PISTA_MENSUAL=False) — ver nota de alcance
    sobre su integración al battery D1/D2 de PASO 4.

    Args:
        df: Dataset diario ya unificado.
        cols: Columnas candidatas.
        n_lags: Cantidad de rezagos a generar (1..n_lags).

    Returns:
        Dict {f"{col}_lag{k}": serie} con las series rezagadas.
    """
    resultado: dict[str, pd.Series] = {}
    for col in cols:
        if col not in df.columns:
            continue
        if not frecuencia_admite_pista(col, "M"):
            continue
        serie_men = df[col].resample("ME").last()
        tipo = TRANSFORM.get(col, "diff")
        s = transformar_columna(serie_men, tipo).dropna()
        if len(s) <= 10:
            continue
        for k in range(1, n_lags + 1):
            s_lag = s.shift(k).dropna()
            if len(s_lag) > 10:
                resultado[f"{col}_lag{k}"] = s_lag
    return resultado


# ════════════════════════════════════════════════════════════
# 6. Funciones — Granger bidireccional
# ════════════════════════════════════════════════════════════


def _granger_raw(data: pd.DataFrame, max_lag: int) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                return grangercausalitytests(
                    data, maxlag=max_lag
                )
            except TypeError:
                return grangercausalitytests(
                    data, maxlag=max_lag, verbose=False
                )


def elegir_rezago_bic(
    serie_y: pd.Series,
    serie_x: pd.Series,
    max_lag: int,
) -> int:
    """
    Elige UN rezago (1..max_lag) por BIC (VAR.select_order).

    CAMBIO I5-1 (Job 8): reemplaza el criterio anterior de
    tratar cada rezago 1..max_lag como un test independiente.

    Returns:
        Rezago elegido (1..max_lag); 1 si falla o faltan datos.
    """
    data = pd.concat(
        [serie_y.rename("y"), serie_x.rename("x")], axis=1
    ).dropna()
    if len(data) < max_lag + 10:
        return 1
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = VAR(data).select_order(maxlags=max_lag)
        lag = int(sel.bic)
    except Exception as exc:
        if DEBUG:
            print(f"[DEBUG] BIC falló, uso rezago 1: {exc}")
        return 1
    return max(1, min(lag, max_lag))


def verificar_estacionariedad(
    serie: pd.Series,
    etiqueta: str,
    alpha: float = ALFA_BH,
) -> bool:
    """
    ADF sobre una serie ya transformada; registra si NO es
    estacionaria (CAMBIO I5-6/I5-f).
    """
    sc = serie.dropna()
    if len(sc) < 15 or float(sc.std()) == 0:
        if DEBUG:
            print(
                f"[DEBUG][ALERTA] {etiqueta}: no se pudo "
                f"verificar estacionariedad (n={len(sc)})"
            )
        return False
    try:
        _, pval, *_ = adfuller(sc, autolag="AIC")
    except Exception as exc:
        if DEBUG:
            print(f"[DEBUG][ALERTA] {etiqueta}: ADF falló: {exc}")
        return False
    estacionaria = float(pval) < alpha
    if not estacionaria and DEBUG:
        print(
            f"[DEBUG][ALERTA] {etiqueta}: NO estacionaria tras "
            f"su transformación elegida (ADF p={pval:.4f})"
        )
    return estacionaria


def granger_par_bidir(
    serie_feat: pd.Series,
    serie_target: pd.Series,
    max_lag: int,
    nombre_feat: str,
    nombre_target: str,
    pista: str,
) -> tuple[list[dict], list[dict]]:
    """
    CAMBIO I5-1/I5-6: devuelve (filas_por_rezago, filas_familia).
    filas_por_rezago es la tabla de apoyo (1 fila/rezago, NO
    entra a BH). filas_familia tiene 1 fila por dirección, en el
    rezago elegido por BIC — esa es la familia de BH. Antes de
    elegir el rezago se verifica estacionariedad de x e y (I5-f).

    Direcciones (CAMBIO B3-5, solo documentación):
      · D1_feat→target: H0 "serie_feat NO Granger-causa a
        serie_target" (y=target, x=feat). Rechazarla = el
        feature anticipa al target.
      · D2_target→feat: la inversa (y=feat, x=target).
    Qué es feat y qué es target depende de la familia de pares
    armada en main(): RWA (feat) vs ΔCISS/STLFSI (target), y
    macro (feat) vs cada RWA (target). Por eso D1 NO significa
    siempre "RWA→estrés". Las etiquetas de Job 9
    (D1_rwa→estres / D2_estres→rwa) son otra convención.
    """
    filas: list[dict] = []
    familia: list[dict] = []

    pares = [
        ("D1_feat→target", serie_target, serie_feat),
        ("D2_target→feat", serie_feat, serie_target),
    ]

    for direccion, y, x in pares:
        data = pd.concat(
            [y.rename("y"), x.rename("x")], axis=1
        ).dropna()
        n_min = max_lag + 10
        fila_base = {
            "pista": pista,
            "serie_feat": nombre_feat,
            "serie_target": nombre_target,
            "direccion": direccion,
            "n": len(data),
            # CAMBIO C4-4 (Job 8): frecuencia nativa del feature,
            # para auditar qué pista le corresponde realmente.
            # "D" por defecto para columnas RWA/onchain (no están
            # en FRECUENCIA_NATIVA: son diarias por construcción).
            "frec_nativa_feat": FRECUENCIA_NATIVA.get(
                nombre_feat, "D"
            ),
        }
        if len(data) < n_min:
            filas.extend([
                {
                    **fila_base, "lag": k,
                    "pval_f": None, "pval_chi2": None,
                    "nota": f"n={len(data)}<{n_min}",
                }
                for k in range(1, max_lag + 1)
            ])
            familia.append({
                **fila_base, "lag": None, "pval_f": None,
                "nota": f"n={len(data)}<{n_min}",
            })
            continue

        verificar_estacionariedad(y, f"{pista}/{direccion}/y")
        verificar_estacionariedad(x, f"{pista}/{direccion}/x")

        try:
            resultado = _granger_raw(data[["y", "x"]], max_lag)
        except Exception as exc:
            filas.extend([
                {
                    **fila_base, "lag": k,
                    "pval_f": None, "pval_chi2": None,
                    "nota": f"error: {str(exc)[:60]}",
                }
                for k in range(1, max_lag + 1)
            ])
            familia.append({
                **fila_base, "lag": None, "pval_f": None,
                "nota": f"error: {str(exc)[:60]}",
            })
            continue

        for k in range(1, max_lag + 1):
            try:
                pval_f = float(resultado[k][0]["ssr_ftest"][1])
                pval_chi2 = float(
                    resultado[k][0]["ssr_chi2test"][1]
                )
            except Exception:
                filas.append({
                    **fila_base, "lag": k,
                    "pval_f": None, "pval_chi2": None,
                    "nota": "sin estadístico ssr",
                })
                continue
            filas.append({
                **fila_base,
                "lag": k,
                "pval_f": round(pval_f, 6),
                "pval_chi2": round(pval_chi2, 6),
                "nota": "",
            })

        # CAMBIO I5-1: UN rezago por BIC para la familia de BH.
        lag_elegido = elegir_rezago_bic(y, x, max_lag)
        fila_elegida = next(
            (
                f for f in filas
                if f["direccion"] == direccion
                and f["lag"] == lag_elegido
            ),
            None,
        )
        if fila_elegida is not None:
            familia.append(fila_elegida)
        else:
            familia.append({
                **fila_base, "lag": lag_elegido, "pval_f": None,
                "nota": "rezago elegido sin estadístico",
            })
    return filas, familia


def aplicar_bh_global(
    df: pd.DataFrame,
    col_p: str = "pval_f",
    alfa: float = ALFA_BH,
) -> pd.DataFrame:
    """
    CAMBIO I5-1: `df` ahora es la familia (1 fila por par ×
    dirección × pista, en su rezago elegido por BIC).
    """
    df = df.copy().reset_index(drop=True)
    df["bh_q"] = np.nan
    df["bh_rechaza"] = False
    df["ambito_bh"] = (
        "global: 2 dir × 2 pistas × todos los pares "
        "(1 rezago por BIC)"
    )
    pvals = pd.to_numeric(df[col_p], errors="coerce")
    validos = pvals.notna()
    if validos.sum() < 2:
        return df
    rechaza, adj, _, _ = multipletests(
        pvals[validos].values, alpha=alfa, method="fdr_bh"
    )
    df.loc[validos, "bh_q"] = np.round(adj, 6)
    df.loc[validos, "bh_rechaza"] = rechaza
    return df


def lag_minimo_pval(df_granger: pd.DataFrame) -> pd.DataFrame:
    validos = df_granger[df_granger["pval_f"].notna()]
    if validos.empty:
        return pd.DataFrame()
    idx = validos.groupby(
        ["pista", "serie_feat", "serie_target", "direccion"]
    )["pval_f"].idxmin()
    return df_granger.loc[idx].copy()


def top_señales(
    df_granger: pd.DataFrame, n: int = TOP_N_SEÑALES,
) -> pd.DataFrame:
    """
    Los N pares con menor bh_q, aunque no sobrevivan el FDR.
    """
    df_opt = lag_minimo_pval(df_granger)
    if df_opt.empty:
        return pd.DataFrame()
    validos = df_opt[df_opt["bh_q"].notna()].copy()
    if validos.empty:
        return pd.DataFrame()
    return (
        validos
        .sort_values("bh_q")
        .head(n)
        .reset_index(drop=True)
    )


# ════════════════════════════════════════════════════════════
# 7. Funciones — Modelo predictivo
# ════════════════════════════════════════════════════════════


def construir_features_lag(
    df_x: pd.DataFrame, K: int,
) -> pd.DataFrame:
    partes: list[pd.DataFrame] = []
    for k in range(1, K + 1):
        shifted = df_x.shift(k)
        shifted.columns = [
            f"{c}_lag{k}" for c in shifted.columns
        ]
        partes.append(shifted)
    return pd.concat(partes, axis=1)


def split_temporal(
    X_lag: pd.DataFrame,
    ser_y: pd.Series,
    ratio_test: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    combinado = pd.concat(
        [ser_y.rename("__y"), X_lag], axis=1
    ).dropna()
    n = len(combinado)
    n_test = max(2, int(n * ratio_test))
    n_train = n - n_test
    if n_train < 10:
        raise ValueError(
            f"Train demasiado pequeño: {n_train} obs"
        )
    y_all = combinado.pop("__y")
    X_all = combinado
    if DEBUG:
        print(
            f"[DEBUG] split: n={n}, train={n_train}, "
            f"test={n_test}, features={X_all.shape[1]}"
        )
    return (
        X_all.iloc[:n_train],
        X_all.iloc[n_train:],
        y_all.iloc[:n_train],
        y_all.iloc[n_train:],
    )


def metricas_regresion(
    y_true: pd.Series,
    y_pred: pd.Series,
    nombre: str,
    pista: str,
    target: str,
    direccion: str,
) -> dict:
    yt = y_true.to_numpy(dtype=float)
    yp = y_pred.to_numpy(dtype=float)
    mask = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[mask], yp[mask]
    base = {
        "pista": pista,
        "target": target,
        "direccion": direccion,
        "modelo": nombre,
    }
    if len(yt) == 0:
        return {
            **base, "n_test": 0, "n_train": 0,
            "n_features": 0,
            "rmse": None, "mae": None, "r2_oos": None,
        }
    err = yt - yp
    mse = float(np.mean(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = (
        1.0 - float(np.sum(err ** 2)) / ss_tot
        if ss_tot > 1e-12 else float("nan")
    )
    return {
        **base,
        "n_test": int(len(yt)),
        "rmse": round(float(np.sqrt(mse)), 8),
        "mae": round(float(np.mean(np.abs(err))), 8),
        "r2_oos": round(r2, 4),
    }


def baseline_persistencia(
    y_train: pd.Series, y_test: pd.Series,
) -> pd.Series:
    y_all = pd.concat([y_train, y_test])
    return y_all.shift(1).loc[y_test.index].rename("persistencia")


def ar_un_paso(
    y_train: pd.Series, y_test: pd.Series, p: int,
) -> pd.Series:
    """
    Pronóstico AR(p) a un paso con el valor REAL del período
    anterior (no el pronosticado) — CAMBIO C2-1.
    """
    res = AutoReg(y_train.to_numpy(), lags=p, trend="c").fit()
    c, phi = res.params[0], res.params[1:]
    y_all = np.r_[y_train.to_numpy(), y_test.to_numpy()]
    pred = [
        c + sum(phi[k] * y_all[t - 1 - k] for k in range(p))
        for t in range(len(y_train), len(y_all))
    ]
    return pd.Series(pred, index=y_test.index, name=f"ar{p}")


def modelo_ar(
    y_train: pd.Series,
    y_test: pd.Series,
    p: int,
    etiqueta: str,
) -> tuple[pd.Series, Any, bool]:
    """
    CAMBIO C1-1: sin old_names=False. CAMBIO C2-1: pronóstico a
    un paso (ar_un_paso), no dinámico. CAMBIO C1-3: sin fallback
    silencioso — si falla, [DEBUG][ALERTA], media como
    "ar_media" (nunca "ar{p}"), baseline_ok=False.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = ar_un_paso(y_train, y_test, p)
            res = AutoReg(
                y_train.to_numpy(), lags=p, trend="c"
            ).fit()
        return pred, res, True
    except Exception as exc:
        print(f"[DEBUG][ALERTA] AR({p}) {etiqueta} falló: {exc}")
        media = float(y_train.mean())
        return (
            pd.Series(
                [media] * len(y_test),
                index=y_test.index, name="ar_media",
            ),
            None, False,
        )


def _cv_splits(n_train: int) -> int:
    return max(2, min(5, n_train // 15))


def _chequear_alpha_borde(
    alpha: float, grilla: np.ndarray, etiqueta: str,
) -> bool:
    """True si alpha quedó en el borde de la grilla (CAMBIO C3-3)."""
    grilla = np.asarray(grilla)
    en_borde = bool(
        np.isclose(alpha, grilla.min())
        or np.isclose(alpha, grilla.max())
    )
    if en_borde:
        print(
            f"[DEBUG][ALERTA] {etiqueta}: alpha en el borde de "
            f"la grilla (alpha={alpha:.4g})"
        )
    return en_borde


def modelo_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    etiqueta: str,
) -> tuple[pd.Series, Any, StandardScaler, bool]:
    """CAMBIO C3-1/C3-3/C3-4: RIDGE_ALPHAS, aviso de borde, std(y)."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xts = scaler.transform(X_test)
    cv = TimeSeriesSplit(n_splits=_cv_splits(len(X_train)))
    if DEBUG:
        print(
            f"[DEBUG] {etiqueta}: std(y_train)="
            f"{float(y_train.std()):.6g}"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = RidgeCV(alphas=RIDGE_ALPHAS, cv=cv)
            model.fit(Xtr, y_train)
        pred = pd.Series(
            model.predict(Xts), index=X_test.index, name="ridge"
        )
        alpha_borde = _chequear_alpha_borde(
            model.alpha_, RIDGE_ALPHAS, f"Ridge {etiqueta}",
        )
        if DEBUG:
            print(
                f"[DEBUG] Ridge {etiqueta}: "
                f"alpha={model.alpha_:.4g}"
            )
        return pred, model, scaler, alpha_borde
    except Exception as exc:
        if DEBUG:
            print(f"[DEBUG] Ridge {etiqueta} falló: {exc}")
        media = float(y_train.mean())
        return (
            pd.Series(
                [media] * len(X_test),
                index=X_test.index, name="ridge",
            ),
            None, scaler, False,
        )


def modelo_lasso(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    etiqueta: str,
) -> tuple[pd.Series, Any, StandardScaler, bool]:
    """
    CAMBIO C3-2/C3-3/C3-4: alphas=None (n_alphas=100, eps=1e-3,
    max_iter=10000), aviso de borde, std(y).
    """
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xts = scaler.transform(X_test)
    cv = TimeSeriesSplit(n_splits=_cv_splits(len(X_train)))
    if DEBUG:
        print(
            f"[DEBUG] {etiqueta}: std(y_train)="
            f"{float(y_train.std()):.6g}"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LassoCV(
                alphas=None, n_alphas=100, eps=1e-3,
                max_iter=10000, cv=cv,
            )
            model.fit(Xtr, y_train)
        alpha_borde = _chequear_alpha_borde(
            model.alpha_, model.alphas_, f"Lasso {etiqueta}",
        )
        pred = pd.Series(
            model.predict(Xts), index=X_test.index, name="lasso"
        )
        if DEBUG:
            n_nz = int(np.sum(model.coef_ != 0))
            print(
                f"[DEBUG] Lasso {etiqueta}: "
                f"alpha={model.alpha_:.4g}, coef_nz={n_nz}"
                f"/{X_train.shape[1]}"
            )
        return pred, model, scaler, alpha_borde
    except Exception as exc:
        if DEBUG:
            print(f"[DEBUG] Lasso {etiqueta} falló: {exc}")
        media = float(y_train.mean())
        return (
            pd.Series(
                [media] * len(X_test),
                index=X_test.index, name="lasso",
            ),
            None, scaler, False,
        )


def coeficientes_lasso(
    modelo: Any,
    columnas: list[str],
    pista: str,
    target: str,
    direccion: str,
) -> pd.DataFrame:
    if modelo is None or not hasattr(modelo, "coef_"):
        return pd.DataFrame()
    coefs = np.asarray(modelo.coef_, dtype=float)
    no_nulos = np.flatnonzero(coefs != 0)
    if no_nulos.size == 0:
        return pd.DataFrame()
    filas = [
        {
            "pista": pista,
            "target": target,
            "direccion": direccion,
            "feature": columnas[i],
            "coef": round(float(coefs[i]), 8),
        }
        for i in no_nulos
    ]
    df_coef = pd.DataFrame(filas)
    return df_coef.reindex(
        df_coef["coef"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def correr_modelo_target(
    feat_dict: dict[str, pd.Series],
    y_series: pd.Series,
    K: int,
    ar_p: int,
    pista: str,
    target_name: str,
    direccion: str,
) -> dict:
    """
    Corre baselines + Ridge + Lasso para un target dado.
    feat_dict contiene las features transformadas.
    """
    cols_ok = [
        c for c in feat_dict
        if len(feat_dict[c]) > 10
    ]
    etiqueta = f"{pista}/{direccion}/{target_name}"

    if not cols_ok:
        if DEBUG:
            print(f"[DEBUG] {etiqueta}: sin features")
        return {
            "metricas": [], "coeficientes": pd.DataFrame(),
            "modelos": {},
        }

    df_x = pd.concat(
        [feat_dict[c].rename(c) for c in cols_ok], axis=1
    )
    X_lag = construir_features_lag(df_x, K)

    try:
        Xtr, Xts, ytr, yts = split_temporal(X_lag, y_series)
    except ValueError as exc:
        if DEBUG:
            print(f"[DEBUG] {etiqueta} split falló: {exc}")
        return {
            "metricas": [], "coeficientes": pd.DataFrame(),
            "modelos": {},
        }

    preds: dict[str, pd.Series] = {}
    preds["persistencia"] = baseline_persistencia(ytr, yts)
    pred_ar, _, ar_ok = modelo_ar(ytr, yts, ar_p, etiqueta)
    preds[pred_ar.name] = pred_ar  # CAMBIO C1-3
    pred_ridge, mdl_ridge, _, ridge_borde = modelo_ridge(
        Xtr, ytr, Xts, etiqueta
    )
    preds["ridge"] = pred_ridge
    pred_lasso, mdl_lasso, _, lasso_borde = modelo_lasso(
        Xtr, ytr, Xts, etiqueta
    )
    preds["lasso"] = pred_lasso

    metricas = [
        metricas_regresion(
            yts, serie, nombre, pista,
            target_name, direccion,
        )
        for nombre, serie in preds.items()
    ]
    _borde_por_modelo = {"ridge": ridge_borde, "lasso": lasso_borde}
    for m in metricas:
        m["n_train"] = int(len(ytr))
        m["n_features"] = int(Xtr.shape[1])
        m["baseline_ok"] = (
            ar_ok if m["modelo"] == pred_ar.name else True
        )
        m["alpha_en_borde"] = _borde_por_modelo.get(
            m["modelo"], False
        )

    df_coef = coeficientes_lasso(
        mdl_lasso, list(Xtr.columns),
        pista, target_name, direccion,
    )

    tag = (
        f"{pista.lower()}_{direccion.lower()}_"
        f"{target_name.replace('macro_', '').replace('rwa_', '')}"
    )[:60]
    modelos: dict[str, Any] = {}
    if mdl_ridge is not None:
        modelos[f"modelo_ridge_{tag}"] = mdl_ridge
    if mdl_lasso is not None:
        modelos[f"modelo_lasso_{tag}"] = mdl_lasso

    return {
        "metricas": metricas,
        "coeficientes": df_coef,
        "modelos": modelos,
    }


# ════════════════════════════════════════════════════════════
# 8. Funciones — Veredicto
# ════════════════════════════════════════════════════════════


def evaluar_corroboracion_direccion(
    direccion: str,
    df_granger: pd.DataFrame,
    df_met: pd.DataFrame,
    df_coefs: pd.DataFrame,
) -> dict:
    sub_g = df_granger[df_granger["direccion"] == direccion]
    sub_m = df_met[df_met["direccion"] == direccion]
    sub_c = (
        df_coefs[df_coefs["direccion"] == direccion]
        if not df_coefs.empty else pd.DataFrame()
    )

    c1 = not sub_c.empty

    n_sig = 0
    n_tests = 0
    if not sub_g.empty:
        pvals = pd.to_numeric(sub_g["pval_f"], errors="coerce")
        n_tests = int(pvals.notna().sum())
        n_sig = int(
            sub_g["bh_rechaza"].fillna(False).astype(bool).sum()
        )
    c2 = n_sig > 0

    mejora_max = 0.0
    mejor_modelo = "n/a"
    if not sub_m.empty:
        ar_rows = sub_m[
            sub_m["modelo"].str.startswith("ar")
        ].dropna(subset=["rmse"])
        feat_rows = sub_m[
            sub_m["modelo"].isin(["ridge", "lasso"])
        ].dropna(subset=["rmse"])
        if not ar_rows.empty and not feat_rows.empty:
            for (pista_val, tgt), grp_ar in ar_rows.groupby(
                ["pista", "target"]
            ):
                grp_ft = feat_rows[
                    (feat_rows["pista"] == pista_val)
                    & (feat_rows["target"] == tgt)
                ]
                if grp_ft.empty:
                    continue
                rmse_ar = float(grp_ar["rmse"].min())
                if rmse_ar <= 0:
                    continue
                for _, fila in grp_ft.iterrows():
                    rmse_f = float(fila["rmse"])
                    mejora = 100.0 * (rmse_ar - rmse_f) / rmse_ar
                    if mejora > mejora_max:
                        mejora_max = mejora
                        mejor_modelo = (
                            f"{fila['modelo']} "
                            f"({pista_val}/{tgt})"
                        )
    c3 = mejora_max >= UMBRAL_MEJORA_MATERIAL

    c4 = False
    if not sub_m.empty:
        pistas_con_mejora: set[str] = set()
        feat_rows = sub_m[
            sub_m["modelo"].isin(["ridge", "lasso"])
        ].dropna(subset=["rmse"])
        ar_rows = sub_m[
            sub_m["modelo"].str.startswith("ar")
        ].dropna(subset=["rmse"])
        for pista_val in sub_m["pista"].unique():
            ar_p = ar_rows[ar_rows["pista"] == pista_val]
            ft_p = feat_rows[feat_rows["pista"] == pista_val]
            if ar_p.empty or ft_p.empty:
                continue
            rmse_ar = float(ar_p["rmse"].min())
            rmse_best = float(ft_p["rmse"].min())
            if rmse_best < rmse_ar:
                pistas_con_mejora.add(pista_val)
        c4 = len(pistas_con_mejora) >= 2

    return {
        "direccion": direccion,
        "c1_lasso_retiene": c1,
        "c2_granger_fdr": c2,
        "c3_mejora_material": c3,
        "c4_ambas_pistas": c4,
        "n_sig_bh": n_sig,
        "n_tests": n_tests,
        "mejora_max_pct": round(mejora_max, 2),
        "mejor_modelo": mejor_modelo,
        "corroborada": bool(c1 and c2 and c3 and c4),
    }


def features_mas_retenidas(
    df_coefs: pd.DataFrame, n: int = 10,
) -> pd.DataFrame:
    """
    Features más frecuentemente retenidas por Lasso across
    todos los modelos.
    """
    if df_coefs.empty:
        return pd.DataFrame()
    conteo = (
        df_coefs.groupby("feature")
        .agg(
            veces_retenida=("coef", "size"),
            coef_medio_abs=("coef", lambda x: np.mean(np.abs(x))),
        )
        .sort_values("veces_retenida", ascending=False)
        .head(n)
        .reset_index()
    )
    conteo["coef_medio_abs"] = conteo["coef_medio_abs"].round(8)
    return conteo


# ════════════════════════════════════════════════════════════
# 9. Proceso principal
# ════════════════════════════════════════════════════════════


def main() -> None:
    print(
        "\n[DEBUG] ═══════════════════════════════════════════"
        "═══════════════"
    )
    print(
        "[DEBUG] 18 — Modelo Completo: Macro + RWA ↔ Estrés"
    )
    print(
        "[DEBUG] ═══════════════════════════════════════════"
        "═══════════════"
    )

    # CAMBIO E2E-04-01: validar el archivo del Job 3 ANTES de
    # las descargas FRED (evita consultas inutiles o un CSV viejo).
    df_base = validar_entrada_largas_ext(ARCHIVO_LARGAS_EXT)

    # CAMBIO B1-3 (Bloque 1): en modo snapshot los pasos 1-2 no
    # descargan FRED: series_completas.csv se restaura del snapshot
    # (hash verificado) y se lee de disco. Pasos 3+ sin cambios.
    if MODO_DATOS == "snapshot":
        print("\n[DEBUG] ══ PASOS 1-2: snapshot (sin descarga FRED) ══")
        restaurar_desde_snapshot("Job 4", [ARCHIVO_COMPLETAS])
        df = cargar_series(ARCHIVO_COMPLETAS)
        series_fred = {
            _col: df[_col] for _col in FRED_SERIES.values()
            if _col in df.columns
        }
        validar_fred_completo(series_fred)
    else:
        # ════════════════════════════════════════════════════════
        # PASO 1 — Descargar variables macro de FRED
        # ════════════════════════════════════════════════════════
        print("\n[DEBUG] ══ PASO 1: Descarga FRED ══")
        series_fred = descargar_todas_fred()
        validar_fred_completo(series_fred)  # CAMBIO E2E-04-01
        print(
            f"\n[DEBUG] Series FRED descargadas: "
            f"{len(series_fred)}/{len(FRED_SERIES)}"
        )

        # ════════════════════════════════════════════════════════
        # PASO 2 — Construir dataset unificado
        # ════════════════════════════════════════════════════════
        print("\n[DEBUG] ══ PASO 2: Dataset unificado ══")

        df = construir_dataset_unificado(df_base, series_fred)
        if df.empty:
            raise RuntimeError(
                "[DEBUG][ERROR] Job 4: dataset unificado vacio; "
                "no se escribe series_completas.csv."
            )

    print(
        f"[DEBUG] Dataset unificado: {df.shape} | "
        f"{df.index.min().date()} → {df.index.max().date()}"
    )

    cobertura = reportar_cobertura(df)
    if DEBUG:
        _cob_cols = [
            "columna", "n_con_valor", "n_nulo",
            "pct_cobertura", "min_fecha", "max_fecha",
        ]
        print("\n[DEBUG] Cobertura:")
        print(cobertura[_cob_cols].to_string(index=False))

    # CAMBIO B1-3: en modo snapshot el CSV ya es el congelado.
    if SAVE_DATA and MODO_DATOS == "descarga":
        os.makedirs(DIR_PROCESSED, exist_ok=True)
        df.index.name = "fecha"
        df.to_csv(ARCHIVO_COMPLETAS)
        if DEBUG:
            print(f"\n[DEBUG] guardado: {ARCHIVO_COMPLETAS}")
        congelar_snapshot([ARCHIVO_COMPLETAS], "Job 4")  # CAMBIO B1-3

    # ════════════════════════════════════════════════════════
    # PASO 3 — Transformaciones
    # ════════════════════════════════════════════════════════
    print("\n[DEBUG] ══ PASO 3: Transformaciones ══")

    all_cols = (
        COLS_RWA_ALL + COLS_MACRO_FEAT
    )
    all_cols = list(dict.fromkeys(all_cols))

    feat_a = transformar_pista_a(df, all_cols)
    feat_b = transformar_pista_b(df, all_cols)

    # CAMBIO C4-3 (Job 8): Pista M (mensual), opcional. Si está
    # apagada, INDPRO/UMCSENT quedan fuera de las 3 pistas — ya
    # excluidas de A y B por frecuencia_admite_pista(), y no se
    # arma la M.
    if INCLUIR_PISTA_MENSUAL:
        feat_m = transformar_pista_m(df, all_cols)
        print(
            f"[DEBUG] Pista M: {len(feat_m)} features "
            f"(rezagos 1-{LAGS_PISTA_M})"
        )
    else:
        feat_m = {}
        print(
            "[DEBUG] Pista M desactivada "
            "(INCLUIR_PISTA_MENSUAL=False): macro_indpro y "
            "macro_umcsent no entran a ninguna prueba en esta "
            "corrida."
        )

    # Targets diarios
    y_ciss_a = df[COL_CISS].diff().dropna() if (
        COL_CISS in df.columns
    ) else pd.Series(dtype=float)

    # Target semanal
    y_stlfsi_b = pd.Series(dtype=float)
    if COL_STLFSI in df.columns:
        y_stlfsi_b = (
            df[COL_STLFSI].resample("W").last().dropna()
        )

    print(f"[DEBUG] Pista A: {len(feat_a)} features transformadas")
    print(f"[DEBUG] Pista B: {len(feat_b)} features transformadas")
    if not y_ciss_a.empty:
        print(f"[DEBUG] ΔCISS: {len(y_ciss_a)} obs diarias")
    if not y_stlfsi_b.empty:
        print(
            f"[DEBUG] STLFSI: {len(y_stlfsi_b)} obs semanales"
        )

    # ════════════════════════════════════════════════════════
    # PASO 4 — Modelos predictivos bidireccionales
    # ════════════════════════════════════════════════════════
    print("\n[DEBUG] ══ PASO 4: Modelos predictivos ══")

    resultados: list[dict] = []
    modelos_guardar: dict[str, Any] = {}

    # ── D1: macro/RWA features → estrés ─────────────────────
    print("\n[DEBUG] ── D1: features → estrés ──")

    # Pista A: todas las features → ΔCISS
    if not y_ciss_a.empty:
        try:
            res = correr_modelo_target(
                feat_a, y_ciss_a, K_FEAT_A, AR_P_A,
                "A", "ΔCISS", "D1_feat→target",
            )
            resultados.append(res)
            modelos_guardar.update(res["modelos"])
        except Exception as exc:
            if DEBUG:
                print(f"[DEBUG] D1 A/ΔCISS falló: {exc}")

    # Pista B: todas las features → STLFSI
    if not y_stlfsi_b.empty:
        try:
            res = correr_modelo_target(
                feat_b, y_stlfsi_b, K_FEAT_B, AR_P_B,
                "B", "STLFSI", "D1_feat→target",
            )
            resultados.append(res)
            modelos_guardar.update(res["modelos"])
        except Exception as exc:
            if DEBUG:
                print(f"[DEBUG] D1 B/STLFSI falló: {exc}")

    # ── D2: estrés → cada RWA individual ────────────────────
    print("\n[DEBUG] ── D2: estrés → RWA ──")

    macro_feat_a = {
        c: feat_a[c] for c in COLS_MACRO_FEAT
        if c in feat_a
    }
    macro_feat_b = {
        c: feat_b[c] for c in COLS_MACRO_FEAT
        if c in feat_b
    }

    for rwa_col in COLS_RWA_ALL:
        # Pista A
        if rwa_col in feat_a and len(feat_a[rwa_col]) > 20:
            try:
                res = correr_modelo_target(
                    macro_feat_a, feat_a[rwa_col],
                    K_FEAT_A, AR_P_A,
                    "A", rwa_col, "D2_target→feat",
                )
                resultados.append(res)
                modelos_guardar.update(res["modelos"])
            except Exception as exc:
                if DEBUG:
                    print(
                        f"[DEBUG] D2 A/{rwa_col} falló: {exc}"
                    )

        # Pista B
        if rwa_col in feat_b and len(feat_b[rwa_col]) > 10:
            try:
                res = correr_modelo_target(
                    macro_feat_b, feat_b[rwa_col],
                    K_FEAT_B, AR_P_B,
                    "B", rwa_col, "D2_target→feat",
                )
                resultados.append(res)
                modelos_guardar.update(res["modelos"])
            except Exception as exc:
                if DEBUG:
                    print(
                        f"[DEBUG] D2 B/{rwa_col} falló: {exc}"
                    )

    # ── Consolidar métricas y coeficientes ──────────────────
    df_metricas = pd.DataFrame(
        [m for r in resultados for m in r["metricas"]]
    )
    # CAMBIO E2E-04-01: sin modelos evaluables, conservar el
    # esquema que necesita el reporte (sin inventar observaciones).
    if df_metricas.empty:
        df_metricas = pd.DataFrame(columns=[
            "direccion", "pista", "target", "modelo", "rmse",
        ])
        if DEBUG:
            print("[DEBUG][ALERTA] Job 4: sin metricas modelables")
    _coefs_list = [
        r["coeficientes"] for r in resultados
        if not r["coeficientes"].empty
    ]
    df_coefs = (
        pd.concat(_coefs_list, ignore_index=True)
        if _coefs_list else pd.DataFrame()
    )

    if not df_metricas.empty and DEBUG:
        print("\n[DEBUG] Métricas (resumen):")
        _cols_met = [
            "direccion", "pista", "target", "modelo",
            "n_train", "n_features", "n_test",
            "rmse", "r2_oos",
        ]
        _cols_met = [
            c for c in _cols_met if c in df_metricas.columns
        ]
        print(df_metricas[_cols_met].to_string(index=False))

    print("\n[DEBUG] Features retenidas por Lasso:")
    if df_coefs.empty:
        print(
            "  → Lasso descartó TODAS las features en todas "
            "las corridas."
        )
    else:
        print(df_coefs.to_string(index=False))
        print(
            f"\n  → {len(df_coefs)} features retenidas en "
            f"total across {len(resultados)} modelos"
        )

    df_top_feat = features_mas_retenidas(df_coefs)
    if not df_top_feat.empty:
        print("\n[DEBUG] Features más retenidas por Lasso:")
        print(df_top_feat.to_string(index=False))

    # ════════════════════════════════════════════════════════
    # PASO 5 — Granger bidireccional + FDR global
    # ════════════════════════════════════════════════════════
    print("\n[DEBUG] ══ PASO 5: Granger bidireccional ══")

    # CAMBIO I5-1: dos acumuladores — filas_granger_rezago (toda
    # la tabla por rezago, apoyo, NO entra a BH) y
    # filas_granger (familia: 1 fila por par×dirección×pista,
    # en su rezago elegido por BIC — ver granger_par_bidir).
    filas_granger_rezago: list[dict] = []
    filas_granger: list[dict] = []

    def _acumular(rezago: list[dict], familia: list[dict]) -> None:
        filas_granger_rezago.extend(rezago)
        filas_granger.extend(familia)

    # RWA vs estrés (ambas pistas)
    rwa_feat_a = {
        c: feat_a[c] for c in COLS_RWA_ALL if c in feat_a
    }
    rwa_feat_b = {
        c: feat_b[c] for c in COLS_RWA_ALL if c in feat_b
    }

    # Pista A: cada RWA vs ΔCISS
    if not y_ciss_a.empty:
        for col, serie in rwa_feat_a.items():
            _acumular(*granger_par_bidir(
                serie, y_ciss_a, K_GRANGER_D,
                col, "ΔCISS", "A",
            ))

    # Pista B: cada RWA vs STLFSI
    if not y_stlfsi_b.empty:
        for col, serie in rwa_feat_b.items():
            _acumular(*granger_par_bidir(
                serie, y_stlfsi_b, K_GRANGER_S,
                col, "STLFSI", "B",
            ))

    # Macro features vs cada RWA (ambas pistas)
    for rwa_col in COLS_RWA_ALL:
        # Pista A
        if rwa_col in feat_a:
            for macro_col, macro_s in macro_feat_a.items():
                _acumular(*granger_par_bidir(
                    macro_s, feat_a[rwa_col], K_GRANGER_D,
                    macro_col, rwa_col, "A",
                ))
        # Pista B
        if rwa_col in feat_b:
            for macro_col, macro_s in macro_feat_b.items():
                _acumular(*granger_par_bidir(
                    macro_s, feat_b[rwa_col], K_GRANGER_S,
                    macro_col, rwa_col, "B",
                ))

    df_granger_rezago = pd.DataFrame(filas_granger_rezago)
    df_granger = pd.DataFrame(filas_granger)
    if not df_granger.empty:
        df_granger = aplicar_bh_global(df_granger)
    else:
        # CAMBIO E2E-04-01: CSV legible incluso con cero pares.
        # Mismos nombres esperados por Jobs 10 y 11, cero filas.
        df_granger = pd.DataFrame(columns=[
            "pista", "serie_feat", "serie_target", "direccion",
            "n", "frec_nativa_feat", "lag", "pval_f",
            "pval_chi2", "nota", "bh_q", "bh_rechaza",
            "ambito_bh",
        ])
        if DEBUG:
            print("[DEBUG][ALERTA] Job 4: Granger sin pares evaluables")
    if df_granger_rezago.empty:
        df_granger_rezago = pd.DataFrame(columns=[
            "pista", "serie_feat", "serie_target", "direccion",
            "n", "frec_nativa_feat", "lag", "pval_f",
            "pval_chi2", "nota",
        ])

    _p = pd.to_numeric(
        df_granger["pval_f"], errors="coerce"
    ) if not df_granger.empty else pd.Series(dtype=float)
    n_tests_global = int(_p.notna().sum())
    n_sig_crudo = int((_p < ALFA_BH).sum())
    n_sig_bh = int(
        df_granger["bh_rechaza"].fillna(False).astype(bool).sum()
    ) if not df_granger.empty else 0

    print(
        f"\n[DEBUG] Granger bidireccional — familia: "
        f"{n_tests_global} tests | p<{ALFA_BH} sin corregir: "
        f"{n_sig_crudo} | FDR GLOBAL: {n_sig_bh}"
    )
    # CAMBIO I5-4 (I5-d): distribución de rezagos elegidos.
    if not df_granger.empty and "lag" in df_granger.columns:
        _dist_lag = (
            df_granger["lag"].dropna().astype(int)
            .value_counts().sort_index()
        )
        if not _dist_lag.empty:
            print(
                "[DEBUG] Distribución de rezagos elegidos "
                f"(BIC): {_dist_lag.to_dict()}"
            )

    # CAMBIO I5-3: familia (BH) y tabla de apoyo por rezago.
    if SAVE_DATA:
        os.makedirs(DIR_REPORTES, exist_ok=True)
        # CAMBIO E2E-04-01: siempre publicar los encabezados;
        # un resultado vacio reemplaza a un CSV de otra corrida.
        df_granger.to_csv(ARCHIVO_GRANGER, index=False)
        df_granger_rezago.to_csv(
            ARCHIVO_GRANGER_REZAGO, index=False
        )
        if DEBUG:
            print(f"[DEBUG] guardado: {ARCHIVO_GRANGER}")
            print(f"[DEBUG] guardado: {ARCHIVO_GRANGER_REZAGO}")

    for dir_label in ["D1_feat→target", "D2_target→feat"]:
        if df_granger.empty:
            break
        sub = df_granger[df_granger["direccion"] == dir_label]
        _pv = pd.to_numeric(sub["pval_f"], errors="coerce")
        _nt = int(_pv.notna().sum())
        _nc = int((_pv < ALFA_BH).sum())
        _ns = int(
            sub["bh_rechaza"].fillna(False).astype(bool).sum()
        )
        print(
            f"[DEBUG]   {dir_label}: {_nt} tests | "
            f"p<0.05 crudo: {_nc} | FDR: {_ns}"
        )

    if n_sig_bh > 0:
        print("\n[DEBUG] Pares que sobreviven FDR global:")
        _sig = df_granger[df_granger["bh_rechaza"]]
        _cols_sig = [
            "pista", "serie_feat", "serie_target",
            "direccion", "lag", "pval_f", "bh_q", "n",
        ]
        _cols_sig = [
            c for c in _cols_sig if c in _sig.columns
        ]
        print(_sig[_cols_sig].to_string(index=False))

    # TOP señales
    df_top = top_señales(df_granger)
    if df_top.empty:
        # CAMBIO E2E-04-01: CSV con esquema y cero filas.
        df_top = pd.DataFrame(columns=df_granger.columns)
    if not df_top.empty:
        print(f"\n[DEBUG] TOP {TOP_N_SEÑALES} señales (menor bh_q):")
        _cols_top = [
            "pista", "serie_feat", "serie_target",
            "direccion", "lag", "pval_f", "bh_q",
            "bh_rechaza", "n",
        ]
        _cols_top = [
            c for c in _cols_top if c in df_top.columns
        ]
        print(df_top[_cols_top].to_string(index=False))

    # ════════════════════════════════════════════════════════
    # PASO 6 — Reporte final
    # ════════════════════════════════════════════════════════
    print("\n" + "═" * 64)
    print(
        "REPORTE FINAL — MODELO COMPLETO "
        "(MACRO + RWA ↔ ESTRÉS)"
    )
    print("═" * 64)
    print(
        f"\nRango: {df.index.min().date()} → "
        f"{df.index.max().date()} ({len(df)} días)"
    )
    print(f"Columnas: {df.shape[1]}")
    print(
        f"Series FRED ({MODO_DATOS}): "  # CAMBIO B1-3
        f"{len(series_fred)}/{len(FRED_SERIES)}"
    )
    print(
        f"Granger: {n_tests_global} tests | "
        f"p<{ALFA_BH} crudo: {n_sig_crudo} | "
        f"FDR GLOBAL: {n_sig_bh}"
    )

    for dir_label in ["D1_feat→target", "D2_target→feat"]:
        ev = evaluar_corroboracion_direccion(
            dir_label, df_granger, df_metricas, df_coefs,
        )
        print(f"\n{'─' * 64}")
        print(f"DIRECCIÓN: {dir_label}")
        print(f"{'─' * 64}")
        criterios = [
            (
                "C1  Lasso retiene ≥1 feature",
                ev["c1_lasso_retiene"],
            ),
            (
                f"C2  Granger sobrevive FDR global "
                f"({ev['n_sig_bh']}/{ev['n_tests']})",
                ev["c2_granger_fdr"],
            ),
            (
                f"C3  Mejora material ≥{UMBRAL_MEJORA_MATERIAL}% "
                f"({ev['mejora_max_pct']:.2f}%)",
                ev["c3_mejora_material"],
            ),
            (
                "C4  Consistente en ambas pistas",
                ev["c4_ambas_pistas"],
            ),
        ]
        for nombre_c, ok_c in criterios:
            print(f"  [{'SÍ' if ok_c else 'NO'}] {nombre_c}")

        if ev["corroborada"]:
            print(
                f"\n{dir_label}: hay evidencia de "
                "aporte predictivo corroborada."
            )

        else:
            print(
                f"\n{dir_label}: sin poder predictivo "
                "robusto."
            )

    # TOP señales aunque no sobrevivan FDR
    if not df_top.empty:
        print(f"\n{'─' * 64}")
        print(
            f"TOP {TOP_N_SEÑALES} INDICIOS (menor bh_q, "
            "aunque no sobrevivan FDR):"
        )
        print("─" * 64)
        for _, row in df_top.iterrows():
            sobrevive = "SÍ" if row.get(
                "bh_rechaza", False
            ) else "NO"
            print(
                f"  {row['serie_feat']} → {row['serie_target']} "
                f"({row['direccion']}, Pista {row['pista']}): "
                f"lag={row['lag']}, p={row['pval_f']:.6f}, "
                f"q={row['bh_q']:.6f}, FDR: {sobrevive}"
            )

    if not df_top_feat.empty:
        print(f"\n{'─' * 64}")
        print("FEATURES MÁS RETENIDAS POR LASSO:")
        print("─" * 64)
        print(df_top_feat.to_string(index=False))

    print(f"\n{'═' * 64}")
    print(f"{NOTA_RWA}")
    print(
        "Tendencias comunes (macro global) pueden generar "
        "correlación espuria en ambas direcciones."
    )
    print(
        "FDR global sobre un universo muy grande de tests: "
        "potencia baja para señales débiles reales."
    )
    print("═" * 64)

    # ════════════════════════════════════════════════════════
    # Guardar
    # ════════════════════════════════════════════════════════
    if SAVE_DATA:
        os.makedirs(DIR_REPORTES, exist_ok=True)
        os.makedirs(DIR_MODELOS, exist_ok=True)

        guardados: list[str] = []

        # CAMBIO E2E-04-01: siempre escribir salidas CSV
        # actuales (encabezados + cero filas si no hay datos).
        df_granger.to_csv(ARCHIVO_GRANGER, index=False)
        guardados.append(ARCHIVO_GRANGER)

        df_metricas.to_csv(ARCHIVO_MET, index=False)
        guardados.append(ARCHIVO_MET)

        df_top.to_csv(ARCHIVO_TOP, index=False)
        guardados.append(ARCHIVO_TOP)

        for nombre, modelo in modelos_guardar.items():
            if modelo is None:
                continue
            ruta = os.path.join(
                DIR_MODELOS, f"{nombre}.joblib"
            )
            joblib.dump(modelo, ruta)
            guardados.append(ruta)

        if DEBUG:
            print("\n[DEBUG] Archivos guardados:")
            for g in guardados:
                print(f"  {g}")
    elif DEBUG:
        print("\n[DEBUG] SAVE_DATA=False — nada guardado")


if __name__ == "__main__":
    main()

