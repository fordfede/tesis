"""# Job 9 - Modelo Bidireccional: RWA ↔ Estrés Financiero"""

# ════════════════════════════════════════════════════════════
# 1. Configuración
# ════════════════════════════════════════════════════════════

DEBUG: bool = True
SAVE_DATA: bool = True

# ════════════════════════════════════════════════════════════
# 2. Imports
# ════════════════════════════════════════════════════════════

import contextlib
import io
import os
import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR  # CAMBIO I5-1 (Job 7)
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller  # CAMBIO I5-6 (Job 7)
from statsmodels.tsa.stattools import grangercausalitytests

from google.colab import drive
drive.mount("/content/drive")

# ════════════════════════════════════════════════════════════
# 3. Paths y Constantes
# ════════════════════════════════════════════════════════════

"""# 3. Paths y Constantes"""

# CAMBIO E2E-09-01: ruta coherente con los Jobs 0-8; mismo valor por defecto.
BASE_DIR: str = os.environ.get(
    "TESIS_BASE_DIR", "/content/drive/MyDrive/Tesis/expansion"
)

DIR_PROCESSED: str = os.path.join(BASE_DIR, "data", "processed")
DIR_REPORTES: str = os.path.join(
    BASE_DIR, "outputs", "reportes"
)
DIR_MODELOS: str = os.path.join(BASE_DIR, "outputs", "modelos")

ARCHIVO_LARGAS: str = os.path.join(
    DIR_PROCESSED, "series_largas_ext.csv"
)
ARCHIVO_GRANGER: str = os.path.join(
    DIR_REPORTES, "granger_bidireccional.csv"
)
# CAMBIO E2E-09-01: tabla de apoyo propia del Job 9 (fuera de BH).
# Evita pisar granger_por_rezago.csv del Job 4 y la tabla del Job 8.
ARCHIVO_GRANGER_REZAGO: str = os.path.join(
    DIR_REPORTES, "granger_bidireccional_por_rezago.csv"
)
ARCHIVO_MET: str = os.path.join(
    DIR_REPORTES, "metricas_bidireccional.csv"
)

# ── Features RWA ────────────────────────────────────────────
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

# ── Targets ─────────────────────────────────────────────────
COL_CISS: str = "macro_euro_ciss"
COL_STLFSI: str = "macro_eeuu_stlfsi"

# ── Granger ─────────────────────────────────────────────────
K_GRANGER_D: int = 14   # Pista A diaria
K_GRANGER_S: int = 8    # Pista B semanal

ALFA_BH: float = 0.05

# ── Modelo predictivo ──────────────────────────────────────
K_FEAT_A: int = 7    # Lags de features diarios
K_FEAT_B: int = 4    # Lags de features semanales
N_CV_SPLITS: int = 5

# CAMBIO H4 (documentación, sin cambio de comportamiento):
# Jobs 9/10 usan un único AR_LAGS_CONTROL=2 para ambas
# pistas, en vez de un valor por pista. No se unificó entre
# Jobs: el usuario confirmó dejarlo así y documentarlo.
AR_P_A: int = 3
AR_P_B: int = 2

# CAMBIO C3-1 (Job 7): grilla de Ridge (reemplaza ALPHAS).
# Lasso ya no usa grilla fija (alphas=None) — ver modelo_lasso.
RIDGE_ALPHAS: np.ndarray = np.logspace(-6, 3, 30)

UMBRAL_MEJORA_MATERIAL: float = 5.0
# CAMBIO B3-4 (Bloque 3): n mínimo del test para evaluar la
# "mejora material". Valor confirmado por el autor (2026-10-27)
# antes de correr; no se ajusta mirando resultados. Debajo de este n la
# mejora se marca "no evaluable" y no cuenta para corroborar.
MIN_N_TEST: int = 30
NO_EVALUABLE: str = "no evaluable"

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


def verificar_entradas_snapshot(rutas: list[str], job: str) -> None:
    """
    Compara el SHA-256 de cada entrada con manifest.json. Aborta
    nombrando el archivo si falta o si el hash no coincide. Si el
    archivo no figura en el manifest: aborta en modo snapshot y
    avisa con [DEBUG][ALERTA] en modo descarga.

    CAMBIO B1-1 (Bloque 1).

    Args:
        rutas: CSV de entrada del Job.
        job: Job que verifica (ej. "Job 5").
    """
    _archivos = leer_manifest()["archivos"]
    for _ruta in rutas:
        _nombre = os.path.basename(_ruta)
        if not os.path.isfile(_ruta):
            raise FileNotFoundError(
                f"[DEBUG][ERROR] {job}: falta la entrada {_ruta}."
            )
        _entrada = _archivos.get(_nombre)
        if _entrada is None:
            _msg = f"{job}: {_nombre} no figura en {ARCHIVO_MANIFEST}"
            if MODO_DATOS == "snapshot":
                raise RuntimeError(f"[DEBUG][ERROR] {_msg}.")
            print(f"[DEBUG][ALERTA] {_msg}: entrada sin verificar.")
            continue
        _sha = sha256_archivo(_ruta)
        if _sha != _entrada.get("sha256"):
            raise RuntimeError(
                f"[DEBUG][ERROR] {job}: el hash de {_ruta} no coincide "
                f"con el snapshot ({_sha[:12]}... vs "
                f"{str(_entrada.get('sha256'))[:12]}...). Re-ejecutar "
                f"el {_entrada.get('job')} o restaurar el snapshot."
            )
        print(
            f"[DEBUG] {job}: entrada verificada {_nombre} | "
            f"{_entrada.get('filas')} filas | {_entrada.get('fecha_min')} "
            f"→ {_entrada.get('fecha_max')} | sha256 {_sha[:12]}..."
        )


# ════════════════════════════════════════════════════════════
# 4. Funciones — Carga y transformación
# ════════════════════════════════════════════════════════════


def validar_entrada_job9(df: pd.DataFrame, ruta: str) -> None:
    """CAMBIO E2E-09-01: contrato de entrada sin alterar el DataFrame.

    Requiere CISS y al menos una RWA observada. STLFSI sigue siendo
    opcional, al igual que Ripple, tal como en el Job original.
    """
    if df.empty:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 9: CSV sin observaciones: {ruta} "
            "(productor: Job 3)."
        )
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 9: fechas invalidas en {ruta} "
            "(productor: Job 3)."
        )
    if df.index.has_duplicates:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 9: fechas duplicadas en {ruta} "
            "(productor: Job 3)."
        )
    if COL_CISS not in df.columns:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 9: falta columna obligatoria "
            f"{COL_CISS} en {ruta} (productor: Job 3)."
        )
    if df[COL_CISS].notna().sum() == 0:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 9: {COL_CISS} no tiene datos "
            f"en {ruta} (productor: Job 3)."
        )
    rwa_disponibles = [col for col in COLS_RWA_ALL if col in df.columns]
    if not rwa_disponibles or not df[rwa_disponibles].notna().any().any():
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 9: no hay series RWA con datos "
            f"en {ruta} (productor: Job 3)."
        )
    if COL_STLFSI not in df.columns and DEBUG:
        print(
            f"[DEBUG][ALERTA] Job 9: {COL_STLFSI} ausente "
            "(Pista B sin target; se omite como en el original)."
        )


def cargar_series(ruta: str) -> pd.DataFrame:
    df = pd.read_csv(ruta, parse_dates=["fecha"])
    df = df.set_index("fecha").sort_index()
    if DEBUG:
        print(f"[DEBUG] {os.path.basename(ruta)}: {df.shape}")
    return df


def _log_diff(serie: pd.Series) -> pd.Series:
    return np.log(serie.where(serie > 0)).diff()


def preparar_datos(
    df_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Alinea a grilla diaria, recorta filas pre-RWA y aplica
    ffill (sin interpolar).
    """
    df = df_raw.asfreq("D")
    df = df.ffill()

    rwa_present = [c for c in COLS_RWA_ALL if c in df.columns]
    if rwa_present:
        mask_all_nan = df[rwa_present].isna().all(axis=1)
        first_valid = (~mask_all_nan).idxmax()
        df = df.loc[first_valid:].copy()
        if DEBUG:
            print(
                f"[DEBUG] recorte pre-RWA: desde "
                f"{first_valid.date()}, {len(df)} días"
            )
    return df


def series_pista_a(
    df: pd.DataFrame,
) -> tuple[dict[str, pd.Series], pd.Series]:
    """
    Pista A (diaria): features RWA log-dif, target ΔCISS.
    """
    x: dict[str, pd.Series] = {}
    for col in COLS_RWA_ALL:
        if col in df.columns:
            x[col] = _log_diff(df[col]).dropna()
    y = df[COL_CISS].diff().dropna()
    return x, y


def series_pista_b(
    df: pd.DataFrame,
) -> tuple[dict[str, pd.Series], pd.Series]:
    """
    Pista B (semanal): RWA W.last() → log-dif, STLFSI nivel
    W.last().
    """
    x: dict[str, pd.Series] = {}
    for col in COLS_RWA_ALL:
        if col not in df.columns:
            continue
        serie_sem = df[col].resample("W").last()
        x[col] = _log_diff(serie_sem).dropna()

    y = pd.Series(dtype=float)
    if COL_STLFSI in df.columns:
        y = df[COL_STLFSI].resample("W").last().dropna()
    return x, y


# ════════════════════════════════════════════════════════════
# 5. Funciones — Granger bidireccional
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
    Elige UN rezago (1..max_lag) por BIC, ajustando un VAR(p)
    sobre [y, x] con VAR.select_order.

    CAMBIO I5-1: reemplaza el criterio anterior de tratar cada
    rezago 1..max_lag como un test independiente. BIC penaliza
    más la complejidad que AIC, así que para N chico (pista
    semanal) tiende a elegir rezagos más bajos.

    Args:
        serie_y: Serie y (target).
        serie_x: Serie x (feature).
        max_lag: Rezago máximo a considerar.

    Returns:
        Rezago elegido (1..max_lag). Si no hay datos
        suficientes o VAR falla, devuelve 1 (y lo registra con
        [DEBUG]).
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
    Corre ADF sobre una serie ya transformada (la que entra al
    test de Granger) y registra si NO resultó estacionaria.

    CAMBIO I5-6 (I5-f): antes de cada test se asume que la
    transformación de series_pista_a/series_pista_b (log-dif /
    nivel) ya es estacionaria; esto lo verifica en vez de darlo
    por sentado.

    Args:
        serie: Serie ya transformada.
        etiqueta: Nombre descriptivo para el log.
        alpha: Nivel de significancia ADF.

    Returns:
        True si es estacionaria (p<alpha); False si no, o si no
        se pudo calcular (muy pocos datos).
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
            print(
                f"[DEBUG][ALERTA] {etiqueta}: ADF falló: {exc}"
            )
        return False
    estacionaria = float(pval) < alpha
    if not estacionaria and DEBUG:
        print(
            f"[DEBUG][ALERTA] {etiqueta}: NO estacionaria tras "
            f"su transformación elegida (ADF p={pval:.4f})"
        )
    return estacionaria


def granger_par_bidir(
    serie_rwa: pd.Series,
    serie_estres: pd.Series,
    max_lag: int,
    nombre_rwa: str,
    nombre_estres: str,
    pista: str,
) -> tuple[list[dict], list[dict]]:
    """
    Granger bidireccional para un par (RWA, estrés).

    D1: RWA → estrés (x=RWA rezagada predice y=estrés).
    D2: estrés → RWA (x=estrés rezagado predice y=RWA).

    CAMBIO I5-1/I5-6: ahora devuelve (filas_por_rezago,
    filas_familia). filas_por_rezago es la tabla de apoyo de
    siempre (1 fila por rezago 1..max_lag, NO entra a BH).
    filas_familia tiene 1 fila por dirección, en el rezago
    elegido por BIC (elegir_rezago_bic) — esa es la familia que
    entra a Benjamini-Hochberg una sola vez. Antes de elegir el
    rezago se verifica que x e y (ya transformadas) sean
    estacionarias (I5-f); si no lo son, se registra pero el test
    se corre igual (no se corta la ejecución).
    """
    filas: list[dict] = []
    familia: list[dict] = []

    pares = [
        ("D1_rwa→estres", serie_estres, serie_rwa),
        ("D2_estres→rwa", serie_rwa, serie_estres),
    ]

    for direccion, y, x in pares:
        data = pd.concat(
            [y.rename("y"), x.rename("x")], axis=1
        ).dropna()
        n_min = max_lag + 10
        fila_base = {
            "pista": pista,
            "serie_rwa": nombre_rwa,
            "serie_estres": nombre_estres,
            "direccion": direccion,
            "n": len(data),
        }
        if len(data) < n_min:
            filas.extend([
                {
                    **fila_base, "lag": k,
                    "pval_f": None, "pval_chi2": None,
                    "nota": f"n={len(data)}<{n_min} insuficiente",
                }
                for k in range(1, max_lag + 1)
            ])
            familia.append({
                **fila_base, "lag": None, "pval_f": None,
                "nota": f"n={len(data)}<{n_min} insuficiente",
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


def tabla_granger_bidir(
    x_a: dict[str, pd.Series],
    y_a: pd.Series,
    x_b: dict[str, pd.Series],
    y_b: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    CAMBIO I5-1: devuelve (df_por_rezago, df_familia) en vez de
    una sola tabla — ver granger_par_bidir.
    """
    filas: list[dict] = []
    familia: list[dict] = []
    for col, serie in x_a.items():
        _f, _fam = granger_par_bidir(
            serie, y_a, K_GRANGER_D, col, "ΔCISS", "A"
        )
        filas.extend(_f)
        familia.extend(_fam)
    if not y_b.empty:
        for col, serie in x_b.items():
            _f, _fam = granger_par_bidir(
                serie, y_b, K_GRANGER_S, col, "STLFSI", "B"
            )
            filas.extend(_f)
            familia.extend(_fam)
    return pd.DataFrame(filas), pd.DataFrame(familia)


def aplicar_bh_global(
    df: pd.DataFrame,
    col_p: str = "pval_f",
    alfa: float = ALFA_BH,
) -> pd.DataFrame:
    """
    CAMBIO I5-1: `df` ahora es la familia (1 fila por par ×
    dirección × pista, en su rezago elegido por BIC) — ya no
    incluye los rezagos descartados.
    """
    df = df.copy().reset_index(drop=True)
    df["bh_q"] = np.nan
    df["bh_rechaza"] = False
    df["ambito_bh"] = (
        "global: 2 direcciones × 2 pistas × features "
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
        ["pista", "serie_rwa", "serie_estres", "direccion"]
    )["pval_f"].idxmin()
    return df_granger.loc[idx].copy()


# ════════════════════════════════════════════════════════════
# 6. Funciones — Modelo predictivo bidireccional
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
    direccion: str,
) -> dict:
    yt = y_true.to_numpy(dtype=float)
    yp = y_pred.to_numpy(dtype=float)
    mask = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[mask], yp[mask]
    base = {
        "pista": pista,
        "direccion": direccion,
        "modelo": nombre,
    }
    if len(yt) == 0:
        return {
            **base, "n_test": 0, "rmse": None,
            "mae": None, "r2_oos": None,
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
    return (
        y_all.shift(1).loc[y_test.index].rename("persistencia")
    )


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
        if DEBUG:
            print(f"[DEBUG] AR({p}) {etiqueta} OK")
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
    return max(2, min(N_CV_SPLITS, n_train // 15))


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


def correr_direccion(
    x_dict: dict[str, pd.Series],
    y_series: pd.Series,
    K: int,
    ar_p: int,
    pista: str,
    direccion: str,
    target_nombre: str,
) -> dict:
    """
    Corre baselines y modelos para UNA dirección en UNA pista.

    En D1 (RWA→estrés): X = RWA rezagadas, y = estrés.
    En D2 (estrés→RWA): X = estrés rezagado, y = cada RWA
    individual. Se corre un modelo por cada RWA como target
    y se reporta el mejor.
    """
    if direccion.startswith("D1"):
        return _correr_d1(
            x_dict, y_series, K, ar_p, pista, direccion,
            target_nombre,
        )
    else:
        return _correr_d2(
            x_dict, y_series, K, ar_p, pista, direccion,
        )


def _correr_d1(
    x_dict: dict[str, pd.Series],
    y_series: pd.Series,
    K: int,
    ar_p: int,
    pista: str,
    direccion: str,
    target_nombre: str,
) -> dict:
    """
    D1 — modo CONJUNTO: todas las RWA a la vez como features →
    un target de estrés. El n efectivo es la intersección tras
    rezagar TODAS las features juntas (hoy ≈166 en pista diaria,
    ≈19 en semanal) — mucho menor que el mínimo entre series
    individuales (ver n_a_min_series/n_b_min_series en main()).
    Para el n por serie individual, ver _correr_d1_individual.
    """
    cols_ok = list(x_dict.keys())
    if not cols_ok:
        raise ValueError(f"{pista}/{direccion}: sin features")

    df_x = pd.concat(
        [s.rename(n) for n, s in x_dict.items()], axis=1
    )
    X_lag = construir_features_lag(df_x, K)
    etiqueta = f"{pista}/{direccion}"
    if DEBUG:
        # CAMBIO I5-5 (I5-e): n real usado por el modelo
        # conjunto, para no confundirlo con el mínimo por serie.
        print(
            f"[DEBUG] {etiqueta} (conjunto, {len(cols_ok)} "
            f"features): n_conjunto={len(X_lag)}"
        )

    try:
        Xtr, Xts, ytr, yts = split_temporal(X_lag, y_series)
    except ValueError as exc:
        if DEBUG:
            print(f"[DEBUG] {etiqueta} split falló: {exc}")
        return {"metricas": [], "coeficientes": pd.DataFrame(),
                "modelos": {}}

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
        metricas_regresion(yts, serie, nombre, pista, direccion)
        for nombre, serie in preds.items()
    ]
    _borde_por_modelo = {"ridge": ridge_borde, "lasso": lasso_borde}
    for m in metricas:
        m["n_train"] = int(len(ytr))
        m["n_features"] = int(Xtr.shape[1])
        m["target"] = target_nombre
        m["modo"] = "conjunto"  # CAMBIO I5-5
        m["baseline_ok"] = (
            ar_ok if m["modelo"] == pred_ar.name else True
        )
        m["alpha_en_borde"] = _borde_por_modelo.get(
            m["modelo"], False
        )

    df_coef = coeficientes_lasso(
        mdl_lasso, list(Xtr.columns), pista, direccion,
    )

    return {
        "metricas": metricas,
        "coeficientes": df_coef,
        "modelos": {
            f"modelo_ridge_{pista.lower()}_{direccion.lower()}":
                mdl_ridge,
            f"modelo_lasso_{pista.lower()}_{direccion.lower()}":
                mdl_lasso,
        },
    }


def _correr_d1_individual(
    x_dict: dict[str, pd.Series],
    y_series: pd.Series,
    K: int,
    ar_p: int,
    pista: str,
    direccion: str,
    target_nombre: str,
) -> dict:
    """
    D1 — modo INDIVIDUAL (CAMBIO I5-5/I5-e): una RWA a la vez
    como feature → el mismo target de estrés. Cada serie corre
    con SU propio n (no la intersección de las 8 RWA), lo que
    evita perder la mayoría de las observaciones por culpa de la
    RWA con menos historia.

    Returns:
        Dict con "metricas" (una fila por modelo × feature,
        modo="individual") y "modelos" (ridge/lasso por feature).
    """
    metricas_todas: list[dict] = []
    modelos: dict[str, Any] = {}
    for nombre, serie in x_dict.items():
        etiqueta = f"{pista}/{direccion}/individual/{nombre}"
        X_lag = construir_features_lag(serie.to_frame(nombre), K)
        if DEBUG:
            print(
                f"[DEBUG] {etiqueta}: n_individual={len(X_lag)}"
            )
        try:
            Xtr, Xts, ytr, yts = split_temporal(X_lag, y_series)
        except ValueError as exc:
            if DEBUG:
                print(f"[DEBUG] {etiqueta} split falló: {exc}")
            continue

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

        _borde_por_modelo = {
            "ridge": ridge_borde, "lasso": lasso_borde,
        }
        for modelo_nombre, serie_pred in preds.items():
            m = metricas_regresion(
                yts, serie_pred, modelo_nombre, pista, direccion,
            )
            m["n_train"] = int(len(ytr))
            m["n_features"] = int(Xtr.shape[1])
            m["target"] = target_nombre
            m["modo"] = "individual"
            m["feature_individual"] = nombre
            m["baseline_ok"] = (
                ar_ok if modelo_nombre == pred_ar.name else True
            )
            m["alpha_en_borde"] = _borde_por_modelo.get(
                modelo_nombre, False
            )
            metricas_todas.append(m)

        tag = f"{pista.lower()}_{direccion.lower()}_{nombre}"
        if mdl_ridge is not None:
            modelos[f"modelo_ridge_{tag}"] = mdl_ridge
        if mdl_lasso is not None:
            modelos[f"modelo_lasso_{tag}"] = mdl_lasso

    return {
        "metricas": metricas_todas,
        "coeficientes": pd.DataFrame(),
        "modelos": modelos,
    }


def _correr_d2(
    x_dict_rwa: dict[str, pd.Series],
    y_estres: pd.Series,
    K: int,
    ar_p: int,
    pista: str,
    direccion: str,
) -> dict:
    """
    D2: estrés como feature → cada RWA individual como target.

    Se corre un modelo por cada variable RWA y se consolidan
    las métricas. Se reporta el mejor resultado.
    """
    all_metricas: list[dict] = []
    all_coefs: list[pd.DataFrame] = []
    all_modelos: dict[str, Any] = {}

    df_estres = y_estres.to_frame("estres")
    X_lag = construir_features_lag(df_estres, K)

    for nombre_rwa, serie_rwa in x_dict_rwa.items():
        etiqueta = f"{pista}/{direccion}/{nombre_rwa}"
        try:
            Xtr, Xts, ytr, yts = split_temporal(
                X_lag, serie_rwa
            )
        except ValueError as exc:
            if DEBUG:
                print(f"[DEBUG] {etiqueta} skip: {exc}")
            continue

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
                yts, serie, nombre, pista, direccion,
            )
            for nombre, serie in preds.items()
        ]
        _borde_por_modelo = {
            "ridge": ridge_borde, "lasso": lasso_borde,
        }
        for m in metricas:
            m["n_train"] = int(len(ytr))
            m["n_features"] = int(Xtr.shape[1])
            m["target"] = nombre_rwa
            m["baseline_ok"] = (
                ar_ok if m["modelo"] == pred_ar.name else True
            )
            m["alpha_en_borde"] = _borde_por_modelo.get(
                m["modelo"], False
            )

        all_metricas.extend(metricas)

        df_coef = coeficientes_lasso(
            mdl_lasso, list(Xtr.columns), pista, direccion,
        )
        if not df_coef.empty:
            df_coef["target"] = nombre_rwa
            all_coefs.append(df_coef)

        sfx = nombre_rwa.replace("rwa_", "")
        tag = f"{pista.lower()}_{direccion.lower()}_{sfx}"
        if mdl_ridge is not None:
            all_modelos[f"modelo_ridge_{tag}"] = mdl_ridge
        if mdl_lasso is not None:
            all_modelos[f"modelo_lasso_{tag}"] = mdl_lasso

    df_coef_all = (
        pd.concat(all_coefs, ignore_index=True)
        if all_coefs else pd.DataFrame()
    )

    return {
        "metricas": all_metricas,
        "coeficientes": df_coef_all,
        "modelos": all_modelos,
    }


# ════════════════════════════════════════════════════════════
# 7. Funciones — Veredicto
# ════════════════════════════════════════════════════════════


def evaluar_corroboracion_direccion(
    direccion: str,
    df_granger: pd.DataFrame,
    df_met: pd.DataFrame,
    df_coefs: pd.DataFrame,
) -> dict:
    """
    Evalúa los 4 criterios de corroboración para UNA dirección.

    C1: Lasso retiene ≥1 feature.
    C2: ≥1 par sobrevive FDR global.
    C3: mejora ≥UMBRAL_MEJORA_MATERIAL% RMSE vs AR, solo con
        n_test ≥ MIN_N_TEST; si ningún modelo llega, C3 es
        "no evaluable" y la dirección no se corrobora (B3-4).
    C4: resultado consistente en ambas pistas.
    """
    sub_g = df_granger[df_granger["direccion"] == direccion]
    sub_m = df_met[df_met["direccion"] == direccion]
    sub_c = (
        df_coefs[df_coefs["direccion"] == direccion]
        if not df_coefs.empty else pd.DataFrame()
    )

    # C1
    c1 = not sub_c.empty

    # C2
    n_sig = 0
    n_tests = 0
    if not sub_g.empty:
        pvals = pd.to_numeric(sub_g["pval_f"], errors="coerce")
        n_tests = int(pvals.notna().sum())
        n_sig = int(
            sub_g["bh_rechaza"].fillna(False).astype(bool).sum()
        )
    c2 = n_sig > 0

    # C3
    mejora_max = 0.0
    mejor_modelo = "n/a"
    r2_mejor: float | None = None
    n_eval_c3 = 0
    if not sub_m.empty:
        ar_rows = sub_m[
            sub_m["modelo"].str.startswith("ar")
        ].dropna(subset=["rmse"])
        feat_rows = sub_m[
            sub_m["modelo"].isin(["ridge", "lasso"])
        ].dropna(subset=["rmse"])
        # CAMBIO B3-4: solo modelos con n_test >= MIN_N_TEST.
        _chicos = feat_rows[feat_rows["n_test"] < MIN_N_TEST]
        if not _chicos.empty:
            print(
                f"[DEBUG][ALERTA] Job 9 {direccion}: {len(_chicos)} "
                f"modelo(s) con n_test < MIN_N_TEST={MIN_N_TEST} "
                "(mejora material no evaluable): "
                f"{sorted(set(_chicos['pista']))}"
            )
        feat_rows = feat_rows[feat_rows["n_test"] >= MIN_N_TEST]
        n_eval_c3 = len(feat_rows)
        if not ar_rows.empty and not feat_rows.empty:
            for pista_val in sub_m["pista"].unique():
                ar_p = ar_rows[ar_rows["pista"] == pista_val]
                ft_p = feat_rows[feat_rows["pista"] == pista_val]
                if ar_p.empty or ft_p.empty:
                    continue
                rmse_ar = float(ar_p["rmse"].min())
                if rmse_ar <= 0:
                    continue
                for _, fila in ft_p.iterrows():
                    rmse_f = float(fila["rmse"])
                    mejora = 100.0 * (rmse_ar - rmse_f) / rmse_ar
                    if mejora > mejora_max:
                        mejora_max = mejora
                        mejor_modelo = str(fila["modelo"])
                        r2_mejor = (
                            float(fila["r2_oos"])
                            if pd.notna(fila.get("r2_oos")) else None
                        )
    c3: bool | str = (
        NO_EVALUABLE if n_eval_c3 == 0
        else mejora_max >= UMBRAL_MEJORA_MATERIAL
    )

    # C4
    c4 = False
    if not sub_m.empty:
        feat_rows = sub_m[
            sub_m["modelo"].isin(["ridge", "lasso"])
        ].dropna(subset=["rmse"])
        ar_rows = sub_m[
            sub_m["modelo"].str.startswith("ar")
        ].dropna(subset=["rmse"])
        mejora_por_pista: dict[str, bool] = {}
        for pista_val in sub_m["pista"].unique():
            ar_p = ar_rows[ar_rows["pista"] == pista_val]
            ft_p = feat_rows[feat_rows["pista"] == pista_val]
            if ar_p.empty or ft_p.empty:
                mejora_por_pista[pista_val] = False
                continue
            rmse_ar = float(ar_p["rmse"].min())
            rmse_best = float(ft_p["rmse"].min())
            mejora_por_pista[pista_val] = rmse_best < rmse_ar
        c4 = all(mejora_por_pista.values()) and len(
            mejora_por_pista
        ) >= 2

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
        "r2_mejor": r2_mejor,  # CAMBIO B3-1
        "corroborada": bool(c1 and c2 and c3 is True and c4),
    }


# ════════════════════════════════════════════════════════════
# 8. Proceso principal
# ════════════════════════════════════════════════════════════


def main() -> None:
    # CAMBIO B1-4 (Bloque 1): hash de las entradas vs. manifest.
    verificar_entradas_snapshot([ARCHIVO_LARGAS], "Job 9")

    print(
        "\n[DEBUG] ═══════════════════════════════════════════"
        "═══════════════"
    )
    print(
        "[DEBUG] 17 — Modelo Bidireccional: RWA ↔ Estrés "
        "Financiero"
    )
    print(
        "[DEBUG] ═══════════════════════════════════════════"
        "═══════════════"
    )

    # ── Carga ───────────────────────────────────────────────
    # CAMBIO E2E-09-01: entrada del Job 3 obligatoria y validada.
    if not os.path.isfile(ARCHIVO_LARGAS):
        raise FileNotFoundError(
            f"[DEBUG][ERROR] Job 9: falta {ARCHIVO_LARGAS}. "
            "Debe producirlo el Job 3."
        )
    try:
        df_raw = cargar_series(ARCHIVO_LARGAS)
    except (OSError, ValueError, KeyError, TypeError,
            pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 9: no se pudo leer "
            f"{ARCHIVO_LARGAS} (productor: Job 3): {exc}"
        ) from exc
    validar_entrada_job9(df_raw, ARCHIVO_LARGAS)
    df = preparar_datos(df_raw)
    if df.empty:
        raise RuntimeError(
            "[DEBUG][ERROR] Job 9: preparar_datos() produjo "
            "cero observaciones."
        )

    print(
        f"[DEBUG] rango: {df.index.min().date()} → "
        f"{df.index.max().date()} ({len(df)} días)"
    )

    # ── Transformaciones ────────────────────────────────────
    print("\n[DEBUG] ── Preparación Pista A (diaria) ──")
    x_a, y_a = series_pista_a(df)
    # CAMBIO I5-5 (I5-e): esto es el MÍNIMO entre series
    # individuales, no el n que efectivamente usa el modelo
    # conjunto de D1 (ese n sale de la intersección tras
    # construir_features_lag — ver _correr_d1 / n_conjunto_*).
    n_a_min_series = min(
        len(y_a),
        min(len(s) for s in x_a.values()) if x_a else 0,
    )
    print(
        f"[DEBUG] Pista A: n mínimo entre series ≈ "
        f"{n_a_min_series} (NO es el n usado en los tests — "
        "ver n_conjunto_a más abajo)"
    )

    print("\n[DEBUG] ── Preparación Pista B (semanal) ──")
    x_b, y_b = series_pista_b(df)
    n_b_min_series = min(
        len(y_b),
        min(len(s) for s in x_b.values()) if x_b else 0,
    )
    print(
        f"[DEBUG] Pista B: n mínimo entre series ≈ "
        f"{n_b_min_series} (NO es el n usado en los tests — "
        "ver n_conjunto_b más abajo)"
    )

    # ════════════════════════════════════════════════════════
    # BLOQUE 1 — Granger bidireccional
    # ════════════════════════════════════════════════════════
    print(
        "\n[DEBUG] ══ BLOQUE 1: Granger bidireccional + "
        "FDR global ══"
    )
    # CAMBIO I5-1: tabla_granger_bidir devuelve (por_rezago,
    # familia). por_rezago es apoyo (granger_bidireccional_por_rezago.csv, no
    # entra a BH); familia (1 fila por par×dirección×pista, en
    # su rezago elegido por BIC) es la que se corrige con BH.
    df_granger_rezago, df_granger = tabla_granger_bidir(
        x_a, y_a, x_b, y_b
    )
    # CAMBIO E2E-09-01: familia vacia = 0 tests, no KeyError en BH.
    if df_granger.empty:
        df_granger = pd.DataFrame(columns=[
            "pista", "serie_rwa", "serie_estres", "direccion",
            "n", "lag", "pval_f", "pval_chi2", "nota",
        ])
    df_granger = aplicar_bh_global(df_granger)

    _p = pd.to_numeric(df_granger["pval_f"], errors="coerce")
    n_tests_global = int(_p.notna().sum())
    n_sig_crudo = int((_p < ALFA_BH).sum())
    n_sig_bh = int(
        df_granger["bh_rechaza"].fillna(False).astype(bool).sum()
    )
    print(
        f"\n[DEBUG] Granger bidireccional (test F) — familia: "
        f"{n_tests_global} tests | p<{ALFA_BH} sin corregir: "
        f"{n_sig_crudo} | sobreviven FDR GLOBAL: {n_sig_bh}"
    )
    # CAMBIO I5-4 (I5-d): distribución de rezagos elegidos.
    if "lag" in df_granger.columns:
        _dist_lag = (
            df_granger["lag"].dropna().astype(int)
            .value_counts().sort_index()
        )
        if not _dist_lag.empty:
            print(
                "[DEBUG] Distribución de rezagos elegidos "
                f"(BIC): {_dist_lag.to_dict()}"
            )

    df_granger_opt = df_granger
    if not df_granger_opt.empty:
        _cols_show = [
            "pista", "serie_rwa", "serie_estres", "direccion",
            "lag", "pval_f", "bh_q", "bh_rechaza", "n",
        ]
        _cols_show = [
            c for c in _cols_show if c in df_granger_opt.columns
        ]
        print("\n[DEBUG] Rezago elegido por par y dirección:")
        print(df_granger_opt[_cols_show].to_string(index=False))

    if n_sig_bh > 0:
        print("\n[DEBUG] Pares que sobreviven FDR global:")
        _sig = df_granger[df_granger["bh_rechaza"]]
        _cols_sig = [
            "pista", "serie_rwa", "direccion",
            "lag", "pval_f", "bh_q", "n",
        ]
        _cols_sig = [
            c for c in _cols_sig if c in _sig.columns
        ]
        print(_sig[_cols_sig].to_string(index=False))
    else:
        print(
            "\n[DEBUG] → Ningún par sobrevive el FDR global "
            "en ninguna dirección."
        )

    # Por dirección
    for dir_label in ["D1_rwa→estres", "D2_estres→rwa"]:
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

    # ════════════════════════════════════════════════════════
    # BLOQUE 2 — Modelos predictivos bidireccionales
    # ════════════════════════════════════════════════════════
    print(
        "\n[DEBUG] ══ BLOQUE 2: Modelos predictivos "
        "bidireccionales ══"
    )

    resultados: list[dict] = []
    modelos_guardar: dict[str, Any] = {}

    # D1: RWA → estrés
    print("\n[DEBUG] ── D1: RWA → estrés ──")
    for pista, x_dict, y_ser, K, ar_p, target_n in [
        ("A", x_a, y_a, K_FEAT_A, AR_P_A, "ΔCISS"),
        ("B", x_b, y_b, K_FEAT_B, AR_P_B, "STLFSI"),
    ]:
        if y_ser.empty:
            if DEBUG:
                print(f"[DEBUG] Pista {pista} D1: sin target")
            continue
        # CAMBIO E2E-09-01: pista sin features -> no evaluable.
        if not x_dict:
            if DEBUG:
                print(f"[DEBUG][ALERTA] Job 9: Pista {pista} D1 sin RWA")
            continue
        try:
            res = correr_direccion(
                x_dict, y_ser, K, ar_p, pista,
                "D1_rwa→estres", target_n,
            )
            resultados.append(res)
            modelos_guardar.update(res["modelos"])
            # CAMBIO I5-5 (I5-e): modo por serie individual,
            # además del modelo conjunto (no lo reemplaza).
            res_ind = _correr_d1_individual(
                x_dict, y_ser, K, ar_p, pista,
                "D1_rwa→estres", target_n,
            )
            resultados.append(res_ind)
            modelos_guardar.update(res_ind["modelos"])
        except Exception as exc:
            raise RuntimeError(
                f"[DEBUG][ERROR] Job 9: fallo inesperado "
                f"en D1 Pista {pista}: {exc}"
            ) from exc

    # D2: estrés → RWA
    print("\n[DEBUG] ── D2: estrés → RWA ──")
    for pista, x_dict, y_ser, K, ar_p in [
        ("A", x_a, y_a, K_FEAT_A, AR_P_A),
        ("B", x_b, y_b, K_FEAT_B, AR_P_B),
    ]:
        if y_ser.empty:
            if DEBUG:
                print(f"[DEBUG] Pista {pista} D2: sin target")
            continue
        # CAMBIO E2E-09-01: pista sin features -> no evaluable.
        if not x_dict:
            if DEBUG:
                print(f"[DEBUG][ALERTA] Job 9: Pista {pista} D2 sin RWA")
            continue
        try:
            res = correr_direccion(
                x_dict, y_ser, K, ar_p, pista,
                "D2_estres→rwa", "",
            )
            resultados.append(res)
            modelos_guardar.update(res["modelos"])
        except Exception as exc:
            raise RuntimeError(
                f"[DEBUG][ERROR] Job 9: fallo inesperado "
                f"en D2 Pista {pista}: {exc}"
            ) from exc

    # ── Consolidar métricas y coeficientes ──────────────────
    df_metricas = pd.DataFrame(
        [m for r in resultados for m in r["metricas"]]
    )
    # CAMBIO E2E-09-01: sin modelos evaluables -> CSV header-only.
    if df_metricas.empty:
        df_metricas = pd.DataFrame(columns=[
            "pista", "direccion", "modelo", "n_test", "rmse",
            "mae", "r2_oos", "n_train", "n_features", "target",
            "modo", "baseline_ok", "alpha_en_borde",
            "feature_individual",
        ])
        print(
            "[DEBUG][ALERTA] Job 9: no hubo modelos evaluables; "
            "no interpretar como resultado predictivo negativo."
        )
    _coefs = [
        r["coeficientes"] for r in resultados
        if not r["coeficientes"].empty
    ]
    df_coefs = (
        pd.concat(_coefs, ignore_index=True)
        if _coefs else pd.DataFrame()
    )

    if not df_metricas.empty:
        print("\n[DEBUG] Métricas por dirección y pista:")
        _cols_met = [
            "direccion", "pista", "modelo", "target",
            "n_train", "n_features", "n_test",
            "rmse", "mae", "r2_oos",
        ]
        _cols_met = [
            c for c in _cols_met if c in df_metricas.columns
        ]
        print(df_metricas[_cols_met].to_string(index=False))

    print("\n[DEBUG] Features retenidas por Lasso:")
    if df_coefs.empty:
        if df_metricas.empty:
            print("  → Lasso no evaluable: no se ajustaron modelos.")
        else:
            print(
                "  → Lasso no retuvo features en las corridas "
                "evaluables (o fallaron ajustes individuales)."
            )
    else:
        print(df_coefs.to_string(index=False))

    # ════════════════════════════════════════════════════════
    # REPORTE FINAL
    # ════════════════════════════════════════════════════════
    print("\n" + "═" * 64)
    print(
        "REPORTE FINAL — MODELO BIDIRECCIONAL "
        "(RWA ↔ ESTRÉS FINANCIERO)"
    )
    print("═" * 64)
    print(
        f"\nRango: {df.index.min().date()} → "
        f"{df.index.max().date()}"
    )
    print(
        f"Pista A: n mínimo entre series ≈ {n_a_min_series} "
        "diario | "
        f"Pista B: n mínimo entre series ≈ {n_b_min_series} "
        "semanal (el n real de D1 conjunto es menor — ver "
        "[DEBUG] n_conjunto arriba)"
    )
    print(
        f"Granger bidireccional (test F): "
        f"{n_tests_global} tests | p<{ALFA_BH} crudo: "
        f"{n_sig_crudo} | FDR GLOBAL: {n_sig_bh}"
    )

    for dir_label in ["D1_rwa→estres", "D2_estres→rwa"]:
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
                + (  # CAMBIO B3-4
                    f"(n_test < {MIN_N_TEST})"
                    if ev["c3_mejora_material"] == NO_EVALUABLE
                    else f"({ev['mejora_max_pct']:.2f}%)"
                ),
                ev["c3_mejora_material"],
            ),
            (
                "C4  Consistente en ambas pistas",
                ev["c4_ambas_pistas"],
            ),
        ]
        for nombre_c, ok_c in criterios:
            _marca = (
                "N/E" if ok_c == NO_EVALUABLE  # CAMBIO B3-4
                else ("SÍ" if ok_c else "NO")
            )
            print(f"  [{_marca}] {nombre_c}")

        # CAMBIO B3-1 (Bloque 3): el texto sale de los criterios y
        # del R² OOS del mejor modelo, no de una frase fija.
        _r2 = ev["r2_mejor"]
        if ev["corroborada"] and _r2 is not None and _r2 > 0:
            print(
                f"\nVEREDICTO {dir_label}: hay evidencia de "
                "aporte predictivo corroborada por los 4 "
                f"criterios (mejor modelo {ev['mejor_modelo']}, "
                f"R² OOS={_r2:.3f})."
            )
            print(
                "CAUTELA: historia larga incluye tendencias "
                "comunes (macro global), adopción incipiente "
                "de RWA y posible correlación espuria. "
                "No se afirma causalidad económica."
            )
        elif ev["corroborada"]:
            print(
                f"\nVEREDICTO {dir_label}: los 4 criterios se "
                f"cumplen, pero el mejor modelo ({ev['mejor_modelo']}) "
                f"tiene R² OOS={_r2} (≤ 0 o no disponible): mejora "
                "relativa al AR, sin poder predictivo frente a la "
                "media del test."
            )
        else:
            _fallan = [
                nombre_c.split()[0] for nombre_c, ok_c in criterios
                if ok_c is not True
            ]
            _extra = (
                f" (C3 no evaluable: n_test < {MIN_N_TEST})"
                if ev["c3_mejora_material"] == NO_EVALUABLE else ""
            )
            print(
                f"\nVEREDICTO {dir_label}: sin poder predictivo "
                f"robusto en esta dirección; no se cumple "
                f"{', '.join(_fallan)}{_extra}."
            )

    print(f"\n{'═' * 64}")
    print(f"{NOTA_RWA}")
    print(
        "Tendencias comunes (macro global) pueden generar "
        "correlación espuria en ambas direcciones."
    )
    print("═" * 64)

    # ════════════════════════════════════════════════════════
    # Guardar
    # ════════════════════════════════════════════════════════
    if SAVE_DATA:
        os.makedirs(DIR_REPORTES, exist_ok=True)
        os.makedirs(DIR_MODELOS, exist_ok=True)

        guardados: list[str] = []

        # CAMBIO E2E-09-01: 0 filas -> CSV con encabezados.
        # Las tablas pobladas se guardan exactamente como antes.
        _tablas_salida: list[tuple[str, pd.DataFrame, list[str]]] = [
            (
                ARCHIVO_GRANGER, df_granger,
                ["pista", "serie_rwa", "serie_estres", "direccion",
                 "n", "lag", "pval_f", "pval_chi2", "nota",
                 "bh_q", "bh_rechaza", "ambito_bh"],
            ),
            (
                ARCHIVO_GRANGER_REZAGO, df_granger_rezago,
                ["pista", "serie_rwa", "serie_estres", "direccion",
                 "n", "lag", "pval_f", "pval_chi2", "nota"],
            ),
            (
                ARCHIVO_MET, df_metricas,
                ["pista", "direccion", "modelo", "n_test", "rmse",
                 "mae", "r2_oos", "n_train", "n_features", "target",
                 "modo", "baseline_ok", "alpha_en_borde",
                 "feature_individual"],
            ),
        ]
        for ruta, tabla, encabezados in _tablas_salida:
            actual = tabla if not tabla.empty else pd.DataFrame(columns=encabezados)
            actual.to_csv(ruta, index=False)
            guardados.append(ruta)

        for nombre, modelo in modelos_guardar.items():
            if modelo is None:
                continue
            ruta = os.path.join(DIR_MODELOS, f"{nombre}.joblib")
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

