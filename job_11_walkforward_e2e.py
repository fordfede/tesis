"""# Job 11 - Validación Walk-Forward de Pares Convergentes"""

# ════════════════════════════════════════════════════════════
# 1. Configuración
# ════════════════════════════════════════════════════════════

DEBUG: bool = True
SAVE_DATA: bool = True

# ════════════════════════════════════════════════════════════
# 2. Imports
# ════════════════════════════════════════════════════════════

import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg

from google.colab import drive

drive.mount("/content/drive")

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

# ════════════════════════════════════════════════════════════
# 3. Paths y Constantes
# ════════════════════════════════════════════════════════════

# CAMBIO E2E-11-01: directorio configurable, mismo predeterminado.
BASE_DIR: str = os.environ.get(
    "TESIS_BASE_DIR", "/content/drive/MyDrive/Tesis/expansion"
)
DIR_PROCESSED: str = os.path.join(BASE_DIR, "data", "processed")
DIR_REPORTES: str = os.path.join(
    BASE_DIR, "outputs", "reportes"
)

ARCHIVO_COMPLETAS: str = os.path.join(
    DIR_PROCESSED, "series_completas.csv"
)
ARCHIVO_GRANGER: str = os.path.join(
    DIR_REPORTES, "granger_completo.csv"
)
ARCHIVO_MET_NL: str = os.path.join(
    DIR_REPORTES, "metricas_nolineal.csv"
)
ARCHIVO_WF_MET: str = os.path.join(
    DIR_REPORTES, "walkforward_metricas.csv"
)
ARCHIVO_WF_PRED: str = os.path.join(
    DIR_REPORTES, "walkforward_predicciones.csv"
)
ARCHIVO_CONV: str = os.path.join(
    DIR_REPORTES, "convergencia_niveles.csv"
)

# CAMBIO C5-1/C5-2/C5-3 (Job 10): PARES ya no se escribe a
# mano — se construye desde granger_completo.csv (Job 4) más
# abajo, con construir_pares_desde_granger(). Ver esa función
# para el criterio completo.
MIN_OBS_PAR: int = 100
MAX_PARES_WF: int = 20
ARCHIVO_PARES_WF: str = os.path.join(
    DIR_REPORTES, "walkforward_pares.csv"
)

# ── Transformaciones (mismas que notebooks 18/19) ──────────

TRANSFORM: dict[str, str] = {
    "rwa_usdy_supply": "log_diff",
    "rwa_red_stellar": "log_diff",
    "rwa_red_ripple": "log_diff",
    "rwa_red_algorand": "log_diff",
    "rwa_tipo_treasurybills": "log_diff",
    "rwa_tipo_commodities": "log_diff",
    "rwa_tipo_privatecredit": "log_diff",
    "rwa_tipo_stocksetfs": "log_diff",
    "rwa_total_global": "log_diff",
    "macro_fed_funds": "diff",
    "macro_sofr": "diff",
    "macro_tsy_1m": "diff",
    "macro_tsy_3m": "diff",
    "macro_tsy_2y": "diff",
    "macro_tsy_10y": "diff",
    "macro_spread_10y2y": "diff",
    "macro_infl_exp_10y": "diff",
    "macro_rrp_overnight": "log_diff",
    "macro_nfci": "nivel",
    "macro_spread_hy": "diff",
    "macro_spread_ig": "diff",
    "macro_indpro": "log_diff",
    "macro_umcsent": "diff",
    "macro_euro_ciss": "diff",
    "macro_eeuu_stlfsi": "nivel",
}

# CAMBIO B2-1 (Bloque 2): Job 4 escribe en granger_completo.csv el
# target de estrés diario con su etiqueta ("ΔCISS"), no con el
# nombre de columna de series_completas.csv, y el par se salteaba
# como "ausente". Se traduce la etiqueta a su columna solo para leer
# los datos; la transformación es la misma de Job 4 (TRANSFORM
# "diff" en pista A: y_ciss_a = df[COL_CISS].diff().dropna()).
# En los reportes el target conserva la etiqueta de Job 4.
# CAMBIO B2-5: idem "STLFSI" (pista B: W.last() en nivel, igual que
# y_stlfsi_b de Job 4).
COLUMNA_DE_ETIQUETA: dict[str, str] = {
    "ΔCISS": "macro_euro_ciss",
    "STLFSI": "macro_eeuu_stlfsi",
}

# CAMBIO C4-7 (Job 10): frecuencia nativa de cada columna macro
# (ver nota completa en Job 8). PARES tiene 3 pares con una
# feature de pista incompatible (macro_indpro×2 y macro_umcsent
# en pista "A", mensuales diferenciados a diario → casi todos
# ceros; macro_nfci en pista "A", semanal) — se detectan y se
# saltean en el walk-forward, no se borran de PARES.
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
    "macro_euro_ciss": "D",
    "macro_eeuu_stlfsi": "W",
}
_RANGO_FRECUENCIA: dict[str, int] = {"D": 0, "W": 1, "M": 2}
_PISTA_A_FRECUENCIA: dict[str, str] = {"A": "D", "B": "W", "M": "M"}


def frecuencia_admite_pista(col: str, pista: str) -> bool:
    """
    True si la frecuencia nativa de `col` es igual o más fina
    que la de `pista` (A→diaria, B→semanal, M→mensual).

    CAMBIO C4-7 (Job 10). Columnas no RWA/onchain fuera de
    FRECUENCIA_NATIVA se consideran diarias (siempre califican).
    """
    nativa = FRECUENCIA_NATIVA.get(col)
    if nativa is None:
        return True
    freq_pista = _PISTA_A_FRECUENCIA.get(pista, "D")
    return _RANGO_FRECUENCIA[nativa] <= _RANGO_FRECUENCIA[freq_pista]


# ── Walk-forward ────────────────────────────────────────────
RATIO_TRAIN_INICIAL: float = 0.60
# CAMBIO H4 (documentación, sin cambio de comportamiento):
# mismo orden AR para ambas pistas acá. Jobs 5/7/8 usan
# AR_P_A=3 (diaria) / AR_P_B=2 (semanal) — valor distinto por
# pista. No se unificó entre Jobs: el usuario confirmó dejarlo
# así y documentarlo.
AR_LAGS_CONTROL: int = 2
UMBRAL_MEJORA: float = 5.0
# CAMBIO B3-4 (Bloque 3): n mínimo del test para evaluar la
# "mejora material". Valor confirmado por el autor (2026-10-27)
# antes de correr; no se ajusta mirando resultados. Debajo de este n la
# mejora se marca "no evaluable" y el par no entra a la
# convergencia.
MIN_N_TEST: int = 30
NO_EVALUABLE: str = "no evaluable"

RF_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 5,
    "random_state": 42,
    "n_jobs": -1,
}

XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "random_state": 42,
}

# CAMBIO H4 (documentación, sin cambio de comportamiento):
# acá Ridge usa un alpha fijo, sin CV, a diferencia de Jobs
# 5/7/8 (RidgeCV con grilla RIDGE_ALPHAS). Es a propósito: el
# walk-forward re-ajusta el modelo en cada paso (cientos de
# pasos por par); correr una CV interna en cada paso sería
# mucho más lento. El usuario confirmó dejarlo fijo y
# documentarlo en vez de cambiar el comportamiento.
RIDGE_ALPHA: float = 100.0

# CAMBIO E2E-11-01: contratos de salida (solo se aplican con 0 filas).
# Un DataFrame con registros sigue escribiendose exactamente como antes.
COLUMNAS_WF_MET: list[str] = [
    "par", "pista", "modelo", "n_wf", "rmse", "mae", "r2_oos",
    "frec_nativa_feat", "mejora_vs_ar_pct",
]
COLUMNAS_WF_PRED: list[str] = [
    "fecha", "par", "pista", "real", f"ar{AR_LAGS_CONTROL}",
    "ridge", "random_forest",
    "xgboost" if _HAS_XGB else "gradient_boost",
]
COLUMNAS_CONVERGENCIA: list[str] = [
    "par", "pista", "lag",
    "n2_rmse", "n2_mejora_pct", "n2_mejora_material",
    "n1_bh_q", "n1_pval_f", "n1_sobrevive",
    "n3_best_r2", "n3_best_modelo", "n3_mejora_pct",
    "n3_mejora_material", "n4_best_rmse", "n4_best_r2",
    "n4_best_modelo", "n4_n_wf", "n4_mejora_pct",
    "n4_supera_ar", "n4_mejora_material", "n4_r2_negativo",
    "converge_laxo", "converge_4_niveles",
    "n2_n_test", "n3_n_test", "convergencia_evaluable",  # CAMBIO B3-4
]


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


def cargar_series(ruta: str) -> pd.DataFrame:
    df = pd.read_csv(ruta, parse_dates=["fecha"])
    df = df.set_index("fecha").sort_index()
    if DEBUG:
        print(f"[DEBUG] {os.path.basename(ruta)}: {df.shape}")
    return df


# CAMBIO E2E-11-01: validaciones de entradas y escritura consistente.
def validar_series_job11(df: pd.DataFrame, ruta: str) -> None:
    """Valida forma y eje temporal; no modifica fechas ni valores."""
    if df.empty or len(df.columns) == 0:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 11: {ruta} sin series validas "
            "(productor: Job 4)."
        )
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 11: fechas invalidas en {ruta} "
            "(productor: Job 4)."
        )
    if df.index.has_duplicates:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 11: fechas duplicadas en {ruta} "
            "(productor: Job 4)."
        )


def validar_granger_job11(df: pd.DataFrame, ruta: str) -> None:
    """Valida el contrato del Job 4 sin cambiar el filtro D1/FDR/n."""
    requeridas = {
        "pista", "serie_feat", "serie_target", "direccion", "n",
        "lag", "bh_q", "bh_rechaza", "pval_f",
    }
    faltantes = sorted(requeridas - set(df.columns))
    if faltantes:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 11: columnas {faltantes} ausentes "
            f"en {ruta} (productor: Job 4)."
        )
    if df.empty:
        return  # Cero tests evaluables es una tabla valida, no archivo ausente.
    if df["bh_rechaza"].isna().any() or not df["bh_rechaza"].map(
        lambda v: isinstance(v, (bool, np.bool_))
    ).all():
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 11: bh_rechaza invalido en {ruta}; "
            "se esperan valores booleanos True/False."
        )
    sig = df.loc[df["bh_rechaza"] == True]  # noqa: E712
    if sig.empty:
        return
    if sig[["serie_feat", "serie_target", "pista", "direccion"]].isna().any().any():
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 11: identificador ausente en "
            f"prueba significativa de {ruta}."
        )
    for col in ("n", "lag", "bh_q", "pval_f"):
        numeros = pd.to_numeric(sig[col], errors="coerce")
        if numeros.isna().any() or not np.isfinite(numeros).all():
            raise RuntimeError(
                f"[DEBUG][ERROR] Job 11: {col} invalido en "
                f"pruebas significativas de {ruta}."
            )
    for col in ("n", "lag"):
        numeros = pd.to_numeric(sig[col], errors="coerce")
        if (numeros < 1).any() or (numeros != np.floor(numeros)).any():
            raise RuntimeError(
                f"[DEBUG][ERROR] Job 11: {col} debe ser entero "
                f"positivo en {ruta}."
            )


def validar_metricas_nl_job11(df: pd.DataFrame, ruta: str) -> None:
    """Admite un CSV de Job 10 con encabezados y cero modelos."""
    requeridas = {"par", "pista", "modelo", "rmse", "r2_oos"}
    faltantes = sorted(requeridas - set(df.columns))
    if faltantes:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 11: columnas {faltantes} ausentes "
            f"en {ruta} (productor: Job 10)."
        )
    if df.empty:
        return
    if df[["par", "pista", "modelo"]].isna().any().any():
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 11: identificadores ausentes en "
            f"{ruta} (productor: Job 10)."
        )
    for col in ("rmse", "r2_oos"):
        reales = df[col].dropna()
        if pd.to_numeric(reales, errors="coerce").isna().any():
            raise RuntimeError(
                f"[DEBUG][ERROR] Job 11: {col} no numerico "
                f"en {ruta} (productor: Job 10)."
            )


def cargar_y_validar_entradas_job11() -> pd.DataFrame:
    """Falla al inicio si una salida de Job 4/10 falta o esta mal formada."""
    dependencias = (
        (ARCHIVO_COMPLETAS, "Job 4"),
        (ARCHIVO_GRANGER, "Job 4"),
        (ARCHIVO_MET_NL, "Job 10"),
    )
    for ruta, productor in dependencias:
        if not os.path.isfile(ruta):
            raise FileNotFoundError(
                f"[DEBUG][ERROR] Job 11: falta {ruta}. "
                f"Debe generarlo el {productor}."
            )
    for ruta, lector, validador in (
        (ARCHIVO_COMPLETAS, cargar_series, validar_series_job11),
        (ARCHIVO_GRANGER, pd.read_csv, validar_granger_job11),
        (ARCHIVO_MET_NL, pd.read_csv, validar_metricas_nl_job11),
    ):
        try:
            datos = lector(ruta)
        except (OSError, ValueError, KeyError, UnicodeError,
                pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise RuntimeError(
                f"[DEBUG][ERROR] Job 11: no se puede leer "
                f"{ruta}: {exc}"
            ) from exc
        validador(datos, ruta)
        if ruta == ARCHIVO_COMPLETAS:
            df_completas = datos
    return df_completas


def guardar_tablas_job11(
    df_met: pd.DataFrame, df_pred: pd.DataFrame,
    df_conv: pd.DataFrame,
) -> list[str]:
    """Publica las 3 salidas; con 0 filas conserva un esquema legible."""
    os.makedirs(DIR_REPORTES, exist_ok=True)
    for tabla, columnas, ruta in (
        (df_met, COLUMNAS_WF_MET, ARCHIVO_WF_MET),
        (df_pred, COLUMNAS_WF_PRED, ARCHIVO_WF_PRED),
        (df_conv, COLUMNAS_CONVERGENCIA, ARCHIVO_CONV),
    ):
        actual = tabla if not tabla.empty else pd.DataFrame(columns=columnas)
        actual.to_csv(ruta, index=False)
    return [ARCHIVO_WF_MET, ARCHIVO_WF_PRED, ARCHIVO_CONV]


def _log_diff(serie: pd.Series) -> pd.Series:
    return np.log(serie.where(serie > 0)).diff()


def transformar(serie: pd.Series, col: str) -> pd.Series:
    tipo = TRANSFORM.get(col, "diff")
    if tipo == "log_diff":
        return _log_diff(serie)
    elif tipo == "diff":
        return serie.diff()
    elif tipo == "nivel":
        return serie.copy()
    return serie.diff()


def n_alineado_par(
    df: pd.DataFrame, feat_col: str, target_col: str, pista: str,
) -> int:
    """
    n del par (feature, target) transformado y alineado, calculado
    como en granger_par_bidir de Job 4 (pista B: W.last() antes de
    transformar), para comparar con la columna "n" de
    granger_completo.csv.

    CAMBIO B2-1 (Bloque 2); CAMBIO B2-5: pista B.

    Args:
        df: series_completas.csv indexado por fecha.
        feat_col: Columna del feature.
        target_col: Columna (o etiqueta de Job 4) del target.
        pista: "A" (diaria) o "B" (semanal).

    Returns:
        Cantidad de fechas con feature y target válidos.
    """
    target_col = COLUMNA_DE_ETIQUETA.get(target_col, target_col)
    s_feat = df[feat_col]
    s_target = df[target_col]
    if pista == "B":
        s_feat = s_feat.resample("W").last()
        s_target = s_target.resample("W").last()
    s_feat = transformar(s_feat, feat_col).dropna()
    s_target = transformar(s_target, target_col).dropna()
    return len(pd.concat([s_target, s_feat], axis=1).dropna())


def informar_n_vs_job4(
    df: pd.DataFrame, feat_col: str, target_col: str,
    pista: str, n_job4: int | None, etiqueta: str,
) -> None:
    """
    Para targets traducidos (ΔCISS, STLFSI), compara el n alineado
    con el n que informó Job 4 y avisa con [DEBUG][ALERTA] si difiere.

    CAMBIO B2-1 (Bloque 2); CAMBIO B2-5: pista B.
    """
    if target_col not in COLUMNA_DE_ETIQUETA or pista not in ("A", "B"):
        return
    _n = n_alineado_par(df, feat_col, target_col, pista)
    if n_job4 is None:
        print(f"[DEBUG] {etiqueta}: n alineado={_n} (Job 4: sin n)")
    elif _n != n_job4:
        print(
            f"[DEBUG][ALERTA] {etiqueta}: n alineado={_n} distinto "
            f"del n de Job 4={n_job4}"
        )
    else:
        print(f"[DEBUG] {etiqueta}: n alineado={_n} = n de Job 4 OK")


def construir_xy_par(
    df: pd.DataFrame,
    feat_col: str,
    target_col: str,
    lag_optimo: int,
    pista: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construye X e y para un par. Misma lógica que notebook 19:
    X = [feature_lag1..lag_optimo, target_lag1, target_lag2].
    y = target en t.
    """
    target_col = COLUMNA_DE_ETIQUETA.get(target_col, target_col)  # B2-1
    if pista == "B":
        s_feat = df[feat_col].resample("W").last()
        s_target = df[target_col].resample("W").last()
    else:
        s_feat = df[feat_col].copy()
        s_target = df[target_col].copy()

    s_feat_t = transformar(s_feat, feat_col).dropna()
    s_target_t = transformar(s_target, target_col).dropna()

    partes: list[pd.Series] = []
    for k in range(1, lag_optimo + 1):
        partes.append(
            s_feat_t.shift(k).rename(f"{feat_col}_lag{k}")
        )
    for k in range(1, AR_LAGS_CONTROL + 1):
        partes.append(
            s_target_t.shift(k).rename(f"{target_col}_lag{k}")
        )

    df_x = pd.concat(partes, axis=1)
    combinado = pd.concat(
        [s_target_t.rename("__y"), df_x], axis=1
    ).dropna()

    y = combinado.pop("__y")
    X = combinado
    return X, y


def construir_pares_desde_granger(
    ruta_granger: str,
    min_obs: int = MIN_OBS_PAR,
    max_pares: int = MAX_PARES_WF,
) -> list[dict[str, Any]]:
    """
    Construye la lista de pares a correr en walk-forward a
    partir de granger_completo.csv (Job 4), en vez de escribirla
    a mano.

    CAMBIO C5-1: toma los pares con bh_rechaza == True en la
    dirección "D1_feat→target" (feat Granger-causa target — la
    dirección relevante acá: macro/RWA → RWA), con al menos
    `min_obs` observaciones en el test. El rezago usado es el
    mismo que eligió BIC en Job 4 (columna "lag", ya corregido
    en F4) — construir_xy_par lo recibe tal cual, sin volver a
    elegirlo (CAMBIO C5-4).

    Direcciones de granger_completo.csv (CAMBIO B3-5, solo
    documentación; el filtro no cambia):
      · D1_feat→target: serie_feat Granger-causa a serie_target
        (el feature anticipa al target). Es la única que se usa.
      · D2_target→feat: la inversa; se descarta acá.
    feat/target dependen de la familia de Job 4: macro→RWA o
    RWA→ΔCISS/STLFSI. "macro/RWA → RWA" arriba describe solo la
    primera familia; con la segunda, D1 es RWA→estrés.

    CAMBIO C5-2: si sobreviven más de `max_pares`, se ordena por
    bh_q ascendente (más significativo primero) y se usan solo
    los primeros; se registra cuántos quedan afuera.

    CAMBIO C5-3: guarda la lista de pares usada y el criterio en
    ARCHIVO_PARES_WF (walkforward_pares.csv).

    Args:
        ruta_granger: Ruta de granger_completo.csv.
        min_obs: Mínimo de observaciones del test de Granger
            para considerar el par.
        max_pares: Máximo de pares a correr en walk-forward.

    Returns:
        Lista de dicts {feat, target, lag, pista}, mismo formato
        que el PARES escrito a mano que reemplaza.
    """
    if not os.path.exists(ruta_granger):
        raise FileNotFoundError(
            f"[DEBUG][ERROR] Job 11: falta {ruta_granger}. "
            "Debe producirlo el Job 4 (granger_completo.csv)."
        )
    df_g = pd.read_csv(ruta_granger)

    requeridas = {
        "pista", "serie_feat", "serie_target", "direccion",
        "n", "lag", "bh_q", "bh_rechaza",
    }
    faltantes = requeridas - set(df_g.columns)
    if faltantes:
        raise KeyError(
            f"granger_completo.csv no tiene las columnas "
            f"{sorted(faltantes)} — verificar salida del Job 4 "
            "(granger_completo.csv)."
        )

    _es_d1 = df_g["direccion"] == "D1_feat→target"
    _sobrevive_bh = df_g["bh_rechaza"] == True  # noqa: E712

    cand = df_g[
        _sobrevive_bh & _es_d1
        & (df_g["n"] >= min_obs)
        & df_g["lag"].notna()
    ].copy()

    n_descartados_obs = int(
        (_sobrevive_bh & _es_d1 & (df_g["n"] < min_obs)).sum()
    )
    if DEBUG:
        print(
            f"[DEBUG] PARES desde Granger: {len(cand)} "
            f"candidatos (bh_rechaza=True, D1_feat→target, "
            f"n>={min_obs})"
        )
        if n_descartados_obs:
            print(
                f"[DEBUG][ALERTA] {n_descartados_obs} pares "
                f"sobreviven BH pero con n<{min_obs}: "
                "descartados por datos insuficientes"
            )

    cand = cand.sort_values("bh_q", ascending=True)
    n_total = len(cand)
    if n_total > max_pares:
        cand_usados = cand.head(max_pares).copy()
        n_afuera = n_total - max_pares
        print(
            f"[DEBUG][ALERTA] {n_total} pares sobreviven BH por "
            f"encima de MAX_PARES_WF={max_pares}: se usan los "
            f"{max_pares} de menor bh_q, quedan {n_afuera} "
            "afuera."
        )
    else:
        cand_usados = cand
        n_afuera = 0

    pares: list[dict[str, Any]] = [
        {
            "feat": str(fila["serie_feat"]),
            "target": str(fila["serie_target"]),
            "lag": int(fila["lag"]),
            "pista": str(fila["pista"]),
            "n": int(fila["n"]) if pd.notna(fila["n"]) else None,  # B2-1
        }
        for _, fila in cand_usados.iterrows()
    ]

    # CAMBIO E2E-11-01: emitir CSV con encabezados incluso con 0 pares.
    if SAVE_DATA:
        os.makedirs(DIR_REPORTES, exist_ok=True)
        _cols_guardar = [
            c for c in (
                "pista", "serie_feat", "serie_target",
                "direccion", "lag", "n", "bh_q",
            )
            if c in cand_usados.columns
        ]
        _df_guardar = cand_usados[_cols_guardar].copy()
        _df_guardar["criterio"] = (
            f"bh_rechaza=True, direccion=D1_feat→target, "
            f"n>={min_obs}, top {max_pares} por bh_q ascendente "
            f"({n_afuera} descartados por exceder MAX_PARES_WF, "
            f"{n_descartados_obs} descartados por n<{min_obs})"
        )
        _df_guardar.to_csv(ARCHIVO_PARES_WF, index=False)
        if DEBUG:
            print(f"[DEBUG] guardado: {ARCHIVO_PARES_WF}")

    return pares



# ════════════════════════════════════════════════════════════
# 5. Funciones — Walk-forward
# ════════════════════════════════════════════════════════════


def ar_un_paso(
    y_train: pd.Series, y_test: pd.Series, p: int,
) -> pd.Series:
    """
    Pronóstico AR(p) a un paso con el valor REAL del período
    anterior (no el pronosticado) — CAMBIO C2-1. En el walk-
    forward de este Job, y_test tiene un solo punto, así que es
    matemáticamente equivalente al predict() dinámico de antes
    (no hay pasos intermedios para encadenar); se unifica con el
    mismo helper que el resto de los Jobs de todos modos.
    """
    res = AutoReg(y_train.to_numpy(), lags=p, trend="c").fit()
    c, phi = res.params[0], res.params[1:]
    y_all = np.r_[y_train.to_numpy(), y_test.to_numpy()]
    pred = [
        c + sum(phi[k] * y_all[t - 1 - k] for k in range(p))
        for t in range(len(y_train), len(y_all))
    ]
    return pd.Series(pred, index=y_test.index, name=f"ar{p}")


def _fit_predict_ar(
    y_train: pd.Series, y_test: pd.Series, p: int,
) -> tuple[float, bool]:
    """
    Ajusta AR(p) sobre y_train y devuelve la predicción a un
    paso (ar_un_paso) para el siguiente punto real (y_test).

    CAMBIO C1-1: sin old_names=False.
    CAMBIO C1-3/C1-d: sin fallback silencioso — si falla, se
    registra con [DEBUG][ALERTA] y se devuelve (media, False).
    walkforward_par cuenta las fallas y las reporta al final.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = ar_un_paso(y_train, y_test, p)
        return float(pred.iloc[0]), True
    except Exception as exc:
        if DEBUG:
            print(f"[DEBUG][ALERTA] AR({p}) falló: {exc}")
        return float(y_train.mean()), False


def _fit_predict_ridge(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    x_new: np.ndarray,
) -> float:
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(Xtr, ytr)
    return float(model.predict(x_new.reshape(1, -1))[0])


def _fit_predict_rf(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    x_new: np.ndarray,
) -> float:
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(Xtr, ytr)
    return float(model.predict(x_new.reshape(1, -1))[0])


def _fit_predict_xgb(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    x_new: np.ndarray,
) -> float:
    if _HAS_XGB:
        model = XGBRegressor(**XGB_PARAMS)
    else:
        model = GradientBoostingRegressor(
            n_estimators=XGB_PARAMS["n_estimators"],
            max_depth=XGB_PARAMS["max_depth"],
            learning_rate=XGB_PARAMS["learning_rate"],
            random_state=XGB_PARAMS["random_state"],
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xtr, ytr)
    return float(model.predict(x_new.reshape(1, -1))[0])


def walkforward_par(
    X: pd.DataFrame,
    y: pd.Series,
    par_tag: str,
    pista: str,
    feat_col: str = "",
) -> tuple[pd.DataFrame, dict]:
    """
    Walk-forward expandente para un par.

    Entrena con los primeros K obs, predice K+1, avanza 1 paso.
    Ventana inicial: 60% de la serie.

    Args:
        feat_col: Nombre de la feature, solo para registrar su
            frecuencia nativa en las métricas (CAMBIO C4-7).

    Returns:
        (df_predicciones, dict_metricas)
    """
    n = len(X)
    n_init = max(20, int(n * RATIO_TRAIN_INICIAL))

    if n_init >= n - 1:
        raise ValueError(
            f"Serie demasiado corta: n={n}, init={n_init}"
        )

    n_wf = n - n_init
    if DEBUG:
        print(
            f"[DEBUG] WF {par_tag}: n={n}, init={n_init}, "
            f"pasos={n_wf}"
        )

    fechas: list = []
    reales: list[float] = []
    pred_ar: list[float] = []
    pred_ridge: list[float] = []
    pred_rf: list[float] = []
    pred_xgb: list[float] = []
    n_fallas_ar = 0  # CAMBIO C1-d

    X_arr = X.values
    y_arr = y.values
    y_series = y.copy()

    scaler = StandardScaler()

    for t in range(n_init, n):
        Xtr_raw = X_arr[:t]
        ytr = y_arr[:t]
        x_new_raw = X_arr[t]
        y_real = float(y_arr[t])

        scaler.fit(Xtr_raw)
        Xtr = scaler.transform(Xtr_raw)
        x_new = scaler.transform(
            x_new_raw.reshape(1, -1)
        ).flatten()

        fechas.append(y.index[t])
        reales.append(y_real)

        # AR baseline
        _pred_ar_t, _ar_ok_t = _fit_predict_ar(
            y_series.iloc[:t], y_series.iloc[t:t + 1],
            AR_LAGS_CONTROL,
        )
        pred_ar.append(_pred_ar_t)
        if not _ar_ok_t:
            n_fallas_ar += 1  # CAMBIO C1-d

        # Ridge
        try:
            pred_ridge.append(
                _fit_predict_ridge(Xtr, ytr, x_new)
            )
        except Exception:
            pred_ridge.append(float(ytr.mean()))

        # Random Forest
        try:
            pred_rf.append(
                _fit_predict_rf(Xtr, ytr, x_new)
            )
        except Exception:
            pred_rf.append(float(ytr.mean()))

        # XGBoost / GradientBoosting
        try:
            pred_xgb.append(
                _fit_predict_xgb(Xtr, ytr, x_new)
            )
        except Exception:
            pred_xgb.append(float(ytr.mean()))

    xgb_label = "xgboost" if _HAS_XGB else "gradient_boost"

    df_pred = pd.DataFrame({
        "fecha": fechas,
        "par": par_tag,
        "pista": pista,
        "real": reales,
        f"ar{AR_LAGS_CONTROL}": pred_ar,
        "ridge": pred_ridge,
        "random_forest": pred_rf,
        xgb_label: pred_xgb,
    })

    # Métricas acumuladas
    modelos = [
        f"ar{AR_LAGS_CONTROL}", "ridge",
        "random_forest", xgb_label,
    ]
    metricas_dict: list[dict] = []
    y_real_arr = np.array(reales)

    for modelo in modelos:
        y_pred_arr = np.array(df_pred[modelo].tolist())
        mask = ~(np.isnan(y_real_arr) | np.isnan(y_pred_arr))
        yt = y_real_arr[mask]
        yp = y_pred_arr[mask]
        # CAMBIO C4-7 (Job 10): frecuencia nativa del feature,
        # para auditar.
        _frec_nativa = FRECUENCIA_NATIVA.get(feat_col, "D")
        if len(yt) == 0:
            metricas_dict.append({
                "par": par_tag, "pista": pista,
                "modelo": modelo, "n_wf": 0,
                "rmse": None, "mae": None, "r2_oos": None,
                "frec_nativa_feat": _frec_nativa,
            })
            continue
        err = yt - yp
        mse = float(np.mean(err ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        r2 = (
            1.0 - float(np.sum(err ** 2)) / ss_tot
            if ss_tot > 1e-12 else float("nan")
        )
        metricas_dict.append({
            "par": par_tag,
            "pista": pista,
            "modelo": modelo,
            "n_wf": int(len(yt)),
            "rmse": round(float(np.sqrt(mse)), 8),
            "mae": round(float(np.mean(np.abs(err))), 8),
            "r2_oos": round(r2, 4),
            "frec_nativa_feat": _frec_nativa,
        })

    # CAMBIO C1-d: reporta las fallas de AR de este par.
    if DEBUG and n_fallas_ar > 0:
        print(
            f"[DEBUG] WF {par_tag}: AR falló en {n_fallas_ar}/"
            f"{n_wf} pasos (fallback a la media)"
        )

    return df_pred, {
        "metricas": metricas_dict, "n_fallas_ar": n_fallas_ar,
    }


# ════════════════════════════════════════════════════════════
# 6. Funciones — Convergencia de niveles
# ════════════════════════════════════════════════════════════


def cargar_nivel1(ruta_granger: str) -> pd.DataFrame:
    """Carga los mejores p-values de Granger por par."""
    if not os.path.exists(ruta_granger):
        return pd.DataFrame()
    df = pd.read_csv(ruta_granger)
    sig = df[df["bh_rechaza"] == True].copy()
    if sig.empty:
        return pd.DataFrame()
    idx = sig.groupby(
        ["serie_feat", "serie_target", "pista"]
    )["bh_q"].idxmin()
    return sig.loc[idx][
        ["serie_feat", "serie_target", "pista",
         "lag", "pval_f", "bh_q"]
    ].reset_index(drop=True)


def cargar_nivel3(ruta_met_nl: str) -> pd.DataFrame:
    """Carga métricas del notebook 19."""
    if not os.path.exists(ruta_met_nl):
        return pd.DataFrame()
    return pd.read_csv(ruta_met_nl)


def es_verdadero(valor: object) -> bool:
    """
    True solo para un booleano verdadero (Python o numpy); False
    para NO_EVALUABLE, NaN o None. CAMBIO B3-4 (Bloque 3).
    """
    return (
        not isinstance(valor, str) and valor is not None
        and bool(pd.notna(valor)) and bool(valor)
    )


def evaluar_mejora_material(
    mejora_pct: float | None, n_test: int | None, etiqueta: str,
) -> bool | str:
    """
    Mejora material (≥UMBRAL_MEJORA % RMSE vs AR) solo si el test
    tiene al menos MIN_N_TEST observaciones; si no, devuelve
    NO_EVALUABLE y lo registra con [DEBUG][ALERTA].

    CAMBIO B3-4 (Bloque 3).

    Args:
        mejora_pct: Mejora % de RMSE vs AR (o None).
        n_test: Observaciones del test (o None).
        etiqueta: Nivel y par, para el log.

    Returns:
        True/False, o NO_EVALUABLE.
    """
    if n_test is None or pd.isna(n_test) or n_test < MIN_N_TEST:
        print(
            f"[DEBUG][ALERTA] {etiqueta}: n_test={n_test} < "
            f"MIN_N_TEST={MIN_N_TEST} → mejora material no evaluable"
        )
        return NO_EVALUABLE
    return bool(mejora_pct is not None and mejora_pct >= UMBRAL_MEJORA)


def evaluar_nivel2_lineal(
    df: pd.DataFrame,
    pares_config: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Nivel 2 (CAMBIO I6-2): modelo lineal (Ridge) por par, para
    poder comparar su mejora de RMSE contra el AR con el mismo
    criterio que los niveles 3 y 4.

    No existe hoy una fuente de Ridge por-par en el pipeline: el
    Ridge de Job 8 se ajusta con TODAS las features juntas (un
    modelo conjunto por pista/target, no uno por par), así que
    no es comparable 1 a 1 con Granger/no-lineal/walk-forward
    para este par específico. Por eso se calcula acá, con un
    único split train/test simple (no walk-forward, para no
    duplicar el costo de nivel 4) sobre el mismo X/y que usan
    los demás niveles (construir_xy_par).

    Args:
        df: Dataset diario completo.
        pares_config: Lista de pares (feat, target, lag, pista).

    Returns:
        DataFrame con par, pista, n2_rmse, n2_rmse_ar,
        n2_mejora_pct, n2_mejora_material.
    """
    filas: list[dict[str, Any]] = []
    for par in pares_config:
        feat, target = par["feat"], par["target"]
        pista, lag = par["pista"], par["lag"]
        par_tag = f"{feat}→{target}"
        fila: dict[str, Any] = {"par": par_tag, "pista": pista}
        try:
            X, y = construir_xy_par(df, feat, target, lag, pista)
            if len(X) < 30:
                fila.update({
                    "n2_rmse": None, "n2_rmse_ar": None,
                    "n2_mejora_pct": None,
                    "n2_mejora_material": False,
                })
                filas.append(fila)
                continue

            n_test = max(5, int(len(X) * 0.2))
            Xtr, Xts = X.iloc[:-n_test], X.iloc[-n_test:]
            ytr, yts = y.iloc[:-n_test], y.iloc[-n_test:]

            scaler = StandardScaler()
            Xtr_sc = scaler.fit_transform(Xtr)
            Xts_sc = scaler.transform(Xts)
            modelo = Ridge(alpha=RIDGE_ALPHA)
            modelo.fit(Xtr_sc, ytr)
            pred_ridge = modelo.predict(Xts_sc)

            pred_ar_s = ar_un_paso(ytr, yts, AR_LAGS_CONTROL)

            yt = yts.to_numpy(dtype=float)
            rmse_ridge = float(np.sqrt(np.mean(
                (yt - pred_ridge) ** 2
            )))
            rmse_ar = float(np.sqrt(np.mean(
                (yt - pred_ar_s.to_numpy(dtype=float)) ** 2
            )))
            mejora_pct = (
                round(100.0 * (rmse_ar - rmse_ridge) / rmse_ar, 2)
                if rmse_ar > 0 else None
            )
            fila.update({
                "n2_rmse": round(rmse_ridge, 8),
                "n2_rmse_ar": round(rmse_ar, 8),
                "n2_mejora_pct": mejora_pct,
                "n2_n_test": int(n_test),  # CAMBIO B3-4
                "n2_mejora_material": evaluar_mejora_material(
                    mejora_pct, n_test, f"nivel2 {par_tag}",
                ),
            })
        except Exception as exc:
            if DEBUG:
                print(f"[DEBUG] nivel2 {par_tag}: {exc}")
            fila.update({
                "n2_rmse": None, "n2_rmse_ar": None,
                "n2_mejora_pct": None,
                "n2_mejora_material": False,
            })
        filas.append(fila)
    return pd.DataFrame(filas)


def tabla_convergencia(
    pares_config: list[dict[str, Any]],
    df_nivel1: pd.DataFrame,
    df_nivel2: pd.DataFrame,
    df_nivel3: pd.DataFrame,
    df_wf_met: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tabla de convergencia de los 4 niveles por par.

    CAMBIO I6-1/I6-2: converge_4_niveles ahora exige los 4
    niveles (antes solo exigía Granger + que el walk-forward
    superase al AR en CUALQUIER margen, sin nivel 2 ni umbral en
    nivel 3). Se conserva la definición vieja como
    "converge_laxo" para comparar (CAMBIO I6-3).
    """
    filas: list[dict] = []

    for par in pares_config:
        feat = par["feat"]
        target = par["target"]
        pista = par["pista"]
        lag = par["lag"]
        par_tag = f"{feat}→{target}"

        fila: dict[str, Any] = {
            "par": par_tag,
            "pista": pista,
            "lag": lag,
        }

        # Nivel 2: modelo lineal (Ridge) — CAMBIO I6-2.
        if not df_nivel2.empty:
            n2 = df_nivel2[
                (df_nivel2["par"] == par_tag)
                & (df_nivel2["pista"] == pista)
            ]
            if not n2.empty:
                fila["n2_rmse"] = n2.iloc[0].get("n2_rmse")
                fila["n2_mejora_pct"] = n2.iloc[0].get(
                    "n2_mejora_pct"
                )
                # CAMBIO B3-4: conserva NO_EVALUABLE (no forzar bool).
                fila["n2_mejora_material"] = n2.iloc[0].get(
                    "n2_mejora_material", False
                )
                fila["n2_n_test"] = n2.iloc[0].get("n2_n_test")
            else:
                fila["n2_rmse"] = None
                fila["n2_mejora_pct"] = None
                fila["n2_mejora_material"] = False
        else:
            fila["n2_rmse"] = None
            fila["n2_mejora_pct"] = None
            fila["n2_mejora_material"] = False

        # Nivel 1: Granger
        if not df_nivel1.empty:
            n1 = df_nivel1[
                (df_nivel1["serie_feat"] == feat)
                & (df_nivel1["serie_target"] == target)
                & (df_nivel1["pista"] == pista)
            ]
            if not n1.empty:
                fila["n1_bh_q"] = float(n1.iloc[0]["bh_q"])
                fila["n1_pval_f"] = float(n1.iloc[0]["pval_f"])
                fila["n1_sobrevive"] = True
            else:
                fila["n1_bh_q"] = None
                fila["n1_pval_f"] = None
                fila["n1_sobrevive"] = False
        else:
            fila["n1_bh_q"] = None
            fila["n1_pval_f"] = None
            fila["n1_sobrevive"] = False

        # Nivel 3: modelo no lineal
        if not df_nivel3.empty:
            n3 = df_nivel3[
                (df_nivel3["par"] == par_tag)
                & (df_nivel3["pista"] == pista)
            ]
            if not n3.empty:
                nl_models = n3[
                    n3["modelo"].isin([
                        "random_forest", "xgboost",
                        "gradient_boost",
                    ])
                ].dropna(subset=["r2_oos"])
                ar_models = n3[
                    n3["modelo"].str.startswith("ar")
                ].dropna(subset=["rmse"])

                # CAMBIO B3-4: n del test de Job 10 (nivel 3).
                fila["n3_n_test"] = (
                    int(nl_models["n_test"].min())
                    if not nl_models.empty
                    and "n_test" in nl_models.columns
                    and nl_models["n_test"].notna().any()
                    else None
                )
                if not nl_models.empty:
                    fila["n3_best_r2"] = float(
                        nl_models["r2_oos"].max()
                    )
                    fila["n3_best_modelo"] = str(
                        nl_models.loc[
                            nl_models["r2_oos"].idxmax(),
                            "modelo",
                        ]
                    )
                else:
                    fila["n3_best_r2"] = None
                    fila["n3_best_modelo"] = "n/a"

                if (
                    not ar_models.empty
                    and not nl_models.empty
                ):
                    rmse_ar = float(ar_models["rmse"].min())
                    rmse_nl = float(nl_models["rmse"].min())
                    if rmse_ar > 0:
                        fila["n3_mejora_pct"] = round(
                            100.0 * (rmse_ar - rmse_nl)
                            / rmse_ar, 2,
                        )
                    else:
                        fila["n3_mejora_pct"] = None
                else:
                    fila["n3_mejora_pct"] = None
            else:
                fila["n3_best_r2"] = None
                fila["n3_best_modelo"] = "n/a"
                fila["n3_mejora_pct"] = None
        else:
            fila["n3_best_r2"] = None
            fila["n3_best_modelo"] = "n/a"
            fila["n3_mejora_pct"] = None

        # CAMBIO I6-1: bandera de umbral para nivel 3 (antes solo
        # se guardaba el %, sin exigir UMBRAL_MEJORA).
        # CAMBIO B3-4: además exige n_test >= MIN_N_TEST.
        fila["n3_mejora_material"] = (
            evaluar_mejora_material(
                fila["n3_mejora_pct"], fila.get("n3_n_test"),
                f"nivel3 {par_tag}",
            )
            if fila.get("n3_mejora_pct") is not None else False
        )

        # Nivel 4: walk-forward
        if not df_wf_met.empty:
            n4 = df_wf_met[
                (df_wf_met["par"] == par_tag)
                & (df_wf_met["pista"] == pista)
            ]
            if not n4.empty:
                feat_models = n4[
                    ~n4["modelo"].str.startswith("ar")
                ].dropna(subset=["rmse"])
                ar_models = n4[
                    n4["modelo"].str.startswith("ar")
                ].dropna(subset=["rmse"])

                if not feat_models.empty:
                    best_idx = feat_models["rmse"].idxmin()
                    fila["n4_best_rmse"] = float(
                        feat_models.loc[best_idx, "rmse"]
                    )
                    fila["n4_best_r2"] = (
                        float(feat_models.loc[best_idx, "r2_oos"])
                        if pd.notna(
                            feat_models.loc[best_idx, "r2_oos"]
                        ) else None
                    )
                    fila["n4_best_modelo"] = str(
                        feat_models.loc[best_idx, "modelo"]
                    )
                    fila["n4_n_wf"] = int(
                        feat_models.loc[best_idx, "n_wf"]
                    )
                else:
                    fila["n4_best_rmse"] = None
                    fila["n4_best_r2"] = None
                    fila["n4_best_modelo"] = "n/a"
                    fila["n4_n_wf"] = 0

                if not ar_models.empty and not feat_models.empty:
                    rmse_ar = float(ar_models["rmse"].min())
                    rmse_best = float(feat_models["rmse"].min())
                    if rmse_ar > 0:
                        mejora = (
                            100.0 * (rmse_ar - rmse_best)
                            / rmse_ar
                        )
                        fila["n4_mejora_pct"] = round(mejora, 2)
                        fila["n4_supera_ar"] = mejora > 0
                        # CAMBIO B3-4: n del WF = n_wf.
                        fila["n4_mejora_material"] = (
                            evaluar_mejora_material(
                                mejora, fila.get("n4_n_wf"),
                                f"nivel4 {par_tag}",
                            )
                        )
                    else:
                        fila["n4_mejora_pct"] = None
                        fila["n4_supera_ar"] = False
                        fila["n4_mejora_material"] = False
                else:
                    fila["n4_mejora_pct"] = None
                    fila["n4_supera_ar"] = False
                    fila["n4_mejora_material"] = False
            else:
                fila["n4_best_rmse"] = None
                fila["n4_best_r2"] = None
                fila["n4_best_modelo"] = "n/a"
                fila["n4_n_wf"] = 0
                fila["n4_mejora_pct"] = None
                fila["n4_supera_ar"] = False
                fila["n4_mejora_material"] = False
        else:
            fila["n4_best_rmse"] = None
            fila["n4_best_r2"] = None
            fila["n4_best_modelo"] = "n/a"
            fila["n4_n_wf"] = 0
            fila["n4_mejora_pct"] = None
            fila["n4_supera_ar"] = False
            fila["n4_mejora_material"] = False

        # CAMBIO I6-4: marca de R² OOS negativo en walk-forward
        # (el modelo predice peor que la media del propio test).
        fila["n4_r2_negativo"] = bool(
            fila.get("n4_best_r2") is not None
            and fila["n4_best_r2"] < 0
        )

        # CAMBIO I6-3: se conserva la definición vieja para
        # comparar ("converge_laxo" — solo exigía Granger + que
        # el walk-forward superara al AR en CUALQUIER margen).
        fila["converge_laxo"] = bool(
            fila.get("n1_sobrevive", False)
            and fila.get("n4_supera_ar", False)
        )

        # CAMBIO I6-1: convergencia real — exige los 4 niveles
        # con el mismo criterio de margen (UMBRAL_MEJORA) en
        # todos, no solo "supera en algo" en walk-forward.
        # CAMBIO B3-4: un nivel "no evaluable" saca al par de la
        # convergencia (no cuenta como converge ni como no converge).
        _niveles = [
            fila.get(k, False) for k in (
                "n2_mejora_material", "n3_mejora_material",
                "n4_mejora_material",
            )
        ]
        fila["convergencia_evaluable"] = not any(
            isinstance(_v, str) and _v == NO_EVALUABLE for _v in _niveles
        )
        fila["converge_4_niveles"] = bool(
            fila["convergencia_evaluable"]
            and fila.get("n1_sobrevive", False)
            and all(es_verdadero(_v) for _v in _niveles)
        )

        filas.append(fila)

    return pd.DataFrame(filas)


# CAMBIO E2E-11-01: PARES se calcula dentro de main(), luego de
# validar las 3 entradas. No se lee Drive al definir el modulo.


# ════════════════════════════════════════════════════════════
# 7. Proceso principal
# ════════════════════════════════════════════════════════════


def main() -> None:
    # CAMBIO B1-4 (Bloque 1): hash de las entradas vs. manifest.
    verificar_entradas_snapshot([ARCHIVO_COMPLETAS], "Job 11")

    print(
        "\n[DEBUG] ═══════════════════════════════════════════"
        "═══════════════"
    )
    print(
        "[DEBUG] 20 — Validación Walk-Forward de Pares "
        "Convergentes"
    )
    print(
        "[DEBUG] ═══════════════════════════════════════════"
        "═══════════════"
    )

    if _HAS_XGB:
        print("[DEBUG] xgboost disponible")
    else:
        print(
            "[DEBUG] xgboost no disponible, "
            "usando GradientBoostingRegressor"
        )

    # ── Cargar y validar entradas ─────────────────────────
    # CAMBIO E2E-11-01: no consumir resultados ausentes o incompletos.
    # Granger y series los genera Job 4; metricas_nolineal Job 10.
    df = cargar_y_validar_entradas_job11()
    PARES: list[dict[str, Any]] = construir_pares_desde_granger(
        ARCHIVO_GRANGER
    )

    # ── Walk-forward por par ────────────────────────────────
    print(
        f"\n[DEBUG] ══ Walk-forward: {len(PARES)} pares ══"
    )

    all_pred: list[pd.DataFrame] = []
    all_met: list[dict] = []
    n_fallas_ar_total = 0  # CAMBIO C1-d

    for par in PARES:
        feat_col = par["feat"]
        target_col = par["target"]
        lag_opt = par["lag"]
        pista = par["pista"]
        par_tag = f"{feat_col}→{target_col}"
        # CAMBIO B2-1: columna real del target (ΔCISS → macro_euro_ciss).
        col_target = COLUMNA_DE_ETIQUETA.get(target_col, target_col)

        if feat_col not in df.columns:
            if DEBUG:
                print(
                    f"[DEBUG] {par_tag}: {feat_col} ausente, "
                    "skip"
                )
            continue
        if col_target not in df.columns:
            if DEBUG:
                print(
                    f"[DEBUG] {par_tag}: {col_target} ausente, "
                    "skip"
                )
            continue

        # CAMBIO C4-7 (Job 10): no correr un par en una pista
        # más fina que la frecuencia nativa de alguna de sus 2
        # series (ej. macro_indpro/macro_umcsent/macro_nfci en
        # pista "A").
        if not frecuencia_admite_pista(feat_col, pista):
            if DEBUG:
                print(
                    f"[DEBUG][ALERTA] {par_tag}: {feat_col} "
                    f"tiene frecuencia nativa "
                    f"{FRECUENCIA_NATIVA.get(feat_col)}, "
                    f"incompatible con pista {pista} — skip"
                )
            continue
        if not frecuencia_admite_pista(col_target, pista):
            if DEBUG:
                print(
                    f"[DEBUG][ALERTA] {par_tag}: {col_target} "
                    f"tiene frecuencia nativa "
                    f"{FRECUENCIA_NATIVA.get(col_target)}, "
                    f"incompatible con pista {pista} — skip"
                )
            continue

        informar_n_vs_job4(  # CAMBIO B2-1
            df, feat_col, target_col, pista, par.get("n"), par_tag,
        )
        try:
            X, y = construir_xy_par(
                df, feat_col, target_col, lag_opt, pista,
            )
        except Exception as exc:
            # CAMBIO E2E-11-01: no publicar resultados parciales.
            raise RuntimeError(
                f"[DEBUG][ERROR] Job 11: error de features "
                f"en {par_tag}, pista={pista}: {exc}"
            ) from exc

        if len(X) < 30:
            if DEBUG:
                print(
                    f"[DEBUG] {par_tag}: n={len(X)} < 30, skip"
                )
            continue

        try:
            df_pred, res = walkforward_par(
                X, y, par_tag, pista, feat_col=feat_col,
            )
            all_pred.append(df_pred)
            all_met.extend(res["metricas"])
            n_fallas_ar_total += res.get("n_fallas_ar", 0)
        except Exception as exc:
            # CAMBIO E2E-11-01: distinguir excepcion de un par no evaluable.
            raise RuntimeError(
                f"[DEBUG][ERROR] Job 11: fallo walk-forward "
                f"para {par_tag}, pista={pista}: {exc}"
            ) from exc

    # CAMBIO C1-d: total de fallas de AR en toda la corrida.
    if DEBUG:
        print(
            f"\n[DEBUG] AR: {n_fallas_ar_total} fallas totales "
            "(fallback a la media) en todos los pares/pasos."
        )

    # ── Consolidar ──────────────────────────────────────────
    df_wf_met = pd.DataFrame(all_met)
    df_wf_pred = (
        pd.concat(all_pred, ignore_index=True)
        if all_pred else pd.DataFrame()
    )

    if df_wf_met.empty:
        print("\n[DEBUG][ALERTA] Job 11: sin resultados de walk-forward.")
        # CAMBIO E2E-11-01: sobrescribir los tres CSV antiguos con 0 filas.
        if SAVE_DATA:
            rutas = guardar_tablas_job11(
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            )
            if DEBUG:
                print(f"[DEBUG] Job 11: {len(rutas)} CSV vacios actualizados")
        # CAMBIO B6-1: la DECISIÓN se movió a Job 15.
        return

    # Agregar mejora vs AR
    df_wf_met["mejora_vs_ar_pct"] = np.nan
    for par_tag in df_wf_met["par"].unique():
        sub = df_wf_met[df_wf_met["par"] == par_tag]
        ar_row = sub[
            sub["modelo"].str.startswith("ar")
        ].dropna(subset=["rmse"])
        if ar_row.empty:
            continue
        rmse_ar = float(ar_row["rmse"].min())
        if rmse_ar <= 0:
            continue
        mask = df_wf_met["par"] == par_tag
        for idx in df_wf_met[mask].index:
            rmse_m = df_wf_met.loc[idx, "rmse"]
            if pd.notna(rmse_m):
                df_wf_met.loc[idx, "mejora_vs_ar_pct"] = round(
                    100.0 * (rmse_ar - rmse_m) / rmse_ar, 2,
                )

    # ── Convergencia de niveles ─────────────────────────────
    print("\n[DEBUG] ══ Convergencia de niveles ══")

    df_nivel1 = cargar_nivel1(ARCHIVO_GRANGER)
    # CAMBIO I6-2: nivel 2 (modelo lineal) se calcula acá mismo.
    df_nivel2 = evaluar_nivel2_lineal(df, PARES)
    df_nivel3 = cargar_nivel3(ARCHIVO_MET_NL)

    df_conv = tabla_convergencia(
        PARES, df_nivel1, df_nivel2, df_nivel3, df_wf_met,
    )

    # ── Reporte ─────────────────────────────────────────────
    print("\n" + "═" * 64)
    print(
        "REPORTE FINAL — VALIDACIÓN WALK-FORWARD"
    )
    print("═" * 64)

    print("\n[DEBUG] Métricas walk-forward por par:")
    _cols_met = [
        "par", "pista", "modelo", "n_wf",
        "rmse", "r2_oos", "mejora_vs_ar_pct",
    ]
    _cols_met = [
        c for c in _cols_met if c in df_wf_met.columns
    ]
    print(df_wf_met[_cols_met].to_string(index=False))

    # Pares con mejora material en WF
    feat_wf = df_wf_met[
        ~df_wf_met["modelo"].str.startswith("ar")
    ].dropna(subset=["mejora_vs_ar_pct"])
    # CAMBIO B3-4: solo modelos con n_wf >= MIN_N_TEST.
    mejora_material = feat_wf[
        (feat_wf["mejora_vs_ar_pct"] >= UMBRAL_MEJORA)
        & (feat_wf["n_wf"] >= MIN_N_TEST)
    ]

    print(f"\n{'─' * 64}")
    if not mejora_material.empty:
        print(
            f"PARES CON MEJORA MATERIAL EN WALK-FORWARD "
            f"(>{UMBRAL_MEJORA}% RMSE vs AR):"
        )
        print("─" * 64)
        print(mejora_material[_cols_met].to_string(index=False))
    else:
        print(
            "Ningún modelo supera al AR en más de "
            f"{UMBRAL_MEJORA}% en walk-forward."
        )

    # Tabla de convergencia
    print(f"\n{'═' * 64}")
    print("CONVERGENCIA DE LOS 4 NIVELES DE EVIDENCIA")
    print("═" * 64)

    _cols_conv = [
        "par", "pista", "lag",
        "n1_bh_q", "n1_sobrevive",
        "n2_mejora_pct", "n2_mejora_material",
        "n3_best_r2", "n3_mejora_pct", "n3_mejora_material",
        "n4_best_r2", "n4_r2_negativo",
        "n4_mejora_pct", "n4_supera_ar", "n4_mejora_material",
        "converge_4_niveles", "converge_laxo",
        "convergencia_evaluable",
    ]
    _cols_conv = [
        c for c in _cols_conv if c in df_conv.columns
    ]
    print(df_conv[_cols_conv].to_string(index=False))

    # CAMBIO I6-1/I6-3: se muestran las dos definiciones para
    # poder comparar — converge_4_niveles (nueva, exige los 4
    # niveles con UMBRAL_MEJORA) y converge_laxo (vieja, solo
    # Granger + superar al AR en cualquier margen en WF).
    n_converge = int(df_conv["converge_4_niveles"].sum())
    n_converge_laxo = int(df_conv["converge_laxo"].sum())
    # CAMBIO B3-4: pares fuera de la convergencia por n_test chico.
    _evaluable = df_conv["convergencia_evaluable"].astype(bool)
    n_no_eval = int((~_evaluable).sum())
    if n_no_eval:
        print(
            f"\n[DEBUG][ALERTA] {n_no_eval}/{len(PARES)} par(es) con "
            f"algún nivel de n_test < MIN_N_TEST={MIN_N_TEST}: "
            "convergencia no evaluable, quedan fuera del conteo: "
            f"{list(df_conv.loc[~_evaluable, 'par'])}"
        )
    print(
        f"\n{n_converge}/{len(PARES)} pares convergen en los "
        "4 niveles de evidencia (criterio estricto: Granger + "
        f"mejora ≥{UMBRAL_MEJORA}% en los 3 niveles de modelo)."
    )
    print(
        f"{n_converge_laxo}/{len(PARES)} pares convergen bajo "
        "el criterio viejo (converge_laxo: Granger + supera al "
        "AR en walk-forward en cualquier margen, sin nivel 2 ni "
        "umbral en nivel 3)."
    )
    if n_converge_laxo > n_converge:
        # CAMBIO B3-4: separar los que dejan de converger por n chico.
        _laxo_sin_estricto = (
            df_conv["converge_laxo"].astype(bool)
            & ~df_conv["converge_4_niveles"].astype(bool)
        )
        _por_n = int((_laxo_sin_estricto & ~_evaluable).sum())
        _por_mejora = int((_laxo_sin_estricto & _evaluable).sum())
        print(
            f"[DEBUG][ALERTA] {n_converge_laxo - n_converge} "
            "par(es) que convergían bajo el criterio viejo ya "
            f"no convergen bajo el nuevo: {_por_mejora} por mejora "
            "no material o ausente en algún nivel, "
            f"{_por_n} por n_test < {MIN_N_TEST} (no evaluables)."
        )

    # VEREDICTO
    print(f"\n{'═' * 64}")
    print("VEREDICTO FINAL")
    print("═" * 64)

    if n_converge > 0:
        pares_ok = df_conv[df_conv["converge_4_niveles"]]
        print(
            f"{n_converge} par(es) superan los 4 niveles "
            "de validación:"
        )
        for _, row in pares_ok.iterrows():
            print(
                f"  - {row['par']} (Pista {row['pista']}, "
                f"lag {row['lag']}): "
                f"Granger q={row.get('n1_bh_q', 'n/a')}, "
                f"WF mejora={row.get('n4_mejora_pct', 'n/a')}%"
            )
        # CAMBIO B3-1 (Bloque 3): el texto sale de las cifras.
        # Antes afirmaba "la variable macro anticipa al RWA" aunque
        # el feature fuera RWA o el R² del walk-forward fuera < 0.
        _r2_neg = pares_ok[pares_ok["n4_r2_negativo"].astype(bool)]
        print(
            "\nEstos pares muestran precedencia temporal "
            "(el feature anticipa al target) validada por "
            "Granger + FDR global, modelo lineal, modelo no "
            "lineal y walk-forward expandente."
        )
        if not _r2_neg.empty:
            print(
                f"De ellos, {len(_r2_neg)} tienen R² OOS < 0 en "
                f"walk-forward ({', '.join(_r2_neg['par'])}): "
                "superan al AR pero no a la media del test."
            )
        print(
            "\nCAUTELA: precedencia temporal ≠ causalidad "
            "económica. Tendencias comunes (política monetaria "
            "global), adopción incipiente de RWA y ffill de "
            "series mensuales pueden inflar la señal. Los "
            "resultados son hallazgos de línea de base, no "
            "predicciones operativas."
        )
    else:
        print(
            f"Ningún par evaluable ({len(PARES) - n_no_eval} de "
            f"{len(PARES)}) converge en los 4 niveles."
        )

    # Pares que no convergen (solo evaluables), con el nivel que falla
    no_conv = df_conv[~df_conv["converge_4_niveles"] & _evaluable]
    if not no_conv.empty:
        print(f"\n{len(no_conv)} par(es) evaluables no convergen:")
        for _, row in no_conv.iterrows():
            _fallan = [
                _nombre for _nombre, _col in (
                    ("Granger", "n1_sobrevive"),
                    ("lineal", "n2_mejora_material"),
                    ("no lineal", "n3_mejora_material"),
                    ("walk-forward", "n4_mejora_material"),
                ) if not es_verdadero(row.get(_col))
            ]
            mejora = row.get("n4_mejora_pct")
            mejora_str = (
                f"{mejora:.2f}%" if pd.notna(mejora) else "n/a"
            )
            print(
                f"  - {row['par']} (Pista {row['pista']}): falla "
                f"{', '.join(_fallan)}; WF mejora={mejora_str}"
            )

    # CAMBIO B6-1: la DECISIÓN se movió a Job 15.

    print(f"\n{'═' * 64}")
    print(
        "RWA de adopción incipiente: log-dif dominada por "
        "crecimiento estructural, no por respuesta a estrés."
    )
    print(
        "No se afirma causalidad económica; "  # CAMBIO B3-1
        + (
            f"{n_converge} par(es) con precedencia estadística "
            "validada en 4 niveles."
            if n_converge > 0
            else "ningún par queda validado en los 4 niveles."
        )
    )
    print("═" * 64)

    # ── Guardar ─────────────────────────────────────────────
    if SAVE_DATA:
        os.makedirs(DIR_REPORTES, exist_ok=True)

        # CAMBIO E2E-11-01: CSV actuales, incluso si no hay filas.
        guardados: list[str] = guardar_tablas_job11(
            df_wf_met, df_wf_pred, df_conv,
        )

        if DEBUG:
            print("\n[DEBUG] Archivos guardados:")
            for g in guardados:
                print(f"  {g}")
    elif DEBUG:
        print("\n[DEBUG] SAVE_DATA=False — nada guardado")


if __name__ == "__main__":
    main()

