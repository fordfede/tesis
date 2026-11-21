"""# Job 10 - Modelos No Lineales sobre Pares Validados por Granger"""

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

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
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

# CAMBIO E2E-10-01: ruta configurable, misma ruta por defecto.
BASE_DIR: str = os.environ.get(
    "TESIS_BASE_DIR", "/content/drive/MyDrive/Tesis/expansion"
)
DIR_PROCESSED: str = os.path.join(BASE_DIR, "data", "processed")
DIR_REPORTES: str = os.path.join(
    BASE_DIR, "outputs", "reportes"
)
DIR_MODELOS: str = os.path.join(BASE_DIR, "outputs", "modelos")

ARCHIVO_COMPLETAS: str = os.path.join(
    DIR_PROCESSED, "series_completas.csv"
)
ARCHIVO_GRANGER: str = os.path.join(
    DIR_REPORTES, "granger_completo.csv"
)
ARCHIVO_MET: str = os.path.join(
    DIR_REPORTES, "metricas_nolineal.csv"
)
ARCHIVO_FI: str = os.path.join(
    DIR_REPORTES, "feature_importance_rf.csv"
)

# ── Transformaciones (mismas que notebook 18) ───────────────
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

# CAMBIO C4-5 (Job 9): frecuencia nativa de cada columna macro
# (ver nota completa en Job 8). Los pares que llegan acá ya
# vienen pre-filtrados por Granger (Job 8/4), pero se valida de
# nuevo por si un par de pista incompatible se coló.
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

    CAMBIO C4-5 (Job 9). Columnas no RWA/onchain fuera de
    FRECUENCIA_NATIVA se consideran diarias (siempre califican).
    """
    nativa = FRECUENCIA_NATIVA.get(col)
    if nativa is None:
        return True
    freq_pista = _PISTA_A_FRECUENCIA.get(pista, "D")
    return _RANGO_FRECUENCIA[nativa] <= _RANGO_FRECUENCIA[freq_pista]


# ── Modelo ──────────────────────────────────────────────────
N_CV_SPLITS: int = 5
# CAMBIO H4 (documentación, sin cambio de comportamiento):
# acá se usa el mismo orden AR para pista diaria y semanal.
# Jobs 5/7/8 usan AR_P_A=3 (diaria) / AR_P_B=2 (semanal) en vez
# de un valor único. Son decisiones tomadas en fases distintas
# del proyecto y no se unificaron — el usuario confirmó dejarlo
# así y documentarlo en vez de cambiar el comportamiento.
AR_LAGS_CONTROL: int = 2
RATIO_TEST: float = 0.2
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

# CAMBIO C3-2: LASSO_ALPHAS ya no se usa (LassoCV con
# alphas=None calcula su propio camino) — ver entrenar_lasso.


# CAMBIO E2E-10-01: esquemas de salida estables para resultados vacios.
# Las columnas replican las emitidas por metricas() y feature_importance_rf().
COLUMNAS_METRICAS_NL: list[str] = [
    "par", "pista", "direccion", "modelo", "n_train", "n_test",
    "n_features", "frec_nativa_feat", "baseline_ok", "alpha_en_borde",
    "rmse", "mae", "r2_oos", "mejora_vs_ar_pct",
    "mejora_material",  # CAMBIO B3-4
]
COLUMNAS_IMPORTANCIA_RF: list[str] = [
    "par", "pista", "direccion", "feature", "importance",
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
# 4. Funciones — Carga y selección de pares
# ════════════════════════════════════════════════════════════


def validar_series_job10(df: pd.DataFrame, ruta: str) -> None:
    """Validacion estructural de series_completas sin modificar sus valores."""
    if df.empty or not len(df.columns):
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 10: serie vacia: {ruta} "
            "(productor: Job 4)."
        )
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 10: fechas invalidas en {ruta} "
            "(productor: Job 4)."
        )
    if df.index.has_duplicates:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 10: fechas duplicadas en {ruta} "
            "(productor: Job 4)."
        )


def validar_granger_job10(df: pd.DataFrame, ruta: str) -> None:
    """Chequea contrato del Job 4; admite cero pruebas con encabezados.

    No cambia la familia FDR ni el filtro de pares. Solo impide que
    CSV faltantes o mal formados se lean como 'cero significativos'.
    """
    obligatorias = [
        "serie_feat", "serie_target", "direccion", "pista", "lag",
        "bh_rechaza", "bh_q",
    ]
    faltantes = [c for c in obligatorias if c not in df.columns]
    if faltantes:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 10: faltan columnas {faltantes} "
            f"en {ruta} (productor: Job 4)."
        )
    if df.empty:
        return  # Granger sin tests evaluables: resultado valido con 0 filas.

    # read_csv infiere bool para las columnas True/False normales del Job 4.
    if df["bh_rechaza"].isna().any() or not df["bh_rechaza"].map(
        lambda v: isinstance(v, (bool, np.bool_))
    ).all():
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 10: bh_rechaza invalido en {ruta}. "
            "Se requieren valores booleanos True/False."
        )

    sig = df.loc[df["bh_rechaza"] == True]
    if sig.empty:
        return
    if sig[["serie_feat", "serie_target", "direccion", "pista"]].isna().any().any():
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 10: par significativo sin identificadores "
            f"en {ruta}."
        )
    lags = pd.to_numeric(sig["lag"], errors="coerce")
    q = pd.to_numeric(sig["bh_q"], errors="coerce")
    if (
        lags.isna().any() or not np.isfinite(lags).all()
        or (lags < 1).any() or (lags != np.floor(lags)).any()
        or q.isna().any() or not np.isfinite(q).all()
    ):
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 10: lag o bh_q invalido en "
            f"pares significativos de {ruta}."
        )


def guardar_tablas_job10(
    df_metricas: pd.DataFrame, df_fi: pd.DataFrame,
) -> list[str]:
    """Actualiza ambos CSV incluso si hay cero resultados.

    No elimina modelos .joblib previos ni modifica valores presentes.
    """
    os.makedirs(DIR_REPORTES, exist_ok=True)
    for tabla, columnas, ruta in (
        (df_metricas, COLUMNAS_METRICAS_NL, ARCHIVO_MET),
        (df_fi, COLUMNAS_IMPORTANCIA_RF, ARCHIVO_FI),
    ):
        actual = tabla if not tabla.empty else pd.DataFrame(columns=columnas)
        actual.to_csv(ruta, index=False)
    return [ARCHIVO_MET, ARCHIVO_FI]


def cargar_series(ruta: str) -> pd.DataFrame:
    df = pd.read_csv(ruta, parse_dates=["fecha"])
    df = df.set_index("fecha").sort_index()
    if DEBUG:
        print(f"[DEBUG] {os.path.basename(ruta)}: {df.shape}")
    return df


def cargar_pares_validados(ruta: str) -> pd.DataFrame:
    """
    Carga granger_completo.csv, filtra bh_rechaza == True,
    y se queda con el lag de menor bh_q por par.
    """
    df = pd.read_csv(ruta)
    validar_granger_job10(df, ruta)  # CAMBIO E2E-10-01
    if DEBUG:
        print(
            f"[DEBUG] granger_completo: {len(df)} filas, "
            f"{df['bh_rechaza'].sum()} sobreviven FDR"
        )

    sig = df[df["bh_rechaza"] == True].copy()
    if sig.empty:
        return pd.DataFrame()

    idx = sig.groupby(
        ["serie_feat", "serie_target", "direccion", "pista"]
    )["bh_q"].idxmin()
    pares = sig.loc[idx].reset_index(drop=True)

    if DEBUG:
        print(f"[DEBUG] {len(pares)} pares validados:")
        _cols = [
            "serie_feat", "serie_target", "direccion",
            "pista", "lag", "pval_f", "bh_q",
        ]
        _cols = [c for c in _cols if c in pares.columns]
        print(pares[_cols].to_string(index=False))

    return pares


# ════════════════════════════════════════════════════════════
# 5. Funciones — Transformación y features
# ════════════════════════════════════════════════════════════


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


def construir_features_par(
    df: pd.DataFrame,
    feat_col: str,
    target_col: str,
    lag_optimo: int,
    pista: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construye X e y para un par validado.

    X = [feature_lag1 .. feature_lag{lag_optimo},
         target_lag1, target_lag2] (control AR).
    y = target en t.

    Pista B: resample a W.last() antes de transformar.
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


# ════════════════════════════════════════════════════════════
# 6. Funciones — Modelos
# ════════════════════════════════════════════════════════════


def split_temporal(
    X: pd.DataFrame,
    y: pd.Series,
    ratio_test: float = RATIO_TEST,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    n = len(X)
    n_test = max(2, int(n * ratio_test))
    n_train = n - n_test
    if n_train < 10:
        raise ValueError(f"Train demasiado pequeño: {n_train}")
    if DEBUG:
        print(
            f"[DEBUG] split: n={n}, train={n_train}, "
            f"test={n_test}, features={X.shape[1]}"
        )
    return (
        X.iloc[:n_train], X.iloc[n_train:],
        y.iloc[:n_train], y.iloc[n_train:],
    )


def metricas(
    y_true: pd.Series,
    y_pred: np.ndarray,
    nombre: str,
    par_tag: str,
    pista: str,
    direccion: str,
    n_train: int,
    n_features: int,
    feat_col: str = "",
    baseline_ok: bool = True,
    alpha_en_borde: bool = False,
) -> dict:
    yt = y_true.to_numpy(dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[mask], yp[mask]
    base = {
        "par": par_tag,
        "pista": pista,
        "direccion": direccion,
        "modelo": nombre,
        "n_train": n_train,
        "n_test": len(yt),
        "n_features": n_features,
        # CAMBIO C4-6 (Job 9): frecuencia nativa del feature,
        # para auditar.
        "frec_nativa_feat": FRECUENCIA_NATIVA.get(feat_col, "D"),
        # CAMBIO C1-3/C3-3 (Job 9).
        "baseline_ok": baseline_ok,
        "alpha_en_borde": alpha_en_borde,
    }
    if len(yt) == 0:
        return {**base, "rmse": None, "mae": None, "r2_oos": None}
    err = yt - yp
    mse = float(np.mean(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = (
        1.0 - float(np.sum(err ** 2)) / ss_tot
        if ss_tot > 1e-12 else float("nan")
    )
    return {
        **base,
        "rmse": round(float(np.sqrt(mse)), 8),
        "mae": round(float(np.mean(np.abs(err))), 8),
        "r2_oos": round(r2, 4),
    }


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


def modelo_ar_baseline(
    y_train: pd.Series,
    y_test: pd.Series,
    p: int,
    etiqueta: str,
) -> tuple[pd.Series, bool]:
    """
    CAMBIO C1-1: sin old_names=False. CAMBIO C2-1: pronóstico a
    un paso (ar_un_paso), no dinámico. CAMBIO C1-3: sin fallback
    silencioso — si falla, [DEBUG][ALERTA], media como
    "ar_media" (nunca "ar{p}"), baseline_ok=False.

    Returns:
        (predicciones, baseline_ok).
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = ar_un_paso(y_train, y_test, p)
        return pred, True
    except Exception as exc:
        print(f"[DEBUG][ALERTA] AR({p}) {etiqueta} falló: {exc}")
        media = float(y_train.mean())
        return (
            pd.Series(
                [media] * len(y_test),
                index=y_test.index, name="ar_media",
            ),
            False,
        )


def _cv_splits(n_train: int) -> int:
    return max(2, min(N_CV_SPLITS, n_train // 15))


def entrenar_rf(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xts: np.ndarray,
    etiqueta: str,
) -> tuple[np.ndarray, Any]:
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(Xtr, ytr)
    pred = model.predict(Xts)
    if DEBUG:
        print(f"[DEBUG] RF {etiqueta}: OK")
    return pred, model


def entrenar_xgb(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xts: np.ndarray,
    etiqueta: str,
) -> tuple[np.ndarray, Any]:
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
    pred = model.predict(Xts)
    tag = "XGB" if _HAS_XGB else "GBR"
    if DEBUG:
        print(f"[DEBUG] {tag} {etiqueta}: OK")
    return pred, model


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


def entrenar_lasso(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xts: np.ndarray,
    n_train: int,
    etiqueta: str,
) -> tuple[np.ndarray, Any, bool]:
    """
    CAMBIO C3-2/C3-3/C3-4: alphas=None (n_alphas=100, eps=1e-3,
    max_iter=10000; reemplaza LASSO_ALPHAS fijo), aviso de
    borde, std(y).
    """
    cv = TimeSeriesSplit(n_splits=_cv_splits(n_train))
    if DEBUG:
        print(f"[DEBUG] {etiqueta}: std(y_train)={float(np.std(ytr)):.6g}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LassoCV(
            alphas=None, n_alphas=100, eps=1e-3,
            max_iter=10000, cv=cv,
        )
        model.fit(Xtr, ytr)
    alpha_borde = _chequear_alpha_borde(
        model.alpha_, model.alphas_, f"Lasso {etiqueta}",
    )
    pred = model.predict(Xts)
    if DEBUG:
        n_nz = int(np.sum(model.coef_ != 0))
        print(
            f"[DEBUG] Lasso {etiqueta}: "
            f"alpha={model.alpha_:.4g}, "
            f"coef_nz={n_nz}/{Xtr.shape[1]}"
        )
    return pred, model, alpha_borde


def feature_importance_rf(
    model: RandomForestRegressor,
    columnas: list[str],
    par_tag: str,
    pista: str,
    direccion: str,
) -> pd.DataFrame:
    imp = model.feature_importances_
    filas = [
        {
            "par": par_tag,
            "pista": pista,
            "direccion": direccion,
            "feature": columnas[i],
            "importance": round(float(imp[i]), 6),
        }
        for i in range(len(columnas))
        if imp[i] > 0.01
    ]
    df = pd.DataFrame(filas)
    if not df.empty:
        df = df.sort_values(
            "importance", ascending=False
        ).reset_index(drop=True)
    return df


# ════════════════════════════════════════════════════════════
# 7. Función — Pipeline por par
# ════════════════════════════════════════════════════════════


def correr_par(
    df: pd.DataFrame,
    feat_col: str,
    target_col: str,
    lag_optimo: int,
    pista: str,
    direccion: str,
    n_job4: int | None = None,
) -> dict:
    """
    Pipeline completo para un par validado:
    construir features → split → AR baseline → RF → XGB → Lasso.
    """
    par_tag = f"{feat_col}→{target_col}"
    etiqueta = f"{pista}/{direccion}/{par_tag}"
    # CAMBIO B2-1: columna real del target (ΔCISS → macro_euro_ciss).
    col_target = COLUMNA_DE_ETIQUETA.get(target_col, target_col)

    if feat_col not in df.columns:
        if DEBUG:
            print(f"[DEBUG] {etiqueta}: {feat_col} ausente")
        return {
            "metricas": [], "fi": pd.DataFrame(),
            "modelos": {},
        }
    if col_target not in df.columns:
        if DEBUG:
            print(f"[DEBUG] {etiqueta}: {col_target} ausente")
        return {
            "metricas": [], "fi": pd.DataFrame(),
            "modelos": {},
        }

    # CAMBIO C4-5 (Job 9): no correr un par en una pista más
    # fina que la frecuencia nativa de alguna de sus 2 series
    # (ej. macro_indpro en pista A o B).
    if not frecuencia_admite_pista(feat_col, pista):
        if DEBUG:
            print(
                f"[DEBUG][ALERTA] {etiqueta}: {feat_col} tiene "
                f"frecuencia nativa "
                f"{FRECUENCIA_NATIVA.get(feat_col)}, incompatible "
                f"con pista {pista} — skip"
            )
        return {
            "metricas": [], "fi": pd.DataFrame(),
            "modelos": {},
        }
    if not frecuencia_admite_pista(col_target, pista):
        if DEBUG:
            print(
                f"[DEBUG][ALERTA] {etiqueta}: {col_target} tiene "
                f"frecuencia nativa "
                f"{FRECUENCIA_NATIVA.get(col_target)}, "
                f"incompatible con pista {pista} — skip"
            )
        return {
            "metricas": [], "fi": pd.DataFrame(),
            "modelos": {},
        }

    informar_n_vs_job4(  # CAMBIO B2-1
        df, feat_col, target_col, pista, n_job4, etiqueta,
    )
    try:
        X, y = construir_features_par(
            df, feat_col, target_col, lag_optimo, pista,
        )
    except Exception as exc:
        # CAMBIO E2E-10-01: un error de codigo/datos no es 'no evaluable'.
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 10: fallo generando features "
            f"para {etiqueta}: {exc}"
        ) from exc

    if len(X) < 20:
        if DEBUG:
            print(
                f"[DEBUG] {etiqueta}: n={len(X)} < 20, skip"
            )
        return {
            "metricas": [], "fi": pd.DataFrame(),
            "modelos": {},
        }

    if DEBUG:
        print(
            f"\n[DEBUG] ── Par: {par_tag} | Pista {pista} | "
            f"{direccion} | n={len(X)} ──"
        )

    try:
        Xtr, Xts, ytr, yts = split_temporal(X, y)
    except ValueError as exc:
        if DEBUG:
            print(f"[DEBUG] {etiqueta} split: {exc}")
        return {
            "metricas": [], "fi": pd.DataFrame(),
            "modelos": {},
        }

    columnas = list(X.columns)
    n_train = len(Xtr)
    n_feat = Xtr.shape[1]

    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(Xtr)
    Xts_sc = scaler.transform(Xts)

    all_metricas: list[dict] = []
    all_modelos: dict[str, Any] = {}

    # AR baseline
    ar_p = AR_LAGS_CONTROL
    pred_ar, ar_ok = modelo_ar_baseline(ytr, yts, ar_p, etiqueta)
    all_metricas.append(
        metricas(
            yts, pred_ar.values, pred_ar.name,  # CAMBIO C1-3
            par_tag, pista, direccion, n_train, 0,
            feat_col=feat_col, baseline_ok=ar_ok,
        )
    )

    # Random Forest
    try:
        pred_rf, mdl_rf = entrenar_rf(
            Xtr_sc, ytr.values, Xts_sc, etiqueta,
        )
        all_metricas.append(
            metricas(
                yts, pred_rf, "random_forest",
                par_tag, pista, direccion, n_train, n_feat,
                feat_col=feat_col,
            )
        )
        tag = _par_tag_short(par_tag, pista, direccion)
        all_modelos[f"modelo_rf_{tag}"] = mdl_rf
    except Exception as exc:
        if DEBUG:
            print(f"[DEBUG] RF {etiqueta} falló: {exc}")
        mdl_rf = None

    # XGBoost / GradientBoosting
    try:
        pred_xgb, mdl_xgb = entrenar_xgb(
            Xtr_sc, ytr.values, Xts_sc, etiqueta,
        )
        xgb_name = "xgboost" if _HAS_XGB else "gradient_boost"
        all_metricas.append(
            metricas(
                yts, pred_xgb, xgb_name,
                par_tag, pista, direccion, n_train, n_feat,
                feat_col=feat_col,
            )
        )
        tag = _par_tag_short(par_tag, pista, direccion)
        all_modelos[f"modelo_xgb_{tag}"] = mdl_xgb
    except Exception as exc:
        if DEBUG:
            print(f"[DEBUG] XGB {etiqueta} falló: {exc}")

    # Lasso
    try:
        pred_lasso, mdl_lasso, lasso_borde = entrenar_lasso(
            Xtr_sc, ytr.values, Xts_sc, n_train, etiqueta,
        )
        all_metricas.append(
            metricas(
                yts, pred_lasso, "lasso",
                par_tag, pista, direccion, n_train, n_feat,
                feat_col=feat_col, alpha_en_borde=lasso_borde,
            )
        )
        tag = _par_tag_short(par_tag, pista, direccion)
        all_modelos[f"modelo_lasso_{tag}"] = mdl_lasso
    except Exception as exc:
        if DEBUG:
            print(f"[DEBUG] Lasso {etiqueta} falló: {exc}")

    # Feature importance RF
    df_fi = pd.DataFrame()
    if mdl_rf is not None:
        df_fi = feature_importance_rf(
            mdl_rf, columnas, par_tag, pista, direccion,
        )

    return {
        "metricas": all_metricas,
        "fi": df_fi,
        "modelos": all_modelos,
    }


def _par_tag_short(
    par_tag: str, pista: str, direccion: str,
) -> str:
    clean = (
        par_tag
        .replace("macro_", "")
        .replace("rwa_", "")
        .replace("→", "_")
        .replace("tipo_", "")
        .replace("red_", "")
    )
    d = "d1" if "D1" in direccion else "d2"
    return f"{pista.lower()}_{d}_{clean}"[:60]


# ════════════════════════════════════════════════════════════
# 8. Funciones — Reporte
# ════════════════════════════════════════════════════════════


def es_verdadero(valor: object) -> bool:
    """
    True solo para un booleano verdadero (Python o numpy); False
    para NO_EVALUABLE, NaN o None. CAMBIO B3-4 (Bloque 3).
    """
    return (
        not isinstance(valor, str) and valor is not None
        and bool(pd.notna(valor)) and bool(valor)
    )


def calcular_mejora_vs_ar(df_met: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada par, calcula mejora % RMSE de cada modelo vs AR.
    """
    df = df_met.copy()
    df["mejora_vs_ar_pct"] = np.nan

    for par in df["par"].unique():
        sub = df[df["par"] == par]
        ar_rows = sub[sub["modelo"].str.startswith("ar")]
        if ar_rows.empty or ar_rows["rmse"].isna().all():
            continue
        rmse_ar = float(ar_rows["rmse"].min())
        if rmse_ar <= 0:
            continue
        mask_par = df["par"] == par
        for idx in df[mask_par].index:
            rmse_m = df.loc[idx, "rmse"]
            if pd.notna(rmse_m):
                mejora = 100.0 * (rmse_ar - rmse_m) / rmse_ar
                df.loc[idx, "mejora_vs_ar_pct"] = round(
                    mejora, 2
                )

    # CAMBIO B3-4 (Bloque 3): mejora material solo con
    # n_test >= MIN_N_TEST; si no, "no evaluable" (y se registra).
    df["mejora_material"] = [
        NO_EVALUABLE if pd.isna(_nt) or _nt < MIN_N_TEST
        else bool(pd.notna(_m) and _m >= UMBRAL_MEJORA)
        for _nt, _m in zip(df["n_test"], df["mejora_vs_ar_pct"])
    ]
    _no_eval = df[
        (df["mejora_material"] == NO_EVALUABLE)
        & ~df["modelo"].str.startswith("ar")
    ]
    if not _no_eval.empty:
        print(
            f"[DEBUG][ALERTA] Job 10: {len(_no_eval)} modelo(s) con "
            f"n_test < MIN_N_TEST={MIN_N_TEST}: mejora material "
            f"no evaluable en {sorted(set(_no_eval['par']))}"
        )
    return df


# ════════════════════════════════════════════════════════════
# 9. Proceso principal
# ════════════════════════════════════════════════════════════


def main() -> None:
    # CAMBIO B1-4 (Bloque 1): hash de las entradas vs. manifest.
    verificar_entradas_snapshot([ARCHIVO_COMPLETAS], "Job 10")

    print(
        "\n[DEBUG] ═══════════════════════════════════════════"
        "═══════════════"
    )
    print(
        "[DEBUG] 19 — Modelos No Lineales sobre Pares "
        "Validados"
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

    # ── Cargar datos ────────────────────────────────────────
    # CAMBIO E2E-10-01: los dos insumos los produce el Job 4.
    for ruta in (ARCHIVO_COMPLETAS, ARCHIVO_GRANGER):
        if not os.path.isfile(ruta):
            raise FileNotFoundError(
                f"[DEBUG][ERROR] Job 10: falta {ruta}. "
                "Debe producirlo el Job 4."
            )
    try:
        df = cargar_series(ARCHIVO_COMPLETAS)
    except (OSError, ValueError, KeyError, pd.errors.ParserError,
            pd.errors.EmptyDataError) as exc:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 10: no se puede leer "
            f"{ARCHIVO_COMPLETAS} (Job 4): {exc}"
        ) from exc
    validar_series_job10(df, ARCHIVO_COMPLETAS)
    try:
        pares = cargar_pares_validados(ARCHIVO_GRANGER)
    except (OSError, ValueError, KeyError, pd.errors.ParserError,
            pd.errors.EmptyDataError) as exc:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 10: no se puede leer "
            f"{ARCHIVO_GRANGER} (Job 4): {exc}"
        ) from exc

    if pares.empty:
        print(
            "\n[DEBUG][ALERTA] Job 10: ningun par sobrevivio "
            "FDR global; no hay pares modelables."
        )
        if SAVE_DATA:
            guardados = guardar_tablas_job10(pd.DataFrame(), pd.DataFrame())
            if DEBUG:
                print("[DEBUG] Job 10: CSV sin filas (resultado vacio):")
                for ruta in guardados:
                    print(f"  {ruta}")
        return

    # ── Correr modelos por par ──────────────────────────────
    print(
        f"\n[DEBUG] ══ Modelado no lineal: "
        f"{len(pares)} pares ══"
    )

    resultados: list[dict] = []
    modelos_guardar: dict[str, Any] = {}

    for _, row in pares.iterrows():
        feat_col = str(row["serie_feat"])
        target_col = str(row["serie_target"])
        lag_opt = int(row["lag"])
        pista = str(row["pista"])
        direccion = str(row["direccion"])

        try:
            res = correr_par(
                df, feat_col, target_col, lag_opt,
                pista, direccion,
                n_job4=(  # CAMBIO B2-1
                    int(row["n"])
                    if "n" in row and pd.notna(row["n"]) else None
                ),
            )
            resultados.append(res)
            modelos_guardar.update(res["modelos"])
        except Exception as exc:
            # CAMBIO E2E-10-01: no publicar un analisis parcial
            # como si el par hubiese sido evaluado sin problemas.
            raise RuntimeError(
                f"[DEBUG][ERROR] Job 10: error inesperado "
                f"en par {feat_col}→{target_col}, "
                f"pista={pista}, direccion={direccion}: {exc}"
            ) from exc

    # ── Consolidar ──────────────────────────────────────────
    df_metricas = pd.DataFrame(
        [m for r in resultados for m in r["metricas"]]
    )
    _fi_list = [
        r["fi"] for r in resultados if not r["fi"].empty
    ]
    df_fi = (
        pd.concat(_fi_list, ignore_index=True)
        if _fi_list else pd.DataFrame()
    )

    if df_metricas.empty:
        print(
            "\n[DEBUG][ALERTA] Job 10: ningun par resulto modelable; "
            "no hay metricas que evaluar."
        )
        if SAVE_DATA:
            guardados = guardar_tablas_job10(df_metricas, df_fi)
            if DEBUG:
                print("[DEBUG] Job 10: CSV sin filas (resultado vacio):")
                for ruta in guardados:
                    print(f"  {ruta}")
        return

    df_metricas = calcular_mejora_vs_ar(df_metricas)

    # ── Reporte ─────────────────────────────────────────────
    print("\n" + "═" * 64)
    print(
        "REPORTE FINAL — MODELOS NO LINEALES "
        "SOBRE PARES VALIDADOS"
    )
    print("═" * 64)

    _cols_met = [
        "par", "pista", "direccion", "modelo",
        "n_train", "n_test", "rmse", "r2_oos",
        "mejora_vs_ar_pct",
    ]
    _cols_met = [
        c for c in _cols_met if c in df_metricas.columns
    ]
    print("\n[DEBUG] Tabla de métricas completa:")
    print(df_metricas[_cols_met].to_string(index=False))

    # TOP pares con mejora material
    feat_models = df_metricas[
        df_metricas["modelo"].isin([
            "random_forest", "xgboost", "gradient_boost",
            "lasso",
        ])
    ].dropna(subset=["mejora_vs_ar_pct"])
    # CAMBIO B3-4: solo modelos con n_test >= MIN_N_TEST.
    mejora_material = feat_models[
        feat_models["mejora_material"].apply(es_verdadero)
    ]

    print(f"\n{'─' * 64}")
    if not mejora_material.empty:
        print(
            f"PARES CON MEJORA MATERIAL (≥{UMBRAL_MEJORA}% "
            f"RMSE vs AR, n_test ≥ {MIN_N_TEST}):"
        )
        print("─" * 64)
        print(
            mejora_material[_cols_met].to_string(index=False)
        )
    else:
        print(
            "Ningún modelo evaluable (n_test ≥ "
            f"{MIN_N_TEST}) supera al AR baseline en "
            f"{UMBRAL_MEJORA}% de RMSE o más."
        )

    # Feature importance RF
    if not df_fi.empty:
        print(f"\n{'─' * 64}")
        print("FEATURE IMPORTANCE (Random Forest, top por par):")
        print("─" * 64)
        print(df_fi.to_string(index=False))

    # VEREDICTO: ¿no lineal agrega sobre lineal?
    print(f"\n{'═' * 64}")
    print("VEREDICTO")
    print("═" * 64)

    nl_models = ["random_forest", "xgboost", "gradient_boost"]
    nl_rows = df_metricas[
        df_metricas["modelo"].isin(nl_models)
    ].dropna(subset=["r2_oos"])
    lasso_rows = df_metricas[
        df_metricas["modelo"] == "lasso"
    ].dropna(subset=["r2_oos"])

    # CAMBIO B3-2 (Bloque 3): cada frase sale de una condición
    # sobre las cifras. Antes se afirmaba "aportan poder
    # predictivo" o "la relación lineal captura la estructura"
    # aunque el R² OOS fuera negativo.
    pares_nl_mejor: list[tuple[str, float, float]] = []
    n_comparables = 0
    for par in df_metricas["par"].unique():
        nl_par = nl_rows[nl_rows["par"] == par]
        ls_par = lasso_rows[lasso_rows["par"] == par]
        if nl_par.empty or ls_par.empty:
            continue
        n_comparables += 1
        best_nl_r2 = float(nl_par["r2_oos"].max())
        best_ls_r2 = float(ls_par["r2_oos"].max())
        if best_nl_r2 > best_ls_r2:
            pares_nl_mejor.append((par, best_nl_r2, best_ls_r2))

    if pares_nl_mejor:
        print(
            "El modelo no lineal supera al Lasso en R² OOS "
            f"en {len(pares_nl_mejor)} de {n_comparables} par(es):"
        )
        for p, r2_nl, r2_ls in pares_nl_mejor:
            _nota = (
                "" if r2_nl > 0
                else " — ambos R² ≤ 0: ninguno supera a la media "
                "del test"
            )
            print(
                f"  - {p}: R² NL={r2_nl:.3f} vs Lasso={r2_ls:.3f}"
                f"{_nota}"
            )
        print(
            "\nCAUTELA: con pocas features y n limitado, "
            "la mejora no lineal puede ser sobreajuste. "
            "Estos son indicios, no confirmación."
        )
    else:
        print(
            "Ningún modelo no lineal supera al Lasso en "
            f"R² OOS ({n_comparables} par(es) comparables)."
        )
        if not lasso_rows.empty and float(lasso_rows["r2_oos"].max()) > 0:
            print(
                "El Lasso tiene R² OOS > 0 en al menos un par: "
                "no hay evidencia de no linealidad explotable."
            )
        else:
            print(
                "Tampoco el Lasso logra R² OOS > 0: ningún modelo "
                "supera a la media del test; no se afirma "
                "estructura predictiva lineal ni no lineal."
            )

    # Mejora sobre AR (solo modelos evaluables, n_test ≥ MIN_N_TEST)
    _mm_r2_pos = mejora_material[mejora_material["r2_oos"] > 0]
    _mm_r2_neg = mejora_material[~(mejora_material["r2_oos"] > 0)]
    if not _mm_r2_pos.empty:
        print(
            f"\n{len(_mm_r2_pos)} modelo(s) superan al AR en "
            f"≥{UMBRAL_MEJORA}% RMSE con R² OOS > 0 "
            f"({', '.join(sorted(set(_mm_r2_pos['par'])))}): en "
            "esos pares las features validadas por Granger "
            "aportan poder predictivo fuera de muestra."
        )
    if not _mm_r2_neg.empty:
        print(
            f"\n{len(_mm_r2_neg)} modelo(s) superan al AR en "
            f"≥{UMBRAL_MEJORA}% RMSE pero con R² OOS ≤ 0 "
            f"({', '.join(sorted(set(_mm_r2_neg['par'])))}): "
            "mejora relativa al AR, sin poder predictivo frente "
            "a la media del test."
        )
    if mejora_material.empty:
        print(
            "\nNingún modelo evaluable supera al AR en "
            f"{UMBRAL_MEJORA}% RMSE o más."
        )
    _n_no_eval = int(
        (feat_models["mejora_material"] == NO_EVALUABLE).sum()
    )
    if _n_no_eval:
        print(
            f"{_n_no_eval} modelo(s) con n_test < {MIN_N_TEST}: "
            "mejora no evaluable (no cuentan arriba)."
        )

    print(
        "\nPrecedencia estadística (Granger + FDR global) "
        "validada en el notebook 18. No se afirma causalidad "
        "económica."
    )
    print("═" * 64)

    # ── Guardar ─────────────────────────────────────────────
    if SAVE_DATA:
        os.makedirs(DIR_REPORTES, exist_ok=True)
        os.makedirs(DIR_MODELOS, exist_ok=True)

        # CAMBIO E2E-10-01: sobreescribir ambos CSV de esta corrida.
        # Con datos conserva .to_csv(index=False) sobre el mismo DataFrame.
        guardados: list[str] = guardar_tablas_job10(df_metricas, df_fi)

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

