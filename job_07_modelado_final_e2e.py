"""# Job 7 - Modelos de Línea de Base FINAL (on-chain + RWA)"""

# 1. Configuración
DEBUG: bool = True
SAVE_DATA: bool = True

# 2. Imports

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg

from google.colab import drive

drive.mount("/content/drive")

"""# 3. Paths y Constantes"""

# CAMBIO E2E-07-01: ruta configurable; valor predeterminado original.
BASE_DIR: str = os.environ.get(
    "TESIS_BASE_DIR", "/content/drive/MyDrive/Tesis/expansion"
)
DIR_PROCESSED: str = os.path.join(BASE_DIR, "data", "processed")
DIR_REPORTES: str = os.path.join(
    BASE_DIR, "outputs", "reportes"
)
DIR_MODELOS: str = os.path.join(BASE_DIR, "outputs", "modelos")

ARCHIVO_EXT: str = os.path.join(
    DIR_PROCESSED, "series_diarias_ext.csv"
)
ARCHIVO_LARGAS: str = os.path.join(
    DIR_PROCESSED, "series_largas_ext.csv"
)
ARCHIVO_PRED: str = os.path.join(
    DIR_REPORTES, "predicciones_final.csv"
)
ARCHIVO_MET: str = os.path.join(
    DIR_REPORTES, "metricas_modelos_final.csv"
)
ARCHIVO_COEF: str = os.path.join(
    DIR_REPORTES, "coeficientes_lasso_final.csv"
)

# CAMBIO I4-2: se elimina el archivo cacheado
# series_diarias_ext_31jul.csv y el recorte fijo al 2026-07-31
# (artefacto de Ripple en DeFiLlama, ya no se verifica — ver
# Job 4). FECHA_CORTE=None usa la ventana completa; si se
# completa con una fecha ISO, recorta todas las columnas a la
# vez (ver cargar_dataset_ventana).
FECHA_CORTE: str | None = None

# ── Features ────────────────────────────────────────────────
COLS_FLUJO: list[str] = [
    "onchain_xlm_txcnt", "onchain_xlm_feetot",
    "onchain_xrp_txcnt", "onchain_xrp_feetot",
    "onchain_algo_txcnt", "onchain_algo_feetot",
]
COLS_STOCK: list[str] = [
    "onchain_xlm_adract",
    "onchain_xrp_adract",
    "onchain_algo_adract",
]
COLS_ONCHAIN: list[str] = COLS_FLUJO + COLS_STOCK
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

# Foco de interés: activo refugio (indicios NO confirmados)
COLS_RWA_FOCO: list[str] = [
    "rwa_tipo_treasurybills",
    "rwa_tipo_commodities",
]

# Bloque on-chain "histórico" = 9 métricas de red + USDY
COLS_BLOQUE_ONCHAIN: list[str] = COLS_ONCHAIN + [COL_USDY]
COLS_FEATURES: list[str] = COLS_BLOQUE_ONCHAIN + COLS_RWA

# Ablación: qué features entran en cada corrida
GRUPOS_FEATURES: dict[str, list[str]] = {
    "onchain": COLS_BLOQUE_ONCHAIN,
    "rwa": COLS_RWA,
    "todas": COLS_FEATURES,
}

COL_CISS: str = "macro_euro_ciss"
COL_STLFSI: str = "macro_eeuu_stlfsi"

# Todas las RWA son STOCKS (TVL): resample semanal con last()
COLS_STOCK_SEMANAL: list[str] = (
    COLS_STOCK + [COL_USDY] + COLS_RWA
)

# Lags de features (K modesto por n corto)
K_FEAT_A: int = 7   # Pista A diaria
K_FEAT_B: int = 4   # Pista B semanal

# Tamaño del conjunto de test
N_TEST_DIAS: int = 56    # ~8 semanas
N_TEST_SEM: int = 8      # semanas

# Parámetros AR del target solo
# CAMBIO H4 (documentación, sin cambio de comportamiento):
# Jobs 9/10 usan un único AR_LAGS_CONTROL=2 para ambas
# pistas, en vez de un valor por pista. No se unificó entre
# Jobs: el usuario confirmó dejarlo así y documentarlo.
AR_P_A: int = 3
AR_P_B: int = 2

# Ventana de la media móvil baseline
VENTANA_MA_A: int = 7
VENTANA_MA_B: int = 4

# CAMBIO C3-1: grilla de Ridge (reemplaza a la lista ALPHAS
# fija de 6 valores, compartida antes con Lasso). Lasso ya no
# usa una grilla fija — ver modelo_lasso (alphas=None).
RIDGE_ALPHAS: np.ndarray = np.logspace(-6, 3, 30)

# Umbral de aviso: features por observación de entrenamiento
RATIO_PN_AVISO: float = 0.5

NOTA_RWA: str = (
    "RWA de adopción incipiente (2025-2026): log-dif refleja "
    "crecimiento estructural, no respuesta a estrés."
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


"""# 4. Funciones"""


def cargar_series(
    ruta: str,
    col_fecha: str = "fecha",
) -> pd.DataFrame:
    """
    Carga un CSV de series con índice datetime.

    Args:
        ruta: Ruta del CSV.
        col_fecha: Nombre de la columna de fecha.

    Returns:
        DataFrame indexado por fecha y ordenado.
    """
    df = pd.read_csv(ruta, parse_dates=[col_fecha])
    df = df.set_index(col_fecha).sort_index()
    if DEBUG:
        print(f"[DEBUG] {os.path.basename(ruta)}: {df.shape}")
    return df


def cargar_dataset_ventana() -> pd.DataFrame:
    """
    Carga series_diarias_ext.csv y, si FECHA_CORTE está
    definido, lo recorta (aplicado a TODAS las columnas a la
    vez). Si FECHA_CORTE es None (default), usa la ventana
    completa.

    CAMBIO I4-2: reemplaza a cargar_dataset_recortado(), que
    dependía de un archivo cacheado (series_diarias_ext_31jul.csv)
    para un recorte fijo que ya no se aplica por defecto.

    Returns:
        DataFrame diario (completo o recortado).
    """
    df_ext = cargar_series(ARCHIVO_EXT)
    if FECHA_CORTE is None:
        if DEBUG:
            print(
                f"[DEBUG] sin recorte: ventana completa "
                f"({len(df_ext)} filas)"
            )
        return df_ext
    corte = pd.Timestamp(FECHA_CORTE)
    df_cortado = df_ext.loc[df_ext.index <= corte].copy()
    df_cortado.index.name = "fecha"
    if DEBUG:
        print(
            f"[DEBUG] recorte <= {FECHA_CORTE}: {len(df_ext)} → "
            f"{len(df_cortado)} filas"
        )
    return df_cortado


def validar_entrada_modelos(
    df: pd.DataFrame, ruta: str, job_origen: str,
) -> None:
    """CAMBIO E2E-07-01: validar contrato de CSV sin cambiar datos.

    La carga y el ordenamiento siguen a cargo de cargar_series.
    No transforma, imputa, filtra ni reordena observaciones.
    """
    if df.empty:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 7: entrada sin datos: {ruta} "
            f"(productor: {job_origen})."
        )
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 7: fechas invalidas en {ruta} "
            f"(productor: {job_origen})."
        )
    if df.index.has_duplicates:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 7: fechas duplicadas en {ruta} "
            f"(productor: {job_origen})."
        )


def es_muestra_insuficiente_modelos(exc: Exception) -> bool:
    """CAMBIO E2E-07-01: separar omision esperada de falla de codigo."""
    return (
        isinstance(exc, ValueError)
        and str(exc).startswith("Train demasiado pequeño:")
    )


def _log_diff(serie: pd.Series) -> pd.Series:
    """
    Log-diferencia segura: ln(x) solo donde x > 0.

    Args:
        serie: Serie de nivel.

    Returns:
        Serie de log-diferencias.
    """
    return np.log(serie.where(serie > 0)).diff()


def transformar_pista_a(
    df: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Pista A (diaria):
      X → log-dif de las columnas pedidas.
      y → primera diferencia de CISS.

    Args:
        df: Dataset diario recortado.
        cols: Features a transformar.

    Returns:
        (df_x, ser_y) alineados, con NaN en las primeras filas.
    """
    partes: list[pd.Series] = []
    for col in cols:
        if col not in df.columns:
            if DEBUG:
                print(f"[DEBUG] col ausente Pista A: {col}")
            continue
        partes.append(_log_diff(df[col]).rename(col))
    if not partes:
        raise ValueError("Pista A sin features disponibles")
    df_x = pd.concat(partes, axis=1)

    if COL_CISS not in df.columns:
        raise KeyError(f"Target no encontrado: {COL_CISS}")
    ser_y = df[COL_CISS].diff().rename("delta_ciss")
    return df_x, ser_y


def transformar_pista_b(
    df: pd.DataFrame,
    df_lrg: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Pista B (semanal):
      X → agregación semanal y log-dif. Los FLUJOS (txcnt,
          feetot) se suman; los STOCKS (adract, USDY y las 8
          RWA, que son TVL) se toman con last(): sumar un stock
          contaría siete veces el mismo capital.
      y → nivel STLFSI semanal (de series_largas_ext).

    Args:
        df: Dataset diario recortado.
        df_lrg: Dataset de series largas.
        cols: Features a transformar.

    Returns:
        (df_x, ser_y) en frecuencia semanal.
    """
    partes: list[pd.Series] = []
    for col in cols:
        if col not in df.columns:
            continue
        if col in COLS_FLUJO:
            serie_sem = df[col].resample("W").sum()
        elif col in COLS_STOCK_SEMANAL:
            serie_sem = df[col].resample("W").last()
        else:
            serie_sem = df[col].resample("W").mean()
        partes.append(_log_diff(serie_sem).rename(col))
    if not partes:
        raise ValueError("Pista B sin features disponibles")
    df_x = pd.concat(partes, axis=1)

    if COL_STLFSI in df_lrg.columns:
        fuente = df_lrg[COL_STLFSI]
    elif COL_STLFSI in df.columns:
        fuente = df[COL_STLFSI]
    else:
        raise KeyError(f"Target ausente: {COL_STLFSI}")
    # CAMBIO I4-2: el target se recorta a la misma ventana que
    # `df` (ya sea completa o cortada por FECHA_CORTE), en vez
    # de a un FECHA_CORTE que ahora puede ser None.
    fuente = fuente.loc[fuente.index <= df.index.max()]
    ser_y = (
        fuente.resample("W").last().dropna()
        .rename("stlfsi_nivel")
    )
    return df_x, ser_y


def construir_features_lag(
    df_x: pd.DataFrame,
    K: int,
) -> pd.DataFrame:
    """
    Crea features rezagadas lag 1..K.

    Cada columna 'c' en lag k pasa a 'c_lag{k}'. Las primeras K
    filas quedan NaN y se eliminan en el split.

    Args:
        df_x: Features contemporáneas.
        K: Lag máximo.

    Returns:
        DataFrame de features rezagadas.
    """
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
    n_test: int,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
]:
    """
    División temporal: las últimas n_test observaciones son test.

    No hay barajado: el orden temporal se respeta, de modo que
    el modelo nunca ve el futuro durante el entrenamiento.

    Args:
        X_lag: Features rezagadas.
        ser_y: Target.
        n_test: Observaciones de test.

    Returns:
        (X_train, X_test, y_train, y_test).
    """
    combinado = pd.concat(
        [ser_y.rename("__y"), X_lag], axis=1
    ).dropna()
    n = len(combinado)
    if n - n_test < 10:
        raise ValueError(
            f"Train demasiado pequeño: {n - n_test} obs "
            f"(total={n}, n_test={n_test})"
        )
    y_all = combinado.pop("__y")
    X_all = combinado
    n_train = n - n_test
    if DEBUG:
        ratio = (
            X_all.shape[1] / n_train if n_train else float("inf")
        )
        print(
            f"[DEBUG] split: n={n}, train={n_train}, "
            f"test={n_test}, features={X_all.shape[1]}, "
            f"p/n={ratio:.2f}"
        )
        if ratio > RATIO_PN_AVISO:
            print(
                f"[DEBUG] AVISO: {X_all.shape[1]} features "
                f"para {n_train} obs de train (p/n="
                f"{ratio:.2f}). La regularización es "
                "imprescindible; resultados orientativos."
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
    grupo: str,
) -> dict:
    """
    MSE, RMSE, MAE y R² OOS entre dos Series alineadas.

    R² OOS < 0 significa que el modelo predice peor que la media
    del test: es un resultado informativo, no un error.

    Args:
        y_true: Valores reales del test.
        y_pred: Predicciones.
        nombre: Nombre del modelo.
        pista: "A" o "B".
        grupo: Grupo de features usado.

    Returns:
        Dict de métricas.
    """
    yt = y_true.to_numpy(dtype=float)
    yp = y_pred.to_numpy(dtype=float)
    mask = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[mask], yp[mask]
    base = {
        "pista": pista,
        "grupo_features": grupo,
        "modelo": nombre,
    }
    if len(yt) == 0:
        return {
            **base, "n_test": 0, "mse": None,
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
        "mse": round(mse, 8),
        "rmse": round(float(np.sqrt(mse)), 8),
        "mae": round(float(np.mean(np.abs(err))), 8),
        "r2_oos": round(r2, 4),
    }


def baseline_persistencia(
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.Series:
    """
    Persistencia: y_pred[t] = y_real[t-1].

    Args:
        y_train: Target de entrenamiento.
        y_test: Target de test.

    Returns:
        Serie de predicciones sobre el índice de test.
    """
    y_all = pd.concat([y_train, y_test])
    return (
        y_all.shift(1).loc[y_test.index].rename("persistencia")
    )


def baseline_media_movil(
    y_train: pd.Series,
    y_test: pd.Series,
    ventana: int,
) -> pd.Series:
    """
    Media móvil: y_pred[t] = media(y[t-ventana:t-1]).

    Args:
        y_train: Target de entrenamiento.
        y_test: Target de test.
        ventana: Tamaño de la ventana.

    Returns:
        Serie de predicciones sobre el índice de test.
    """
    y_all = pd.concat([y_train, y_test])
    return (
        y_all.rolling(ventana).mean().shift(1)
        .loc[y_test.index].rename(f"ma{ventana}")
    )


def ar_un_paso(
    y_train: pd.Series, y_test: pd.Series, p: int,
) -> pd.Series:
    """
    Pronóstico AR(p) a un paso, con el valor REAL del período
    anterior (no el pronosticado por el propio modelo).

    CAMBIO C2-1: reemplaza el pronóstico dinámico (predict()
    desde el final del entrenamiento hasta el final del test,
    que encadena sus propios pronósticos y acumula error).
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
    pista: str,
) -> tuple[pd.Series, object, bool]:
    """
    AR(p) del target solo, ajustado sobre y_train, pronóstico a
    un paso (ar_un_paso).

    Es el baseline que los modelos con features deben superar
    para poder afirmar que las señales aportan algo.

    CAMBIO C1-1: se quitó old_names=False (deprecado/eliminado
    según versión de statsmodels).
    CAMBIO C1-3: sin fallback silencioso — si el ajuste falla,
    se registra con [DEBUG][ALERTA] y se devuelve la media,
    nombrada "ar_media" (nunca "ar{p}"), con baseline_ok=False.

    Args:
        y_train: Target de entrenamiento.
        y_test: Target de test.
        p: Orden autorregresivo.
        pista: Etiqueta de pista.

    Returns:
        (predicciones, modelo, baseline_ok).
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = ar_un_paso(y_train, y_test, p)
            res = AutoReg(
                y_train.to_numpy(), lags=p, trend="c"
            ).fit()
        if DEBUG:
            print(f"[DEBUG] AR({p}) Pista {pista} OK")
        return pred, res, True
    except Exception as exc:
        print(f"[DEBUG][ALERTA] AR({p}) Pista {pista} falló: {exc}")
        media = float(y_train.mean())
        return (
            pd.Series(
                [media] * len(y_test),
                index=y_test.index, name="ar_media",
            ),
            None, False,
        )


def _cv_splits(n_train: int) -> int:
    """
    Número de splits para la CV temporal según el tamaño.

    Args:
        n_train: Observaciones de entrenamiento.

    Returns:
        Número de splits, entre 2 y 5.
    """
    return max(2, min(5, n_train // 15))


def _chequear_alpha_borde(
    alpha: float, grilla: np.ndarray, etiqueta: str,
) -> bool:
    """
    True si `alpha` es el primero o el último valor de `grilla`
    — la CV no encontró un óptimo interior.

    CAMBIO C3-3.
    """
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
) -> tuple[pd.Series, object, StandardScaler, bool]:
    """
    Ridge con CV temporal (TimeSeriesSplit) y estandarización.

    CAMBIO C3-1: grilla RIDGE_ALPHAS = np.logspace(-6, 3, 30).
    CAMBIO C3-3: avisa si el alpha elegido queda en el borde.
    CAMBIO C3-4: reporta la desviación estándar de y (escala).

    Args:
        X_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        X_test: Features de test.
        etiqueta: Etiqueta para el log.

    Returns:
        (predicciones, modelo, scaler, alpha_en_borde).
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
) -> tuple[pd.Series, object, StandardScaler, bool]:
    """
    Lasso con CV temporal y estandarización.

    Es el test central del notebook: si Lasso lleva todos los
    coeficientes a cero, la selección de features concluye que
    ninguna señal —on-chain ni RWA— aporta información sobre el
    target más allá del ruido. Ese descarte es el hallazgo.

    CAMBIO C3-2: alphas=None, n_alphas=100, eps=1e-3,
    max_iter=10000 — el camino de penalizaciones se calcula
    sobre los datos (se quitó el filtro "a >= 0.1" de ALPHAS).
    CAMBIO C3-3: avisa si el alpha elegido queda en el borde.
    CAMBIO C3-4: reporta la desviación estándar de y (escala).

    Args:
        X_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        X_test: Features de test.
        etiqueta: Etiqueta para el log.

    Returns:
        (predicciones, modelo, scaler, alpha_en_borde).
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
    modelo: object,
    columnas: list[str],
    pista: str,
    grupo: str,
) -> pd.DataFrame:
    """
    Extrae los coeficientes no nulos de un Lasso ajustado.

    Args:
        modelo: LassoCV ajustado (o None).
        columnas: Nombres de las features.
        pista: Etiqueta de pista.
        grupo: Grupo de features.

    Returns:
        DataFrame con las features retenidas, ordenadas por
        |coef|. Vacío si el modelo descartó todas.
    """
    if modelo is None or not hasattr(modelo, "coef_"):
        return pd.DataFrame()
    coefs = np.asarray(modelo.coef_, dtype=float)
    no_nulos = np.flatnonzero(coefs != 0)
    if no_nulos.size == 0:
        return pd.DataFrame()
    filas = [
        {
            "pista": pista,
            "grupo_features": grupo,
            "feature": columnas[i],
            "coef": round(float(coefs[i]), 8),
            "bloque": (
                "rwa" if columnas[i].startswith("rwa_")
                else "onchain"
            ),
        }
        for i in no_nulos
    ]
    df_coef = pd.DataFrame(filas)
    return df_coef.reindex(
        df_coef["coef"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def armar_predicciones(
    y_test: pd.Series,
    predicciones: dict[str, pd.Series],
    pista: str,
    grupo: str,
    target_nombre: str,
) -> pd.DataFrame:
    """
    Consolida las predicciones de una corrida en un DataFrame.

    Args:
        y_test: Target real del test.
        predicciones: Dict nombre_modelo → predicciones.
        pista: Etiqueta de pista.
        grupo: Grupo de features.
        target_nombre: Nombre del target.

    Returns:
        DataFrame largo con una fila por fecha.
    """
    datos = {"target_actual": y_test}
    datos.update(predicciones)
    df_pred = pd.DataFrame(datos)
    df_pred.index.name = "fecha"
    df_pred.insert(0, "pista", pista)
    df_pred.insert(1, "grupo_features", grupo)
    df_pred.insert(2, "target", target_nombre)
    return df_pred.reset_index()


def correr_pista(
    X_base: pd.DataFrame,
    ser_y: pd.Series,
    cols_grupo: list[str],
    K: int,
    n_test: int,
    ar_p: int,
    ventana_ma: int,
    pista: str,
    grupo: str,
    target_nombre: str,
) -> dict:
    """
    Corre baselines y modelos con features para un grupo.

    Los baselines (persistencia, media móvil, AR) dependen solo
    del target, así que son idénticos entre grupos: se calculan
    igual en cada uno para que la comparación sea directa dentro
    de la misma tabla.

    Args:
        X_base: Features contemporáneas de la pista.
        ser_y: Target.
        cols_grupo: Columnas del grupo de ablación.
        K: Lags de features.
        n_test: Tamaño del test.
        ar_p: Orden AR.
        ventana_ma: Ventana de la media móvil.
        pista: "A" o "B".
        grupo: Nombre del grupo.
        target_nombre: Nombre del target.

    Returns:
        Dict con métricas, predicciones, coeficientes y modelos.
    """
    cols_ok = [c for c in cols_grupo if c in X_base.columns]
    etiqueta = f"{pista}/{grupo}"
    if not cols_ok:
        raise ValueError(f"{etiqueta}: sin features")

    X_lag = construir_features_lag(X_base[cols_ok], K)
    Xtr, Xts, ytr, yts = split_temporal(X_lag, ser_y, n_test)

    preds: dict[str, pd.Series] = {}
    preds["persistencia"] = baseline_persistencia(ytr, yts)
    preds[f"ma{ventana_ma}"] = baseline_media_movil(
        ytr, yts, ventana_ma
    )
    pred_ar, mdl_ar, ar_ok = modelo_ar(ytr, yts, ar_p, etiqueta)
    # CAMBIO C1-3: la clave usa el nombre real de la serie
    # (ar{p} si el ajuste anduvo, "ar_media" si no), no un
    # f"ar{ar_p}" fijo que mentiría sobre qué modelo corrió.
    preds[pred_ar.name] = pred_ar
    pred_ridge, mdl_ridge, _, ridge_borde = modelo_ridge(
        Xtr, ytr, Xts, etiqueta
    )
    preds["ridge"] = pred_ridge
    pred_lasso, mdl_lasso, _, lasso_borde = modelo_lasso(
        Xtr, ytr, Xts, etiqueta
    )
    preds["lasso"] = pred_lasso

    metricas = [
        metricas_regresion(yts, serie, nombre, pista, grupo)
        for nombre, serie in preds.items()
    ]
    # CAMBIO C3-3: alpha_en_borde por fila (solo aplica a
    # ridge/lasso; el resto no usa alpha de regularización).
    _borde_por_modelo = {"ridge": ridge_borde, "lasso": lasso_borde}
    for fila in metricas:
        fila["n_train"] = int(len(ytr))
        fila["n_features"] = int(Xtr.shape[1])
        fila["nota"] = NOTA_RWA if grupo != "onchain" else ""
        # CAMBIO C1-3: baseline_ok solo es relevante para la
        # fila del AR; el resto no depende de ese ajuste.
        fila["baseline_ok"] = (
            ar_ok if fila["modelo"] == pred_ar.name else True
        )
        fila["alpha_en_borde"] = _borde_por_modelo.get(
            fila["modelo"], False
        )

    return {
        "metricas": metricas,
        "predicciones": armar_predicciones(
            yts, preds, pista, grupo, target_nombre
        ),
        "coeficientes": coeficientes_lasso(
            mdl_lasso, list(Xtr.columns), pista, grupo
        ),
        "modelos": {
            f"modelo_ar_{pista.lower()}_{grupo}": mdl_ar,
            f"modelo_ridge_{pista.lower()}_{grupo}": mdl_ridge,
            f"modelo_lasso_{pista.lower()}_{grupo}": mdl_lasso,
        },
    }


"""# 5. Proceso"""

# CAMBIO B1-4 (Bloque 1): hash de las entradas vs. manifest.
verificar_entradas_snapshot([ARCHIVO_EXT, ARCHIVO_LARGAS], "Job 7")

print(
    "\n[LÍNEA DE BASE] Re-ejecución final (on-chain + RWA).\n"
    "[LÍNEA DE BASE] Es el punto cero: no se busca un predictor "
    "que funcione, se documenta el poder predictivo real.\n"
    "[LÍNEA DE BASE] Un resultado nulo o débil es válido."
)

# ── Carga de datos ──────────────────────────────────────────
# CAMBIO E2E-07-01: dos archivos obligatorios del Job 3.
# No se reemplazan por archivos anteriores ni datos sinteticos.
for _ruta in (ARCHIVO_EXT, ARCHIVO_LARGAS):
    if not os.path.isfile(_ruta):
        raise FileNotFoundError(
            f"[DEBUG][ERROR] Job 7: falta {_ruta}. "
            "Debe generarlo el Job 3."
        )
try:
    df = cargar_dataset_ventana()
    df_lrg = cargar_series(ARCHIVO_LARGAS)
except Exception as _exc:
    raise RuntimeError(
        f"[DEBUG][ERROR] Job 7: error leyendo archivos "
        f"de Job 3: {_exc}"
    ) from _exc

validar_entrada_modelos(df, ARCHIVO_EXT, "Job 3")
validar_entrada_modelos(df_lrg, ARCHIVO_LARGAS, "Job 3")
if COL_CISS not in df.columns:
    raise RuntimeError(
        f"[DEBUG][ERROR] Job 7: falta {COL_CISS} en {ARCHIVO_EXT}. "
        "El target diario es obligatorio."
    )
if COL_STLFSI not in df_lrg.columns and COL_STLFSI not in df.columns:
    raise RuntimeError(
        f"[DEBUG][ERROR] Job 7: falta {COL_STLFSI} "
        f"en {ARCHIVO_LARGAS} y {ARCHIVO_EXT}."
    )
# CAMBIO B2-3 (Bloque 2): rwa_red_ripple sale de COLS_RWA si no
# existe en los datos (columna ausente o sin ningún valor). Se
# decide por existencia, no por resultados; se informa en el log.
_COL_RIPPLE = "rwa_red_ripple"
if _COL_RIPPLE in COLS_RWA and (
    _COL_RIPPLE not in df.columns or df[_COL_RIPPLE].notna().sum() == 0
):
    for _lista in (COLS_RWA, COLS_FEATURES, COLS_STOCK_SEMANAL):
        if _COL_RIPPLE in _lista:
            _lista.remove(_COL_RIPPLE)
    print(
        f"[DEBUG][ALERTA] Job 7: {_COL_RIPPLE} sin datos en "
        f"{os.path.basename(ARCHIVO_EXT)}: se quita de COLS_RWA "
        f"({len(COLS_RWA)} series RWA)."
    )
elif DEBUG:
    print(f"[DEBUG] Job 7: {_COL_RIPPLE} presente, queda en COLS_RWA")

if not any(c in df.columns for c in COLS_FEATURES):
    raise RuntimeError(
        "[DEBUG][ERROR] Job 7: ninguna feature configurada "
        f"esta presente en {ARCHIVO_EXT}."
    )

if DEBUG:
    print(
        f"[DEBUG] ventana: {df.index.min().date()} → "
        f"{df.index.max().date()} ({len(df)} días, "
        f"{df.shape[1]} columnas)"
    )
    _falta = [c for c in COLS_FEATURES if c not in df.columns]
    if _falta:
        print(f"[DEBUG] features AUSENTES: {_falta}")
    _foco = [c for c in COLS_RWA_FOCO if c in df.columns]
    print(
        f"[DEBUG] foco RWA (refugio, indicios no "
        f"confirmados): {_foco}"
    )

# CAMBIO B2-4 (Bloque 2): la línea de base on-chain queda en la
# ventana 2026 de series_diarias_ext (on-chain diario solo existe
# ahí) y se declara en el reporte. No se compara con la ventana
# larga de los Jobs 9-14.
VENTANA_INICIO_REPORTE: str = str(df.index.min().date())
VENTANA_FIN_REPORTE: str = str(df.index.max().date())
NOTA_VENTANA_REPORTE: str = (
    "ventana 2026 on-chain (series_diarias_ext); no comparable "
    "con la ventana larga de Jobs 9-14"
)
if df.index.min().year != 2026 or df.index.max().year != 2026:
    print(
        f"[DEBUG][ALERTA] Job 7: la ventana "
        f"{VENTANA_INICIO_REPORTE} → {VENTANA_FIN_REPORTE} no es "
        "2026; revisar series_diarias_ext."
    )
print(
    f"[DEBUG] Job 7: ventana declarada {VENTANA_INICIO_REPORTE} → "
    f"{VENTANA_FIN_REPORTE} ({NOTA_VENTANA_REPORTE})"
)

resultados: list[dict] = []
modelos_guardar: dict[str, object] = {}

# ── PISTA A: ΔCISS (diaria) ─────────────────────────────────
print("\n[DEBUG] ── PISTA A: ΔCISS diaria ──")
try:
    X_a, y_a = transformar_pista_a(df, COLS_FEATURES)
    for _grupo, _cols in GRUPOS_FEATURES.items():
        try:
            _res = correr_pista(
                X_a, y_a, _cols, K_FEAT_A, N_TEST_DIAS,
                AR_P_A, VENTANA_MA_A, "A", _grupo,
                "delta_ciss",
            )
            resultados.append(_res)
            modelos_guardar.update(_res["modelos"])
        except Exception as _e:
            # CAMBIO E2E-07-01: la muestra insuficiente sigue
            # sin modelarse; otras excepciones no se silencian.
            if not es_muestra_insuficiente_modelos(_e):
                raise RuntimeError(
                    f"[DEBUG][ERROR] Job 7: fallo A/{_grupo}: {_e}"
                ) from _e
            if DEBUG:
                print(f"[DEBUG][ALERTA] Pista A/{_grupo} no evaluable: {_e}")
except Exception as _exc_a:
    raise RuntimeError(
        f"[DEBUG][ERROR] Job 7: fallo la Pista A: {_exc_a}"
    ) from _exc_a

# ── PISTA B: STLFSI semanal ─────────────────────────────────
print("\n[DEBUG] ── PISTA B: STLFSI semanal ──")
try:
    X_b, y_b = transformar_pista_b(df, df_lrg, COLS_FEATURES)
    print(
        f"[DEBUG] Pista B: {len(y_b)} semanas de target. "
        "Train muy corto: alta varianza, resultados "
        "orientativos."
    )
    for _grupo, _cols in GRUPOS_FEATURES.items():
        try:
            _res = correr_pista(
                X_b, y_b, _cols, K_FEAT_B, N_TEST_SEM,
                AR_P_B, VENTANA_MA_B, "B", _grupo,
                "stlfsi_nivel",
            )
            resultados.append(_res)
            modelos_guardar.update(_res["modelos"])
        except Exception as _e:
            # CAMBIO E2E-07-01: no evaluable por n_train sigue
            # excluido, pero errores inesperados detienen el Job.
            if not es_muestra_insuficiente_modelos(_e):
                raise RuntimeError(
                    f"[DEBUG][ERROR] Job 7: fallo B/{_grupo}: {_e}"
                ) from _e
            if DEBUG:
                print(f"[DEBUG][ALERTA] Pista B/{_grupo} no evaluable: {_e}")
except Exception as _exc_b:
    raise RuntimeError(
        f"[DEBUG][ERROR] Job 7: fallo la Pista B: {_exc_b}"
    ) from _exc_b

# ── Consolidación ───────────────────────────────────────────
# CAMBIO E2E-07-01: una corrida sin modelos evaluables no puede
# mostrarse como "Lasso descarto todo" ni darse por exitosa.
if not resultados:
    raise RuntimeError(
        "[DEBUG][ERROR] Job 7: ninguna pista/grupo produjo "
        "metricas evaluables. No se guardan resultados de esta corrida."
    )

df_metricas: pd.DataFrame = pd.DataFrame(
    [m for r in resultados for m in r["metricas"]]
)
df_pred_all: pd.DataFrame = (
    pd.concat(
        [r["predicciones"] for r in resultados],
        ignore_index=True,
    )
    if resultados else pd.DataFrame()
)
_coefs = [
    r["coeficientes"] for r in resultados
    if not r["coeficientes"].empty
]
df_coefs: pd.DataFrame = (
    pd.concat(_coefs, ignore_index=True)
    if _coefs else pd.DataFrame()
)

# ── Tabla comparativa y lectura ─────────────────────────────
if not df_metricas.empty:
    print("\n[LÍNEA DE BASE] Métricas por pista y grupo:")
    _cols_met = [
        "pista", "grupo_features", "modelo", "n_train",
        "n_features", "n_test", "rmse", "mae", "r2_oos",
    ]
    print(df_metricas[_cols_met].to_string(index=False))
    print(
        "\n[LÍNEA DE BASE] Lectura: R² OOS < 0 significa que el "
        "modelo predice peor que la media del test. Con este n "
        "es un resultado esperable, no un error de cómputo."
    )

    # ¿Algún modelo con features supera a los baselines?
    for _pista in ["A", "B"]:
        _sub = df_metricas[df_metricas["pista"] == _pista]
        if _sub.empty:
            continue
        _base = _sub[
            _sub["modelo"].str.startswith(("persistencia", "ar"))
        ]
        if _base.empty or _base["rmse"].isna().all():
            continue
        _rmse_base = float(_base["rmse"].min())
        _mejor_base = str(
            _base.loc[_base["rmse"].idxmin(), "modelo"]
        )
        _con_feat = _sub[
            _sub["modelo"].isin(["ridge", "lasso"])
        ].dropna(subset=["rmse"])
        _superan = _con_feat[_con_feat["rmse"] < _rmse_base]
        print(
            f"\n[LÍNEA DE BASE] Pista {_pista} — mejor baseline "
            f"sin features: {_mejor_base} (RMSE={_rmse_base:.6g})"
        )
        if _superan.empty:
            print(
                "  → Ningún modelo con features supera al "
                "baseline. Las señales no aportan poder "
                "predictivo en esta ventana."
            )
        else:
            for _, _f in _superan.iterrows():
                print(
                    f"  → {_f['modelo']} ({_f['grupo_features']})"
                    f" RMSE={_f['rmse']:.6g} < baseline. "
                    "Diferencia marginal puede ser ruido: "
                    "confirmar en 16_validacion_final."
                )

# ── Qué retuvo el Lasso (el test central) ───────────────────
print("\n[LÍNEA DE BASE] Features retenidas por Lasso:")
if df_coefs.empty:
    print(
        "  → Lasso descartó TODAS las features en todas las "
        "corridas (todos los coeficientes a cero), incluidas "
        "las 8 RWA. La selección concluye que ninguna señal "
        "aporta información sobre el target más allá del ruido. "
        "ESE DESCARTE ES EL HALLAZGO."
    )
else:
    print(df_coefs.to_string(index=False))
    _rwa_ret = df_coefs[df_coefs["bloque"] == "rwa"]
    print(
        f"  → features retenidas: {len(df_coefs)} "
        f"({len(_rwa_ret)} del bloque RWA)"
    )
    _foco_ret = df_coefs[
        df_coefs["feature"].str.startswith(
            tuple(COLS_RWA_FOCO)
        )
    ]
    if not _foco_ret.empty:
        print(
            "  → incluye features del foco de refugio "
            "(treasurybills/commodities): candidatas a "
            "confirmar en Granger, NO señal confirmada."
        )
print(f"\n[LÍNEA DE BASE] {NOTA_RWA}")
_nota_ventana = (
    f"recortada a {FECHA_CORTE}"
    if FECHA_CORTE is not None else "completa (sin recorte)"
)
print(
    f"[LÍNEA DE BASE] Limitaciones: ventana {_nota_ventana} "
    f"{VENTANA_INICIO_REPORTE} → {VENTANA_FIN_REPORTE} "  # CAMBIO B2-4
    f"({NOTA_VENTANA_REPORTE}), "
    "test de 56 días / 8 semanas, p/n alto y lag post-hoc en "
    "fases previas."
)

"""# 6. Save"""

if SAVE_DATA:
    os.makedirs(DIR_PROCESSED, exist_ok=True)
    os.makedirs(DIR_REPORTES, exist_ok=True)
    os.makedirs(DIR_MODELOS, exist_ok=True)

    _guardados: list[str] = []
    df_metricas["ventana_inicio"] = VENTANA_INICIO_REPORTE  # CAMBIO B2-4
    df_metricas["ventana_fin"] = VENTANA_FIN_REPORTE
    df_metricas["ventana_nota"] = NOTA_VENTANA_REPORTE

    # CAMBIO I4-2: ya no se cachea un dataset recortado aparte.

    # CAMBIO E2E-07-01: publicar todos los CSV de ESTA corrida.
    # Coeficientes vacios = cero filas (no "archivo ausente").
    # No se altera ninguna tabla que ya contiene registros.
    _salidas_csv: list[tuple[str, pd.DataFrame, list[str]]] = [
        (
            ARCHIVO_PRED, df_pred_all,
            ["fecha", "pista", "grupo_features", "target",
             "target_actual", "persistencia", "ridge", "lasso"],
        ),
        (
            ARCHIVO_MET, df_metricas,
            ["pista", "grupo_features", "modelo", "n_test",
             "mse", "rmse", "mae", "r2_oos", "n_train",
             "n_features", "nota", "baseline_ok", "alpha_en_borde",
             "ventana_inicio", "ventana_fin", "ventana_nota"],
        ),
        (
            ARCHIVO_COEF, df_coefs,
            ["pista", "grupo_features", "feature", "coef", "bloque"],
        ),
    ]
    for _ruta, _tabla, _columnas in _salidas_csv:
        _salida = _tabla if not _tabla.empty else pd.DataFrame(columns=_columnas)
        _salida.to_csv(_ruta, index=False)
        _guardados.append(_ruta)

    for _nombre, _modelo in modelos_guardar.items():
        if _modelo is None:
            continue
        _ruta = os.path.join(DIR_MODELOS, f"{_nombre}.joblib")
        joblib.dump(_modelo, _ruta)
        _guardados.append(_ruta)

    if DEBUG:
        for _g in _guardados:
            print(f"[DEBUG] guardado: {_g}")
elif DEBUG:
    print(
        "\n[DEBUG] SAVE_DATA=False — predicciones, métricas y "
        "modelos solo en memoria"
    )

