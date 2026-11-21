"""# Job 8 - Validación y Reporte de Línea de Base FINAL"""

# 1. Configuración

DEBUG: bool = True
SAVE_DATA: bool = True

# 2. Imports

import contextlib
import io
import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR  # CAMBIO I5-1 (Job 6)
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller  # CAMBIO I5-6 (Job 6)
from statsmodels.tsa.stattools import grangercausalitytests

from google.colab import drive

drive.mount("/content/drive")

"""# 3. Paths y Constantes"""

# CAMBIO E2E-08-01: igual ruta predeterminada, configurable por entorno.
BASE_DIR: str = os.environ.get(
    "TESIS_BASE_DIR", "/content/drive/MyDrive/Tesis/expansion"
)
DIR_PROCESSED: str = os.path.join(BASE_DIR, "data", "processed")
DIR_REPORTES: str = os.path.join(
    BASE_DIR, "outputs", "reportes"
)

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
ARCHIVO_GRANGER: str = os.path.join(
    DIR_REPORTES, "granger_final.csv"
)
# CAMBIO E2E-08-01: tabla propia del Job 8, sin pisar al Job 4.
# Sigue fuera de la familia BH (misma tabla y mismos valores).
ARCHIVO_GRANGER_REZAGO: str = os.path.join(
    DIR_REPORTES, "granger_final_por_rezago.csv"
)
ARCHIVO_CONF: str = os.path.join(
    DIR_REPORTES, "granger_final_confirmacion.csv"
)
ARCHIVO_BT: str = os.path.join(
    DIR_REPORTES, "backtesting_final.csv"
)
ARCHIVO_EVENTOS: str = os.path.join(
    DIR_REPORTES, "eventos_estres_final.csv"
)
ARCHIVO_REPORTE: str = os.path.join(
    DIR_REPORTES, "reporte_linea_base_final.csv"
)

# CAMBIO I4-3: se elimina series_diarias_ext_31jul.csv y el
# recorte fijo al 2026-07-31 (ver Job 4/5). FECHA_CORTE=None usa
# la ventana completa.
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
COLS_FEATURES: list[str] = (
    COLS_ONCHAIN + [COL_USDY] + COLS_RWA
)

# Todas las RWA y USDY son stocks: resample semanal con last()
COLS_STOCK_SEMANAL: list[str] = (
    COLS_STOCK + [COL_USDY] + COLS_RWA
)

COL_CISS: str = "macro_euro_ciss"
COL_STLFSI: str = "macro_eeuu_stlfsi"

# Confirmación específica: indicios de 14_relaciones_rwa
# (activo refugio). Se testean para CONFIRMAR O REFUTAR.
CONFIRMACIONES: list[tuple[str, str]] = [
    ("rwa_tipo_treasurybills", "B"),
    ("rwa_tipo_commodities", "B"),
]

# Lags de Granger
K_GRANGER_D: int = 14   # Pista A diaria
K_GRANGER_S: int = 6    # Pista B semanal

ALFA_BH: float = 0.05
PERCENTIL_EVENTO: int = 80

# Bajo estos umbrales se advierte potencia/estabilidad
N_MIN_EVENTOS: int = 10
N_MIN_SEMANAS: int = 40

# Mejora mínima sobre el baseline (%) para considerarla algo
# más que ruido de un test corto. Por debajo de esto, la
# diferencia de RMSE no es distinguible de la variabilidad
# muestral con 56 días / 8 semanas de test.
UMBRAL_MEJORA_MATERIAL: float = 5.0

NOTA_RWA: str = (
    "RWA de adopción incipiente (2025-2026): dinámica dominada "
    "por crecimiento estructural, no por respuesta a estrés."
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
    definido, lo recorta (aplicado a todas las columnas a la
    vez). Si FECHA_CORTE es None (default), usa la ventana
    completa.

    CAMBIO I4-3: reemplaza a cargar_dataset_recortado(), que
    dependía de series_diarias_ext_31jul.csv.

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
    if DEBUG:
        print(f"[DEBUG] se recorta ext a <= {FECHA_CORTE}")
    corte = pd.Timestamp(FECHA_CORTE)
    df_cortado = df_ext.loc[df_ext.index <= corte].copy()
    df_cortado.index.name = "fecha"
    return df_cortado


def validar_entrada_series_job8(
    df: pd.DataFrame, ruta: str, job_productor: str,
) -> None:
    """CAMBIO E2E-08-01: validar entrada sin modificar observaciones."""
    if df.empty:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 8: serie sin registros: {ruta} "
            f"(productor: {job_productor})."
        )
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 8: fechas invalidas en {ruta} "
            f"(productor: {job_productor})."
        )
    if df.index.has_duplicates:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 8: fechas duplicadas en {ruta} "
            f"(productor: {job_productor})."
        )


def leer_csv_job8(
    ruta: str,
    columnas_obligatorias: list[str],
    job_productor: str,
    permitir_vacio: bool = False,
    parse_dates: list[str] | None = None,
) -> pd.DataFrame:
    """CAMBIO E2E-08-01: diferenciar archivo ausente de cero filas.

    CSV con solo encabezados es valido EXCLUSIVAMENTE si permitir_vacio.
    No intenta reconstruir ni recuperar datos de ejecuciones previas.
    """
    if not os.path.isfile(ruta):
        raise FileNotFoundError(
            f"[DEBUG][ERROR] Job 8: falta {ruta} "
            f"(productor: Job {job_productor})."
        )
    try:
        tabla = pd.read_csv(ruta, parse_dates=parse_dates)
    except (OSError, ValueError, pd.errors.ParserError,
            pd.errors.EmptyDataError) as exc:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 8: CSV ilegible: {ruta} "
            f"(productor: Job {job_productor}): {exc}"
        ) from exc
    faltantes = [c for c in columnas_obligatorias if c not in tabla.columns]
    if faltantes:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 8: columnas faltantes {faltantes} "
            f"en {ruta} (productor: Job {job_productor})."
        )
    if tabla.empty and not permitir_vacio:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 8: CSV sin observaciones: {ruta} "
            f"(productor: Job {job_productor})."
        )
    if parse_dates is not None and not tabla.empty:
        for columna in parse_dates:
            fechas = pd.to_datetime(tabla[columna], errors="coerce")
            if fechas.isna().any():
                raise RuntimeError(
                    f"[DEBUG][ERROR] Job 8: fechas invalidas "
                    f"en {ruta}, columna {columna}."
                )
    return tabla


def _log_diff(serie: pd.Series) -> pd.Series:
    """
    Log-diferencia segura: ln(x) solo donde x > 0.

    Args:
        serie: Serie de nivel.

    Returns:
        Serie de log-diferencias.
    """
    return np.log(serie.where(serie > 0)).diff()


def _granger_raw(data: pd.DataFrame, max_lag: int) -> dict:
    """
    Llama a grangercausalitytests de forma compatible y SILENTE.

    Dos problemas de la función de statsmodels que se resuelven
    acá:
      · `verbose` fue deprecado y luego eliminado según la
        versión, así que se prueba primero la firma nueva.
      · En statsmodels 0.14 la firma nueva IMPRIME el detalle de
        cada test por stdout. Con ~36 pares × hasta 14 lags eso
        inunda el log y tapa el reporte. Se redirige stdout a un
        buffer descartable: los p-values se leen del dict que
        devuelve, no de lo impreso.

    Args:
        data: DataFrame con columnas ["y", "x"].
        max_lag: Lag máximo.

    Returns:
        Dict de resultados de statsmodels.
    """
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

    CAMBIO I5-1 (Job 6): reemplaza el criterio anterior de
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


def granger_par(
    y: pd.Series,
    x: pd.Series,
    max_lag: int,
    nombre_x: str,
    nombre_y: str,
    pista: str,
) -> tuple[list[dict], dict]:
    """
    Test de Granger: ¿x ayuda a predecir y más allá de y_{t-k}?

    ESTADÍSTICO DE DECISIÓN: el test F (ssr_ftest), NO el chi².
    El chi² es asintótico y sobre-rechaza de forma severa con
    muestras chicas: verificado sobre 1200 tests de ruido puro
    independiente con n=30 y maxlag=6 (el tamaño de la Pista B),
    el chi² rechaza el 25.4% de las veces bajo la hipótesis nula
    —debería ser 5%— y 157 de esos tests sobreviven incluso la
    corrección BH. El test F, con corrección de muestra finita,
    rechaza el 6.0% y ninguno sobrevive BH. Corregir por FDR no
    arregla p-values inválidos de origen: por eso la decisión se
    toma con F. El p del chi² se guarda igual, como columna de
    trazabilidad y para comparar con la Fase 6 original.

    Un p bajo NO implica causalidad económica.

    Args:
        y: Target transformado.
        x: Feature transformada.
        max_lag: Lag máximo.
        nombre_x: Nombre de la feature.
        nombre_y: Nombre del target.
        pista: "A" o "B".

    CAMBIO I5-1/I5-6: devuelve (filas_por_rezago, fila_familia).
    filas_por_rezago es la tabla de apoyo (1 fila/lag, NO entra
    a BH). fila_familia es 1 sola fila, en el rezago elegido por
    BIC (elegir_rezago_bic) — esa es la que entra a BH. Antes de
    elegir el rezago se verifica estacionariedad de x e y
    (I5-f).

    Returns:
        (filas_por_rezago, fila_familia).
    """
    data = pd.concat(
        [y.rename("y"), x.rename("x")], axis=1
    ).dropna()
    n_min = max_lag + 10
    fila_base = {
        "pista": pista,
        "serie_x": nombre_x,
        "bloque": (
            "rwa" if nombre_x.startswith("rwa_") else "onchain"
        ),
        "target": nombre_y,
        "n": len(data),
    }
    if len(data) < n_min:
        filas = [
            {
                **fila_base, "lag": k,
                "pval_f": None, "pval_chi2": None,
                "nota": f"n={len(data)}<{n_min} insuficiente",
            }
            for k in range(1, max_lag + 1)
        ]
        return filas, {
            **fila_base, "lag": None, "pval_f": None,
            "nota": f"n={len(data)}<{n_min} insuficiente",
        }

    verificar_estacionariedad(y, f"{pista}/{nombre_y}/y")
    verificar_estacionariedad(x, f"{pista}/{nombre_x}/x")

    try:
        resultado = _granger_raw(data[["y", "x"]], max_lag)
    except Exception as exc:
        filas = [
            {
                **fila_base, "lag": k,
                "pval_f": None, "pval_chi2": None,
                "nota": f"error: {str(exc)[:60]}",
            }
            for k in range(1, max_lag + 1)
        ]
        return filas, {
            **fila_base, "lag": None, "pval_f": None,
            "nota": f"error: {str(exc)[:60]}",
        }
    filas: list[dict] = []
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
            "nota": (
                NOTA_RWA
                if fila_base["bloque"] == "rwa" else ""
            ),
        })

    # CAMBIO I5-1: UN rezago por BIC para la familia de BH.
    lag_elegido = elegir_rezago_bic(y, x, max_lag)
    fila_elegida = next(
        (f for f in filas if f["lag"] == lag_elegido), None,
    )
    if fila_elegida is None:
        fila_elegida = {
            **fila_base, "lag": lag_elegido, "pval_f": None,
            "nota": "rezago elegido sin estadístico",
        }
    return filas, fila_elegida


def series_pista_a(
    df: pd.DataFrame,
) -> tuple[dict[str, pd.Series], pd.Series]:
    """
    Series transformadas de la Pista A (diaria).

    Args:
        df: Dataset diario recortado.

    Returns:
        (features log-dif, ΔCISS).
    """
    x: dict[str, pd.Series] = {}
    for col in COLS_FEATURES:
        if col in df.columns:
            x[col] = _log_diff(df[col]).dropna()
    y = df[COL_CISS].diff().dropna()
    return x, y


def series_pista_b(
    df: pd.DataFrame,
    df_lrg: pd.DataFrame,
) -> tuple[dict[str, pd.Series], pd.Series]:
    """
    Series transformadas de la Pista B (semanal).

    Flujos → sum semanal; stocks (adract, USDY, RWA) → last().

    Args:
        df: Dataset diario recortado.
        df_lrg: Dataset de series largas.

    Returns:
        (features log-dif semanales, STLFSI nivel semanal).
    """
    x: dict[str, pd.Series] = {}
    for col in COLS_FEATURES:
        if col not in df.columns:
            continue
        if col in COLS_FLUJO:
            serie_sem = df[col].resample("W").sum()
        elif col in COLS_STOCK_SEMANAL:
            serie_sem = df[col].resample("W").last()
        else:
            serie_sem = df[col].resample("W").mean()
        x[col] = _log_diff(serie_sem).dropna()

    if COL_STLFSI in df_lrg.columns:
        fuente = df_lrg[COL_STLFSI]
    elif COL_STLFSI in df.columns:
        fuente = df[COL_STLFSI]
    else:
        return x, pd.Series(dtype=float)
    # CAMBIO I4-3: recorta a la ventana real de `df`, no a un
    # FECHA_CORTE que ahora puede ser None.
    fuente = fuente.loc[fuente.index <= df.index.max()]
    return x, fuente.resample("W").last().dropna()


def tabla_granger(
    x_a: dict[str, pd.Series],
    y_a: pd.Series,
    x_b: dict[str, pd.Series],
    y_b: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Corre Granger para todos los pares de ambas pistas.

    CAMBIO I5-1: devuelve (df_por_rezago, df_familia) en vez de
    una sola tabla — ver granger_par.

    Args:
        x_a: Features diarias.
        y_a: ΔCISS.
        x_b: Features semanales.
        y_b: STLFSI nivel semanal.

    Returns:
        (df_por_rezago, df_familia).
    """
    filas: list[dict] = []
    familia: list[dict] = []
    for col, serie in x_a.items():
        _f, _fam = granger_par(
            y_a, serie, K_GRANGER_D, col, "ΔCISS", "A"
        )
        filas.extend(_f)
        familia.append(_fam)
    if not y_b.empty:
        for col, serie in x_b.items():
            _f, _fam = granger_par(
                y_b, serie, K_GRANGER_S, col, "STLFSI", "B"
            )
            filas.extend(_f)
            familia.append(_fam)
    elif DEBUG:
        print("[DEBUG] Pista B sin target: Granger omitido")
    return pd.DataFrame(filas), pd.DataFrame(familia)


def aplicar_bh_global(
    df: pd.DataFrame,
    col_p: str = "pval_f",
    alfa: float = ALFA_BH,
) -> pd.DataFrame:
    """
    Corrección Benjamini-Hochberg sobre el conjunto COMPLETO.

    Se aplica una sola vez sobre todos los p-values de la tabla
    (ambas pistas, todos los lags, todas las features). Un
    ajuste por subconjunto sobreestimaría la significancia.
    Las filas sin p-value quedan como no significativas.

    Corrige sobre pval_f (test F), no sobre el chi²: ver
    granger_par para por qué el chi² no es admisible con este n.

    Args:
        df: Tabla de resultados de Granger.
        col_p: Columna de p-values.
        alfa: Nivel de FDR.

    Returns:
        DataFrame con bh_q, bh_rechaza y ambito_bh.
    """
    df = df.copy().reset_index(drop=True)
    df["bh_q"] = np.nan
    df["bh_rechaza"] = False
    # CAMBIO I5-1: ya no incluye lags — 1 rezago por BIC.
    df["ambito_bh"] = "global: 2 pistas × features (1 rezago por BIC)"
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
    """
    Para cada par, el lag de menor p-value (el caso más
    favorable), con su q global arrastrado.

    Args:
        df_granger: Tabla de Granger con BH global aplicado.

    Returns:
        DataFrame con una fila por par.
    """
    validos = df_granger[df_granger["pval_f"].notna()]
    if validos.empty:
        return pd.DataFrame()
    idx = validos.groupby(
        ["pista", "serie_x", "target"]
    )["pval_f"].idxmin()
    return df_granger.loc[idx].copy()


def tabla_confirmacion(
    df_granger: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extrae los pares bajo confirmación específica.

    Son los indicios en dirección teórica de 14_relaciones_rwa
    (treasurybills y commodities → STLFSI). El objetivo es
    confirmarlos o refutarlos con Granger + FDR global.

    Args:
        df_granger: Tabla de Granger con BH global.

    Returns:
        DataFrame con una fila por par confirmado/refutado.
    """
    filas: list[dict] = []
    for serie_x, pista in CONFIRMACIONES:
        sub = df_granger[
            (df_granger["serie_x"] == serie_x)
            & (df_granger["pista"] == pista)
            & (df_granger["pval_f"].notna())
        ]
        if sub.empty:
            filas.append({
                "serie_x": serie_x,
                "pista": pista,
                "lag_mejor": None,
                "pval_min_f": None,
                "pval_min_chi2": None,
                "bh_q": None,
                "sobrevive_fdr_global": False,
                "veredicto": "sin resultados de Granger",
            })
            continue
        fila = sub.loc[sub["pval_f"].idxmin()]
        sobrevive = bool(fila["bh_rechaza"])
        filas.append({
            "serie_x": serie_x,
            "pista": pista,
            "target": str(fila["target"]),
            "lag_mejor": int(fila["lag"]),
            "pval_min_f": float(fila["pval_f"]),
            "pval_min_chi2": float(fila["pval_chi2"]),
            "bh_q": (
                float(fila["bh_q"])
                if pd.notna(fila["bh_q"]) else None
            ),
            "n": int(fila["n"]),
            "sobrevive_fdr_global": sobrevive,
            "veredicto": (
                "CONFIRMADO: resiste Granger + FDR global"
                if sobrevive else
                "REFUTADO: el indicio no resiste Granger con "
                "FDR global; era compatible con azar"
            ),
        })
    return pd.DataFrame(filas)


def comparar_backtesting(
    df_met: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Compara los modelos con features contra los baselines.

    Para cada (pista, grupo de features) toma el mejor baseline
    sin features (persistencia / media móvil / AR) y calcula
    cuánto mejora —o empeora— cada modelo con features.

    Args:
        df_met: Métricas de 15_modelado_final, o None.

    Returns:
        DataFrame con la comparación; vacío si no hay métricas.
    """
    if df_met is None or df_met.empty:
        return pd.DataFrame()
    es_baseline = df_met["modelo"].str.startswith(
        ("persistencia", "ma", "ar")
    )
    filas: list[dict] = []
    for (pista, grupo), sub in df_met.groupby(
        ["pista", "grupo_features"], sort=False
    ):
        base = sub[
            es_baseline.reindex(sub.index, fill_value=False)
        ].dropna(subset=["rmse"])
        if base.empty:
            continue
        idx_base = base["rmse"].idxmin()
        rmse_base = float(base.loc[idx_base, "rmse"])
        nombre_base = str(base.loc[idx_base, "modelo"])
        con_feat = sub[
            sub["modelo"].isin(["ridge", "lasso"])
        ].dropna(subset=["rmse"])
        for _, fila in con_feat.iterrows():
            rmse = float(fila["rmse"])
            mejora_pct = (
                100.0 * (rmse_base - rmse) / rmse_base
                if rmse_base > 0 else float("nan")
            )
            filas.append({
                "pista": pista,
                "grupo_features": grupo,
                "modelo": str(fila["modelo"]),
                "rmse": rmse,
                "mae": fila.get("mae"),
                "r2_oos": fila.get("r2_oos"),
                "baseline": nombre_base,
                "rmse_baseline": rmse_base,
                "mejora_pct_vs_baseline": round(mejora_pct, 2),
                "mejora": bool(rmse < rmse_base),
                "n_test": fila.get("n_test"),
            })
    return pd.DataFrame(filas)


def metricas_clasificacion(
    y_true_bin: np.ndarray,
    y_pred_bin: np.ndarray,
) -> dict:
    """
    Matriz de confusión y métricas de clasificación binaria.

    Args:
        y_true_bin: Verdad binaria (1 = evento de estrés).
        y_pred_bin: Predicción binaria.

    Returns:
        Dict con tp, fp, fn, tn, precision, recall y f1.
    """
    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * prec * rec / (prec + rec)
        if (prec + rec) > 0 else 0.0
    )
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


def evaluar_eventos(
    df_pred: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clasificación de eventos de estrés (target > percentil 80).

    El umbral se calcula sobre el target real del test de cada
    corrida. Se reporta el número de eventos: con pocos, las
    métricas son inestables y así queda registrado.

    Args:
        df_pred: Predicciones de 15_modelado_final.

    Returns:
        DataFrame con una fila por (pista, grupo, modelo).
    """
    filas: list[dict] = []
    modelos = [
        c for c in df_pred.columns
        if c not in (
            "fecha", "pista", "grupo_features", "target",
            "target_actual",
        )
    ]
    for (pista, grupo), sub in df_pred.groupby(
        ["pista", "grupo_features"], sort=False
    ):
        reales = pd.to_numeric(
            sub["target_actual"], errors="coerce"
        ).dropna()
        if reales.empty:
            continue
        umbral = float(
            np.nanpercentile(reales.to_numpy(), PERCENTIL_EVENTO)
        )
        y_bin = (reales.to_numpy() > umbral).astype(int)
        n_eventos = int(y_bin.sum())
        for modelo in modelos:
            pred = pd.to_numeric(
                sub[modelo], errors="coerce"
            ).reindex(reales.index)
            if pred.isna().all():
                continue
            pred_bin = (
                pred.fillna(-np.inf).to_numpy() > umbral
            ).astype(int)
            filas.append({
                "pista": pista,
                "grupo_features": grupo,
                "modelo": modelo,
                "percentil": PERCENTIL_EVENTO,
                "umbral": round(umbral, 8),
                "n_test": int(len(y_bin)),
                "n_eventos": n_eventos,
                **metricas_clasificacion(y_bin, pred_bin),
                "aviso": (
                    f"solo {n_eventos} eventos: métricas "
                    "inestables"
                    if n_eventos < N_MIN_EVENTOS else ""
                ),
            })
    return pd.DataFrame(filas)


def evaluar_corroboracion(
    n_sig_bh: int,
    n_tests: int,
    df_bt: pd.DataFrame,
    df_coefs: pd.DataFrame | None,
    df_conf: pd.DataFrame,
) -> dict:
    """
    Decide si una mejora sobre el baseline constituye evidencia.

    Una mejora de RMSE en un solo modelo, en una sola pista y
    con margen chico NO es señal: con 56 días de test, esa
    diferencia entra dentro de la variabilidad muestral, y un
    modelo con más features siempre puede ajustar ruido. Para
    que cuente como evidencia de poder predictivo debe estar
    CORROBORADA por criterios independientes:

      C1 selección  — Lasso retiene al menos una feature. Si la
        regularización las descarta todas, el propio
        procedimiento de selección concluye que no hay señal.
      C2 Granger    — al menos un test sobrevive el FDR global.
        Es el control de precedencia, independiente del ajuste.
      C3 replicación— la mejora aparece en AMBAS pistas para el
        mismo modelo. Si se invierte en la otra, es específica
        de una muestra, no una relación.
      C4 margen     — la mejora supera UMBRAL_MEJORA_MATERIAL.
        Un margen menor no se distingue del ruido.

    Solo se afirma evidencia si se cumplen los cuatro. Si hay
    mejora pero falla alguno, la conclusión es sobreajuste de
    ruido, no señal.

    Args:
        n_sig_bh: Tests de Granger que sobreviven el FDR global.
        n_tests: Total de tests de Granger corridos.
        df_bt: Comparación de backtesting contra baseline.
        df_coefs: Coeficientes retenidos por Lasso (o None).
        df_conf: Tabla de confirmación de los indicios refugio.

    Returns:
        Dict con los criterios, la mejor mejora observada y el
        diagnóstico agregado.
    """
    mejoras = (
        df_bt[df_bt["mejora"]].copy()
        if not df_bt.empty else pd.DataFrame()
    )
    hay_mejora = not mejoras.empty

    mejor: dict = {}
    if hay_mejora:
        idx = mejoras["mejora_pct_vs_baseline"].idxmax()
        fila = mejoras.loc[idx]
        mejor = {
            "modelo": str(fila["modelo"]),
            "pista": str(fila["pista"]),
            "grupo": str(fila["grupo_features"]),
            "margen_pct": float(
                fila["mejora_pct_vs_baseline"]
            ),
        }
        # Comportamiento del mismo modelo en la otra pista
        otra = df_bt[
            (df_bt["modelo"] == mejor["modelo"])
            & (df_bt["pista"] != mejor["pista"])
        ]
        if not otra.empty:
            mejor["otra_pista"] = str(otra.iloc[0]["pista"])
            mejor["otra_pista_margen_pct"] = float(
                otra["mejora_pct_vs_baseline"].max()
            )
            mejor["otra_pista_mejora"] = bool(
                otra["mejora"].any()
            )

    # C1 — ¿Lasso retuvo alguna feature?
    c_lasso = df_coefs is not None and not df_coefs.empty
    # C2 — ¿Granger sobrevive el FDR global?
    c_granger = n_sig_bh > 0
    # C3 — ¿la mejora se replica en ambas pistas?
    c_replica = False
    if not mejoras.empty:
        por_modelo = mejoras.groupby("modelo")["pista"].nunique()
        c_replica = bool((por_modelo >= 2).any())
    # C4 — ¿el margen es material?
    c_margen = bool(
        hay_mejora
        and mejor.get("margen_pct", 0.0)
        >= UMBRAL_MEJORA_MATERIAL
    )

    conf_ok = (
        bool(df_conf["sobrevive_fdr_global"].any())
        if not df_conf.empty else False
    )

    return {
        "hay_mejora": hay_mejora,
        "mejor": mejor,
        "c1_lasso_retiene": c_lasso,
        "c2_granger_fdr": c_granger,
        "c3_replica_pistas": c_replica,
        "c4_margen_material": c_margen,
        "corroborada": bool(
            hay_mejora
            and c_lasso and c_granger
            and c_replica and c_margen
        ),
        "confirmacion_refugio": conf_ok,
        "n_sig_bh": int(n_sig_bh),
        "n_tests": int(n_tests),
    }


def reporte_consolidado(
    df_granger: pd.DataFrame,
    df_bt: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reporte final por pista: mejor modelo, mejora sobre
    baseline y veredicto de Granger con FDR global.

    Args:
        df_granger: Tabla de Granger con BH global.
        df_bt: Comparación de backtesting.

    Returns:
        DataFrame con una fila por pista.
    """
    filas: list[dict] = []
    for pista, target in [("A", "ΔCISS"), ("B", "STLFSI")]:
        sub_g = (
            df_granger[df_granger["pista"] == pista]
            if not df_granger.empty else pd.DataFrame()
        )
        n_tests = (
            int(sub_g["pval_f"].notna().sum())
            if not sub_g.empty else 0
        )
        sig = (
            sub_g[sub_g["bh_rechaza"]]
            if not sub_g.empty else pd.DataFrame()
        )
        pares_sig = (
            sorted(set(sig["serie_x"]))
            if not sig.empty else []
        )

        sub_b = (
            df_bt[df_bt["pista"] == pista]
            if not df_bt.empty else pd.DataFrame()
        )
        mejor_modelo = "n/a"
        mejor_rmse = None
        mejor_r2 = None
        mejora_alguna = False
        if not sub_b.empty:
            idx = sub_b["rmse"].idxmin()
            mejor_modelo = (
                f"{sub_b.loc[idx, 'modelo']}"
                f" ({sub_b.loc[idx, 'grupo_features']})"
            )
            mejor_rmse = float(sub_b.loc[idx, "rmse"])
            mejor_r2 = sub_b.loc[idx, "r2_oos"]
            mejora_alguna = bool(sub_b["mejora"].any())

        if pares_sig:
            conclusion = (
                f"{len(sig)} test(s) Granger sobreviven el FDR "
                f"global: {pares_sig}. Interpretar con cautela: "
                "n corto y lag post-hoc sesgan los p-values."
            )
        elif n_tests == 0:
            conclusion = "Sin tests de Granger disponibles."
        else:
            conclusion = (
                f"Ninguno de los {n_tests} tests de Granger "
                "sobrevive el FDR global: no hay precedencia "
                "estadísticamente sostenible."
            )
        if not sub_b.empty and not mejora_alguna:
            conclusion += (
                " Ningún modelo con features supera al baseline "
                "sin features."
            )

        filas.append({
            "pista": pista,
            "target": target,
            "n_tests_granger": n_tests,
            "n_granger_sig_fdr_global": (
                int(len(sig)) if not sig.empty else 0
            ),
            "features_sig": str(pares_sig),
            "mejor_modelo_rmse": mejor_modelo,
            "mejor_rmse": mejor_rmse,
            "mejor_r2_oos": mejor_r2,
            "mejora_sobre_baseline": mejora_alguna,
            "conclusion": conclusion,
        })
    return pd.DataFrame(filas)


"""# 5. Proceso"""

# CAMBIO B1-4 (Bloque 1): hash de las entradas vs. manifest.
verificar_entradas_snapshot([ARCHIVO_EXT, ARCHIVO_LARGAS], "Job 8")

print(
    "\n[LÍNEA DE BASE] Validación final (on-chain + RWA).\n"
    "[LÍNEA DE BASE] Punto cero: se documenta el poder "
    "predictivo real, no se vende señal inexistente."
)

# ── Carga de datos ──────────────────────────────────────────
# CAMBIO E2E-08-01: todas las entradas obligatorias se comprueban
# antes de Granger; falta de archivo != resultado estadistico nulo.
for _ruta in (ARCHIVO_EXT, ARCHIVO_LARGAS):
    if not os.path.isfile(_ruta):
        raise FileNotFoundError(
            f"[DEBUG][ERROR] Job 8: falta {_ruta}. "
            "Debe generarlo el Job 3."
        )
try:
    df = cargar_dataset_ventana()
    df_lrg = cargar_series(ARCHIVO_LARGAS)
except Exception as _exc:
    raise RuntimeError(
        f"[DEBUG][ERROR] Job 8: error leyendo series del Job 3: {_exc}"
    ) from _exc
validar_entrada_series_job8(df, ARCHIVO_EXT, "Job 3")
validar_entrada_series_job8(df_lrg, ARCHIVO_LARGAS, "Job 3")
if COL_CISS not in df.columns:
    raise RuntimeError(
        f"[DEBUG][ERROR] Job 8: falta {COL_CISS} en {ARCHIVO_EXT}."
    )
if COL_STLFSI not in df_lrg.columns and COL_STLFSI not in df.columns:
    raise RuntimeError(
        f"[DEBUG][ERROR] Job 8: falta {COL_STLFSI} en ambas "
        "fuentes del Job 3."
    )
if not any(c in df.columns for c in COLS_FEATURES):
    raise RuntimeError(
        f"[DEBUG][ERROR] Job 8: faltan todas las features "
        f"configuradas en {ARCHIVO_EXT}."
    )

# Salidas del Job 7: metricas/predicciones obligatorias; la tabla
# de Lasso puede tener cero filas, pero debe existir con encabezados.
df_met: pd.DataFrame = leer_csv_job8(
    ARCHIVO_MET,
    ["pista", "grupo_features", "modelo", "rmse", "mae", "r2_oos", "n_test"],
    job_productor="7",
)
df_coefs: pd.DataFrame = leer_csv_job8(
    ARCHIVO_COEF,
    ["pista", "grupo_features", "feature", "coef", "bloque"],
    job_productor="7", permitir_vacio=True,
)
_df_pred: pd.DataFrame = leer_csv_job8(
    ARCHIVO_PRED,
    ["fecha", "pista", "grupo_features", "target", "target_actual",
     "persistencia", "ridge", "lasso"],
    job_productor="7", parse_dates=["fecha"],
)
if DEBUG:
    print(
        "[DEBUG] Job 8: entradas validadas "
        "(series Job 3, metricas/predicciones/coeficientes Job 7)"
    )

if DEBUG:
    print(
        f"[DEBUG] ventana: {df.index.min().date()} → "
        f"{df.index.max().date()} ({len(df)} días)"
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
        f"[DEBUG][ALERTA] Job 8: la ventana "
        f"{VENTANA_INICIO_REPORTE} → {VENTANA_FIN_REPORTE} no es "
        "2026; revisar series_diarias_ext."
    )
print(
    f"[DEBUG] Job 8: ventana declarada {VENTANA_INICIO_REPORTE} → "
    f"{VENTANA_FIN_REPORTE} ({NOTA_VENTANA_REPORTE})"
)

# ── Parte 1: Granger con FDR GLOBAL ─────────────────────────
print("\n[DEBUG] ── Parte 1: Granger + FDR global ──")
df_granger: pd.DataFrame = pd.DataFrame()
df_granger_rezago: pd.DataFrame = pd.DataFrame()
df_granger_opt: pd.DataFrame = pd.DataFrame()
n_tests_global: int = 0
n_sig_crudo: int = 0
n_sig_bh: int = 0
try:
    _x_a, _y_a = series_pista_a(df)
    _x_b, _y_b = series_pista_b(df, df_lrg)
    if len(_y_b) < N_MIN_SEMANAS:
        print(
            f"[LÍNEA DE BASE] AVISO: Pista B con {len(_y_b)} "
            f"semanas (< {N_MIN_SEMANAS}): potencia muy baja."
        )
    # CAMBIO I5-1: tabla_granger devuelve (por_rezago, familia).
    df_granger_rezago, df_granger = tabla_granger(
        _x_a, _y_a, _x_b, _y_b
    )
    # CAMBIO E2E-08-01: una familia vacia tiene estructura, no p-valores.
    # Se conserva intacto aplicar_bh_global() y el universo de tests.
    if df_granger.empty:
        df_granger = pd.DataFrame(columns=[
            "pista", "serie_x", "bloque", "target", "n", "lag",
            "pval_f", "pval_chi2", "nota",
        ])
    df_granger = aplicar_bh_global(df_granger)

    _p = pd.to_numeric(df_granger["pval_f"], errors="coerce")
    n_tests_global = int(_p.notna().sum())
    n_sig_crudo = int((_p < ALFA_BH).sum())
    n_sig_bh = int(
        df_granger["bh_rechaza"].fillna(False).astype(bool).sum()
    )
    print(
        f"\n[LÍNEA DE BASE] Granger (test F, ssr_ftest) — "
        f"familia: {n_tests_global} tests | p<{ALFA_BH} sin "
        f"corregir: {n_sig_crudo} | sobreviven FDR GLOBAL: "
        f"{n_sig_bh}"
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

    # CAMBIO E2E-08-01: guardar ambos CSV al final con el resto
    # de reportes, evitando publicar resultados a mitad del Job.

    if n_sig_bh == 0:
        print(
            "  → Ningún par Granger-causa al target tras la "
            "corrección global."
        )
    else:
        _cols_sig = [
            "pista", "serie_x", "bloque", "target",
            "lag", "pval_f", "pval_chi2", "bh_q", "n",
        ]
        print(
            df_granger[df_granger["bh_rechaza"]][_cols_sig]
            .to_string(index=False)
        )
except Exception as _exc_gr:
    raise RuntimeError(
        f"[DEBUG][ERROR] Job 8: fallo Granger: {_exc_gr}"
    ) from _exc_gr

# ── Parte 2: confirmación treasurybills / commodities ───────
print(
    "\n[DEBUG] ── Parte 2: confirmación de indicios "
    "(refugio) ──"
)
df_conf: pd.DataFrame = pd.DataFrame()
try:
    if df_granger.empty:
        print("[DEBUG] sin tabla de Granger: se omite")
    else:
        df_conf = tabla_confirmacion(df_granger)
        print(
            "[LÍNEA DE BASE] Indicios de 14_relaciones_rwa "
            "(dirección teórica de activo refugio), puestos a "
            "prueba con Granger + FDR global:"
        )
        for _, _f in df_conf.iterrows():
            print(
                f"  - {_f['serie_x']} → "
                f"{_f.get('target', 'STLFSI')}: "
                f"lag={_f['lag_mejor']}, "
                f"p_F={_f['pval_min_f']}, "
                f"q_global={_f['bh_q']} → {_f['veredicto']}"
            )
except Exception as _exc_cf:
    raise RuntimeError(
        f"[DEBUG][ERROR] Job 8: fallo confirmacion: {_exc_cf}"
    ) from _exc_cf

# ── Parte 3: backtesting ────────────────────────────────────
print("\n[DEBUG] ── Parte 3: backtesting ──")
df_bt: pd.DataFrame = pd.DataFrame()
# CAMBIO E2E-08-01: cero filas de coeficientes = Lasso no retuvo
# features; un archivo ausente o corrupto ya aborto al cargar.
print(f"[LÍNEA DE BASE] Lasso retuvo {len(df_coefs)} feature(s):")
if not df_coefs.empty:
    print(df_coefs.to_string(index=False))

try:
    if DEBUG:
        print(f"[DEBUG] métricas cargadas: {df_met.shape}")
    df_bt = comparar_backtesting(df_met)
    if df_bt.empty:
        print(
            "[DEBUG] sin comparación posible "
            "(faltan baselines o modelos con features)"
        )
    else:
        _cols_bt = [
            "pista", "grupo_features", "modelo", "rmse",
            "r2_oos", "baseline", "rmse_baseline",
            "mejora_pct_vs_baseline", "mejora",
        ]
        print(
            "\n[LÍNEA DE BASE] ¿Mejoran los modelos con "
            "features sobre el baseline sin features?"
        )
        print(df_bt[_cols_bt].to_string(index=False))
        if not df_bt["mejora"].any():
            print(
                "  → NINGÚN modelo con features (on-chain, "
                "RWA ni ambas) supera al baseline."
            )
except Exception as _exc_bt:
    raise RuntimeError(
        f"[DEBUG][ERROR] Job 8: fallo backtesting: {_exc_bt}"
    ) from _exc_bt

# ── Parte 4: eventos de estrés ──────────────────────────────
print("\n[DEBUG] ── Parte 4: eventos de estrés ──")
df_eventos: pd.DataFrame = pd.DataFrame()
try:
    df_eventos = evaluar_eventos(_df_pred)
    if df_eventos.empty:
        print("[DEBUG] sin eventos evaluables")
    else:
        _cols_ev = [
            "pista", "grupo_features", "modelo",
            "n_test", "n_eventos", "tp", "fp", "fn", "tn",
            "precision", "recall", "f1",
        ]
        print(
            f"\n[LÍNEA DE BASE] Eventos de estrés "
            f"(target > p{PERCENTIL_EVENTO}):"
        )
        print(df_eventos[_cols_ev].to_string(index=False))
        _n_ev = sorted(set(df_eventos["n_eventos"]))
        print(
            f"[LÍNEA DE BASE] nº de eventos por corrida: "
            f"{_n_ev} — con tan pocos, precision/recall/F1 "
            "son inestables y no soportan conclusiones."
        )
except Exception as _exc_ev:
    raise RuntimeError(
        f"[DEBUG][ERROR] Job 8: fallo eventos: {_exc_ev}"
    ) from _exc_ev

# ── Parte 5: reporte consolidado y veredicto ────────────────
print("\n[DEBUG] ── Parte 5: reporte final ──")
df_reporte: pd.DataFrame = pd.DataFrame()
try:
    df_reporte = reporte_consolidado(df_granger, df_bt)
    df_reporte["ventana_inicio"] = VENTANA_INICIO_REPORTE  # CAMBIO B2-4
    df_reporte["ventana_fin"] = VENTANA_FIN_REPORTE
    df_reporte["ventana_nota"] = NOTA_VENTANA_REPORTE
    print("\n" + "─" * 64)
    print("REPORTE FINAL — LÍNEA DE BASE (on-chain + RWA)")
    print("─" * 64)
    _nota_ventana = (
        f"recorte en {FECHA_CORTE}"
        if FECHA_CORTE is not None else "sin recorte"
    )
    print(
        f"\nVentana: {df.index.min().date()} → "
        f"{df.index.max().date()} ({_nota_ventana}; "
        f"{NOTA_VENTANA_REPORTE})"  # CAMBIO B2-4
    )
    print(
        f"Granger (test F): {n_tests_global} tests | "
        f"p<{ALFA_BH} crudo: {n_sig_crudo} | "
        f"sobreviven FDR GLOBAL: {n_sig_bh}"
    )
    for _, _fila in df_reporte.iterrows():
        print(
            f"\n  Pista {_fila['pista']} ({_fila['target']}):"
        )
        print(
            f"    Mejor modelo con features: "
            f"{_fila['mejor_modelo_rmse']} | "
            f"RMSE={_fila['mejor_rmse']} | "
            f"R² OOS={_fila['mejor_r2_oos']}"
        )
        print(
            f"    ¿Mejora sobre baseline?: "
            f"{_fila['mejora_sobre_baseline']}"
        )
        print(
            f"    Granger sig. (FDR global): "
            f"{_fila['n_granger_sig_fdr_global']}"
        )
        print(f"    {_fila['conclusion']}")

    # ── VEREDICTO ───────────────────────────────────────────
    # Una mejora de RMSE NO alcanza por sí sola: debe estar
    # corroborada por selección (Lasso), control de precedencia
    # (Granger + FDR global), replicación entre pistas y margen
    # material. Ver evaluar_corroboracion.
    _ev = evaluar_corroboracion(
        n_sig_bh, n_tests_global, df_bt, df_coefs, df_conf
    )

    print("\n" + "─" * 64)
    print("CORROBORACIÓN DE LA EVIDENCIA")
    print("─" * 64)
    _criterios = [
        (
            "C1  Lasso retiene alguna feature",
            _ev["c1_lasso_retiene"],
            (
                f"{len(df_coefs)} retenidas"
                if df_coefs is not None and not df_coefs.empty
                else "Lasso descartó TODAS las features"
            ),
        ),
        (
            "C2  Granger sobrevive FDR global",
            _ev["c2_granger_fdr"],
            f"{n_sig_bh}/{n_tests_global} tests",
        ),
        (
            "C3  La mejora se replica en ambas pistas",
            _ev["c3_replica_pistas"],
            (
                "el mismo modelo mejora en A y en B"
                if _ev["c3_replica_pistas"]
                else "la mejora aparece en una sola pista"
            ),
        ),
        (
            "C4  Margen material "
            f"(>= {UMBRAL_MEJORA_MATERIAL}%)",
            _ev["c4_margen_material"],
            (
                f"mejor margen: "
                f"{_ev['mejor'].get('margen_pct', 0.0):.2f}%"
                if _ev["hay_mejora"] else "sin mejoras"
            ),
        ),
    ]
    for _nombre_c, _ok_c, _detalle_c in _criterios:
        print(
            f"  [{'SÍ' if _ok_c else 'NO'}] {_nombre_c}: "
            f"{_detalle_c}"
        )

    print("\n" + "─" * 64)
    if _ev["corroborada"]:
        _m = _ev["mejor"]
        print(
            "VEREDICTO: hay evidencia de aporte predictivo que "
            "supera el control aplicado."
        )
        print(
            f"La mejora ({_m['modelo']}, Pista {_m['pista']}, "
            f"grupo {_m['grupo']}, {_m['margen_pct']:.2f}%) está "
            "corroborada por los cuatro criterios: Lasso la "
            "retiene, Granger la respalda tras FDR global, se "
            "replica entre pistas y el margen es material."
        )
        print(
            "CAUTELA: n corto, ventana única y lag post-hoc en "
            "fases previas sesgan A FAVOR de encontrar señal. "
            "No se afirma causalidad económica."
        )
    elif _ev["hay_mejora"]:
        # Hay mejora pero no corrobora: sobreajuste de ruido.
        _m = _ev["mejor"]
        _fallos: list[str] = []
        if not _ev["c1_lasso_retiene"]:
            _fallos.append("no es retenida por Lasso")
        if not _ev["c2_granger_fdr"]:
            _fallos.append(
                "no supera el control de Granger "
                f"({n_sig_bh}/{n_tests_global} tras FDR global)"
            )
        if not _ev["c3_replica_pistas"]:
            _otra = _m.get("otra_pista")
            if _otra is not None:
                _pct_otra = _m.get(
                    "otra_pista_margen_pct", float("nan")
                )
                _fallos.append(
                    f"se invierte en la Pista {_otra} "
                    f"({_pct_otra:.2f}%)"
                )
            else:
                _fallos.append(
                    "no se replica en la otra pista"
                )
        if not _ev["c4_margen_material"]:
            _fallos.append(
                f"el margen ({_m['margen_pct']:.2f}%) está por "
                f"debajo de {UMBRAL_MEJORA_MATERIAL}% y no se "
                "distingue del ruido"
            )
        print(
            "VEREDICTO: no hay evidencia de poder predictivo "
            "robusto."
        )
        print(
            f"La única mejora ({_m['modelo']}, Pista "
            f"{_m['pista']}, grupo {_m['grupo']}, margen "
            f"{_m['margen_pct']:.2f}%) " + ", ".join(_fallos)
            + ", por lo que se atribuye a SOBREAJUSTE DE RUIDO "
            "y no a señal."
        )
        print("Punto 0 confirmado.")
    else:
        print(
            "VEREDICTO: ninguna señal —on-chain ni RWA— aporta "
            "poder predictivo sobre el estrés financiero tras "
            "el control riguroso."
        )
        print(
            f"De {n_tests_global} tests de Granger, "
            f"{n_sig_crudo} dan p<{ALFA_BH} crudo y {n_sig_bh} "
            "sobreviven el FDR global; y ningún modelo con "
            "features supera al baseline del target solo."
        )
        print("Punto 0 confirmado.")

    if not df_conf.empty and not _ev["confirmacion_refugio"]:
        print(
            "Los indicios de refugio (treasurybills, "
            "commodities) quedan REFUTADOS: no resisten Granger "
            "con corrección global."
        )
    print(
        "Este resultado acota con rigor lo que las señales "
        "on-chain y RWA pueden y no pueden decir sobre el "
        "estrés financiero en esta ventana: es el hallazgo de "
        "la línea de base, y es científicamente válido."
    )
    print(
        f"\n{NOTA_RWA}\n"
        "Limitaciones: ventana única recortada, ~30 semanas en "
        "Pista B, pocos eventos de estrés en el test y "
        "selección post-hoc en fases previas."
    )
    print("─" * 64)
except Exception as _exc_rep:
    raise RuntimeError(
        f"[DEBUG][ERROR] Job 8: fallo reporte: {_exc_rep}"
    ) from _exc_rep

"""# 6. Save"""

if SAVE_DATA:
    os.makedirs(DIR_REPORTES, exist_ok=True)
    # CAMBIO E2E-08-01: mismo CSV para tablas con registros;
    # los resultados vacios se publican con encabezados y 0 filas.
    # Un resultado vacio no deja vigente el CSV de la corrida anterior.
    _salidas_csv: list[tuple[str, pd.DataFrame, list[str]]] = [
        (
            ARCHIVO_GRANGER, df_granger,
            ["pista", "serie_x", "bloque", "target", "n", "lag",
             "pval_f", "pval_chi2", "nota", "bh_q", "bh_rechaza",
             "ambito_bh"],
        ),
        (
            ARCHIVO_GRANGER_REZAGO, df_granger_rezago,
            ["pista", "serie_x", "bloque", "target", "n", "lag",
             "pval_f", "pval_chi2", "nota"],
        ),
        (
            ARCHIVO_CONF, df_conf,
            ["serie_x", "pista", "target", "lag_mejor", "pval_min_f",
             "pval_min_chi2", "bh_q", "n", "sobrevive_fdr_global",
             "veredicto"],
        ),
        (
            ARCHIVO_BT, df_bt,
            ["pista", "grupo_features", "modelo", "rmse", "mae", "r2_oos",
             "baseline", "rmse_baseline", "mejora_pct_vs_baseline",
             "mejora", "n_test"],
        ),
        (
            ARCHIVO_EVENTOS, df_eventos,
            ["pista", "grupo_features", "modelo", "percentil", "umbral",
             "n_test", "n_eventos", "tp", "fp", "fn", "tn",
             "precision", "recall", "f1", "aviso"],
        ),
        (
            ARCHIVO_REPORTE, df_reporte,
            ["pista", "target", "n_tests_granger",
             "n_granger_sig_fdr_global", "features_sig", "mejor_modelo_rmse",
             "mejor_rmse", "mejor_r2_oos", "mejora_sobre_baseline",
             "conclusion", "ventana_inicio", "ventana_fin", "ventana_nota"],
        ),
    ]
    for _ruta, _tabla, _encabezados in _salidas_csv:
        _actual = _tabla if not _tabla.empty else pd.DataFrame(columns=_encabezados)
        _actual.to_csv(_ruta, index=False)
        if DEBUG:
            print(f"[DEBUG] guardado: {_ruta} ({len(_actual)} filas)")
elif DEBUG:
    print("\n[DEBUG] SAVE_DATA=False — reportes solo en memoria")

