"""# Job 5 - EDA: Caracterización de las series RWA"""

# 1. Configuración

DEBUG: bool = True
SAVE_DATA: bool = True

# 2. Imports

import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from statsmodels.tsa.stattools import adfuller
    _HAS_STATSMODELS: bool = True
except ImportError:
    _HAS_STATSMODELS = False

from google.colab import drive

drive.mount("/content/drive")

"""# 3. Paths y Constantes"""

# CAMBIO E2E-05-01: mismo destino configurable que Jobs 0-4.
BASE_DIR: str = os.environ.get(
    "TESIS_BASE_DIR", "/content/drive/MyDrive/Tesis/expansion"
)
DIR_PROCESSED: str = os.path.join(BASE_DIR, "data", "processed")
DIR_REPORTES: str = os.path.join(
    BASE_DIR, "outputs", "reportes"
)
DIR_FIGURAS: str = os.path.join(
    DIR_REPORTES, "figuras_eda_rwa"
)

# CAMBIO I2-1: ARCHIVO_DIARIAS apunta al dataset final (23
# series + fecha) en vez de series_diarias_ext.csv (que no
# tiene las 14 series FRED). ARCHIVO_LARGAS se mantiene: es la
# única fuente "larga" disponible hoy (cubre STLFSI, CISS y las
# 7 RWA; las 14 FRED no tienen un archivo largo propio en el
# pipeline actual — ver nota de alcance en el cierre de F3).
ARCHIVO_DIARIAS: str = os.path.join(
    DIR_PROCESSED, "series_completas.csv"
)
ARCHIVO_LARGAS: str = os.path.join(
    DIR_PROCESSED, "series_largas_ext.csv"
)

# CAMBIO I2-1: constante única de columnas (reemplaza
# COLS_RWA_RED, COLS_RWA_TIPO, COL_RWA_TOTAL, COLS_ONCHAIN,
# COL_USDY y COLS_BLOQUE_ONCHAIN, que usaban nombres viejos
# —p.ej. "rwa_tipo_treasuries"— y bloques que ya no están en
# series_completas.csv —on-chain, USDY, Ripple—). Mismo
# contenido que COLUMNAS_RWA (Job 2, H5) + COLS_MACRO_FEAT
# (Job 8, H5).
COLS_RWA: list[str] = [
    "rwa_red_stellar",
    "rwa_red_algorand",
    "rwa_tipo_treasurybills",
    "rwa_tipo_commodities",
    "rwa_tipo_privatecredit",
    "rwa_tipo_stocksetfs",
    "rwa_total_global",
]
COL_CISS: str = "macro_euro_ciss"
COL_STLFSI: str = "macro_eeuu_stlfsi"
COLS_MACRO: list[str] = [
    "macro_fed_funds", "macro_sofr",
    "macro_tsy_1m", "macro_tsy_3m",
    "macro_tsy_2y", "macro_tsy_10y",
    "macro_spread_10y2y", "macro_infl_exp_10y",
    "macro_rrp_overnight", "macro_nfci",
    "macro_spread_hy", "macro_spread_ig",
    "macro_indpro", "macro_umcsent",
    COL_CISS, COL_STLFSI,
]
COLUMNAS_FINALES: list[str] = COLS_RWA + COLS_MACRO

ETIQUETAS: dict[str, str] = {
    "rwa_red_stellar": "RWA Stellar (TVL USD)",
    "rwa_red_algorand": "RWA Algorand (TVL USD)",
    "rwa_tipo_treasurybills": "RWA Treasury Bills (TVL USD)",
    "rwa_tipo_commodities": "RWA Commodities (TVL USD)",
    "rwa_tipo_privatecredit": "RWA Private Credit (TVL USD)",
    "rwa_tipo_stocksetfs": "RWA Stocks & ETFs (TVL USD)",
    "rwa_total_global": "RWA Total global (TVL USD)",
    "macro_fed_funds": "Fed Funds Rate (%)",
    "macro_sofr": "SOFR (%)",
    "macro_tsy_1m": "Treasury 1M (%)",
    "macro_tsy_3m": "Treasury 3M (%)",
    "macro_tsy_2y": "Treasury 2Y (%)",
    "macro_tsy_10y": "Treasury 10Y (%)",
    "macro_spread_10y2y": "Spread 10Y-2Y (pp)",
    "macro_infl_exp_10y": "Inflación esperada 10Y (%)",
    "macro_rrp_overnight": "Reverse Repo O/N (USD)",
    "macro_nfci": "NFCI (índice)",
    "macro_spread_hy": "Spread High Yield (pp)",
    "macro_spread_ig": "Spread Investment Grade (pp)",
    "macro_indpro": "Producción Industrial (índice)",
    "macro_umcsent": "Confianza del Consumidor Michigan (índice)",
    "macro_euro_ciss": "CISS Zona Euro (índice)",
    "macro_eeuu_stlfsi": "STLFSI4 (índice)",
}

ADF_ALPHA: float = 0.05
UMBRAL_SALTO_Z: float = 3.0
N_MIN_ADF: int = 15          # mínimo de obs. para correr ADF
N_POTENCIA_ADF: int = 50     # bajo esto, potencia reducida

# Orden de preferencia de transformación. La log-diferencia va
# antes que la 1ª diferencia: en series de adopción con
# crecimiento multiplicativo, la tasa de crecimiento es la
# lectura económica correcta y estabiliza la varianza.
ORDEN_TRANSF: list[str] = ["nivel", "log_dif", "1a_dif"]

# CAMBIO I3-1: frecuencia nativa de cada columna macro (mismo
# contenido que Job 8/9/10, F2-C4). NFCI y STLFSI4 son semanales;
# INDPRO y UMCSENT son mensuales. Antes de diferenciar o correr
# ADF se resamplea a esta frecuencia (ver resamplear_a_nativa),
# para no repetir "casi todos ceros" dentro del propio ADF.
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

# CAMBIO I3-2: copia de solo lectura de TRANSFORM (Job 8), para
# comparar contra la transformación elegida por ADF. No se
# modifica ni se usa para transformar nada en este Job — solo
# para el reporte de I3-d.
TRANSFORM_JOB8: dict[str, str] = {
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
    COL_CISS: "diff",
    COL_STLFSI: "nivel",
}
# Job 8 nombra "diff"/"log_diff"/"nivel"; este Job usa
# "1a_dif"/"log_dif"/"nivel" — mapeo solo para comparar (I3-d).
_MAPA_TRANSF_JOB8: dict[str, str] = {
    "diff": "1a_dif", "log_diff": "log_dif", "nivel": "nivel",
}

sns.set_theme(style="whitegrid", palette="muted")

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


def cargar_series(ruta: str) -> pd.DataFrame:
    """
    Carga un CSV de series con índice fecha (DatetimeIndex).

    Args:
        ruta: Ruta del CSV.

    Returns:
        DataFrame indexado por fecha.
    """
    df = pd.read_csv(
        ruta, parse_dates=["fecha"], index_col="fecha"
    )
    if DEBUG:
        print(f"[DEBUG] cargado: {df.shape} — {ruta}")
    return df


def validar_entrada_eda(ruta: str, job_origen: str) -> pd.DataFrame:
    """CAMBIO E2E-05-01: validar archivo y fechas sin cambiar valores.

    Rechaza entradas ausentes, ilegibles, vacias, con fechas
    invalidas, duplicadas o desordenadas. No imputa ni ordena.
    """
    if not os.path.isfile(ruta):
        raise FileNotFoundError(
            f"[DEBUG][ERROR] Job 5: falta {ruta}. "
            f"Debe generarlo el {job_origen}."
        )
    try:
        df = cargar_series(ruta)
    except Exception as exc:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 5: no se pudo leer {ruta} "
            f"(salida del {job_origen}); verificar CSV y columna 'fecha': {exc}"
        ) from exc
    if df.empty:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 5: entrada vacia: {ruta} "
            f"(salida del {job_origen})."
        )
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 5: fechas invalidas en {ruta}. "
            f"Revisar {job_origen}."
        )
    if df.index.has_duplicates:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 5: fechas duplicadas en {ruta}. "
            "No se puede procesar un indice temporal ambiguo."
        )
    if not df.index.is_monotonic_increasing:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 5: fechas desordenadas en {ruta}. "
            "Revisar la salida original; no se reordena automaticamente."
        )
    return df


def estadistica_descriptiva(
    df: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """
    Estadísticas básicas (n, media, std, min, max, CV, skew,
    cuartiles, atípicos IQR y % de observaciones repetidas).

    CAMBIO I2-2: agrega q25/q50/q75, el recuento de atípicos por
    la regla 1.5×IQR (valores fuera de [q25-1.5·IQR, q75+1.5·IQR])
    y "pct_repetido_ffill": el % de días cuyo valor es idéntico
    al del día anterior. Esto último es una PROXY de ffill, no la
    máscara real (acá no está disponible): un valor puede
    repetirse por una razón genuina y no solo por ffill, así que
    léase como cota superior de la cobertura rellenada.

    Args:
        df: Dataset con las series en columnas.
        cols: Columnas a describir.

    Returns:
        DataFrame con una fila por serie.
    """
    filas: list[dict] = []
    for col in cols:
        s = (
            df[col].dropna()
            if col in df.columns
            else pd.Series(dtype=float)
        )
        if s.empty:
            filas.append({"serie": col, "n": 0})
            continue
        media = float(s.mean())
        cv = s.std() / media if media != 0 else float("nan")
        q25 = float(s.quantile(0.25))
        q50 = float(s.quantile(0.50))
        q75 = float(s.quantile(0.75))
        iqr = q75 - q25
        lim_inf = q25 - 1.5 * iqr
        lim_sup = q75 + 1.5 * iqr
        n_atipicos = int(((s < lim_inf) | (s > lim_sup)).sum())
        serie_completa = (
            df[col] if col in df.columns else pd.Series(dtype=float)
        )
        n_repetido = int((serie_completa.diff() == 0).sum())
        pct_repetido = (
            round(n_repetido / len(serie_completa) * 100, 1)
            if len(serie_completa) > 0 else 0.0
        )
        filas.append({
            "serie": col,
            "etiqueta": ETIQUETAS.get(col, col),
            "n": int(s.notna().sum()),
            "n_nulo": int(df[col].isna().sum()),
            "media": round(media, 4),
            "desvio": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "q25": round(q25, 4),
            "q50_mediana": round(q50, 4),
            "q75": round(q75, 4),
            "n_atipicos_iqr": n_atipicos,
            "cv": round(float(cv), 4),
            "skew": round(float(s.skew()), 4),
            "pct_repetido_ffill": pct_repetido,
            "min_fecha": s.index.min().date().isoformat(),
            "max_fecha": s.index.max().date().isoformat(),
        })
    return pd.DataFrame(filas)


def _guardar_fig(ruta: str) -> None:
    """Guarda la figura activa a disco si SAVE_DATA=True."""
    if SAVE_DATA:
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        plt.savefig(ruta, dpi=150, bbox_inches="tight")
        if DEBUG:
            print(f"[DEBUG] figura: {ruta}")


def plot_serie_simple(
    serie: pd.Series,
    titulo: str,
    fname: str,
) -> None:
    """
    Línea temporal simple para una serie.

    Args:
        serie: Serie con DatetimeIndex.
        titulo: Título de la figura.
        fname: Nombre del archivo PNG.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    serie.plot(ax=ax, lw=1.0, color="steelblue")
    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel("Fecha", fontsize=9)
    ax.set_ylabel("USD", fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    _guardar_fig(os.path.join(DIR_FIGURAS, fname))
    plt.show()
    plt.close()


def plot_tipos_comparado(
    df: pd.DataFrame,
    cols: list[str],
    fname: str = "rwa_tipos_comparado.png",
) -> None:
    """
    Compara en un mismo eje las series por tipo de activo, para
    ver magnitudes relativas.

    Dibuja dos paneles: escala lineal (magnitudes absolutas) y
    escala logarítmica (dinámica relativa, legible cuando los
    órdenes de magnitud difieren).

    Args:
        df: Dataset con las series.
        cols: Columnas de tipo de activo a comparar.
        fname: Nombre del archivo PNG.
    """
    presentes = [c for c in cols if c in df.columns]
    if not presentes:
        if DEBUG:
            print("[DEBUG] sin columnas de tipo para comparar")
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for col in presentes:
        etq = ETIQUETAS.get(col, col)
        df[col].plot(ax=axes[0], lw=1.1, label=etq)
        positivos = df[col].where(df[col] > 0)
        positivos.plot(ax=axes[1], lw=1.1, label=etq)
    axes[0].set_title(
        "RWA por tipo de activo — escala lineal (magnitudes)",
        fontsize=11,
    )
    axes[0].set_ylabel("USD", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_title(
        "RWA por tipo de activo — escala log (dinámica)",
        fontsize=11,
    )
    axes[1].set_ylabel("USD (log)", fontsize=9)
    axes[1].set_xlabel("Fecha", fontsize=9)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=8)
    plt.tight_layout()
    _guardar_fig(os.path.join(DIR_FIGURAS, fname))
    plt.show()
    plt.close()


def _veredicto_adf(p: float) -> str:
    """Clasifica el resultado ADF según significancia."""
    if p < 0.01:
        return "estacionaria (p<0.01)"
    if p < 0.05:
        return "estacionaria (p<0.05)"
    if p < 0.10:
        return "borderline (p<0.10)"
    return "no estacionaria"


def resamplear_a_nativa(serie: pd.Series, col: str) -> pd.Series:
    """
    Resamplea una serie a su frecuencia nativa antes de
    diferenciarla o correr ADF.

    CAMBIO I3-1: regla de F2/C4. Sin esto, una serie
    semanal/mensual ffilleada a diario muestra "casi todos
    ceros" al diferenciarla a diario, y el ADF la declara
    estacionaria por el motivo equivocado (varianza casi nula,
    no reversión a la media real).

    Args:
        serie: Serie con DatetimeIndex (nivel, diaria).
        col: Nombre real de la columna (clave de
            FRECUENCIA_NATIVA). Las columnas RWA (no están en
            FRECUENCIA_NATIVA) son diarias y no se resamplean.

    Returns:
        Serie resampleada ("last") a su frecuencia nativa, o la
        misma serie si ya es diaria.
    """
    nativa = FRECUENCIA_NATIVA.get(col, "D")
    if nativa == "W":
        return serie.resample("W").last()
    if nativa == "M":
        return serie.resample("ME").last()
    return serie


def _adf_una(
    s: pd.Series,
    nombre: str,
    transf: str,
    fuente: str,
    serie_nivel: pd.Series | None = None,
) -> dict:
    """
    Corre ADF sobre una transformación concreta de la serie.

    CAMBIO I3-3: agrega n_nan_transf, n_ceros_nivel y
    n_atipicos_transf — diagnóstico automático de ceros/atípicos/
    NaN para poder explicar estadísticos ADF anómalos (ver caso
    rwa_tipo_privatecredit, I3-c) sin tener que inspeccionar cada
    serie a mano.

    Args:
        s: Serie ya transformada (y ya resampleada a su
            frecuencia nativa si corresponde).
        nombre: Nombre de la serie original.
        transf: "nivel" | "1a_dif" | "log_dif".
        fuente: "ventana" (grilla 2026) | "larga" (historia).
        serie_nivel: Serie de nivel (sin transformar, misma
            frecuencia) usada para contar ceros de origen.

    Returns:
        Dict con serie, fuente, transf, n, adf_stat, p_valor,
        veredicto, aviso de potencia y diagnóstico de
        ceros/atípicos/NaN.
    """
    sc = s.dropna()
    n_ceros_nivel = (
        int((serie_nivel.dropna() == 0).sum())
        if serie_nivel is not None else 0
    )
    base: dict = {
        "serie": nombre,
        "fuente": fuente,
        "transf": transf,
        "n": int(len(sc)),
        "n_nan_transf": int(s.isna().sum()),
        "n_ceros_nivel": n_ceros_nivel,
        "n_atipicos_transf": 0,
        "adf_stat": None,
        "p_valor": None,
        "veredicto": "n/a",
        "potencia": "",
    }
    if len(sc) < N_MIN_ADF:
        base["veredicto"] = f"n<{N_MIN_ADF} ({len(sc)})"
        return base
    if float(sc.std()) == 0:
        base["veredicto"] = "constante"
        return base
    q1, q3 = sc.quantile(0.25), sc.quantile(0.75)
    iqr = q3 - q1
    if iqr > 0:
        base["n_atipicos_transf"] = int(
            ((sc < q1 - 1.5 * iqr) | (sc > q3 + 1.5 * iqr)).sum()
        )
    try:
        stat, pval, _, nobs, *_ = adfuller(sc, autolag="AIC")
        base["adf_stat"] = round(float(stat), 4)
        base["p_valor"] = round(float(pval), 4)
        base["n"] = int(nobs)
        base["veredicto"] = _veredicto_adf(float(pval))
        base["potencia"] = (
            f"n<{N_POTENCIA_ADF}: potencia reducida"
            if int(nobs) < N_POTENCIA_ADF else "ok"
        )
    except Exception as exc:
        base["veredicto"] = f"error: {exc}"
    return base


def adf_serie(
    serie: pd.Series,
    nombre: str,
    fuente: str,
    col_real: str = "",
) -> list[dict]:
    """
    ADF en nivel, 1ª diferencia y log-diferencia.

    La log-diferencia solo se calcula si la serie es
    estrictamente positiva en todo su recorrido; si tiene ceros
    o negativos se registra "n/a" en lugar de forzar el cálculo.

    CAMBIO I3-1: antes de testear, la serie se resamplea a su
    frecuencia nativa (resamplear_a_nativa) si `col_real` es
    semanal/mensual (FRECUENCIA_NATIVA).

    Args:
        serie: Serie a testear (sin transformar).
        nombre: Nombre de la serie (etiqueta, para el reporte).
        fuente: "ventana" o "larga".
        col_real: Nombre real de la columna (clave de
            FRECUENCIA_NATIVA); si no se indica, se asume diaria.

    Returns:
        Lista de 3 dicts (uno por transformación).
    """
    if not _HAS_STATSMODELS:
        return [{
            "serie": nombre,
            "fuente": fuente,
            "transf": "—",
            "n": 0,
            "adf_stat": None,
            "p_valor": None,
            "veredicto": "statsmodels no disponible",
            "potencia": "",
        }]
    serie = resamplear_a_nativa(serie, col_real)
    resultados: list[dict] = []
    resultados.append(
        _adf_una(serie, nombre, "nivel", fuente, serie)
    )
    resultados.append(
        _adf_una(serie.diff(), nombre, "1a_dif", fuente, serie)
    )
    s_pos = serie.dropna()
    if not s_pos.empty and (s_pos > 0).all():
        log_dif = np.log(serie).diff()
        resultados.append(
            _adf_una(log_dif, nombre, "log_dif", fuente, serie)
        )
    else:
        resultados.append({
            "serie": nombre,
            "fuente": fuente,
            "transf": "log_dif",
            "n": 0,
            "adf_stat": None,
            "p_valor": None,
            "veredicto": "n/a (valores ≤ 0)",
            "potencia": "",
        })
    return resultados


def _es_estacionaria(fila: pd.Series) -> bool:
    """
    Indica si una fila de la tabla ADF pasa el umbral ADF_ALPHA.

    Args:
        fila: Fila con columna p_valor.

    Returns:
        True si p_valor < ADF_ALPHA.
    """
    pval = fila.get("p_valor")
    return pval is not None and not pd.isna(pval) and (
        float(pval) < ADF_ALPHA
    )


def elegir_transformacion(
    df_adf: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """
    Decide la transformación estacionaria de cada serie.

    Criterio:
      - Se prioriza la evidencia de la serie LARGA (más
        observaciones ⇒ más potencia del ADF). Si no hay serie
        larga para esa columna, se usa la ventana diaria.
      - Dentro de esa fuente se toma la primera transformación
        de ORDEN_TRANSF que resulte estacionaria a ADF_ALPHA:
        nivel → log_dif → 1a_dif. La log-diferencia precede a la
        1ª diferencia por ser la tasa de crecimiento, lectura
        económica correcta en series de adopción.
      - Si ninguna transformación resulta estacionaria, se
        registra "ninguna" y se deja anotado: la serie NO entra
        al modelo sin tratamiento adicional.

    Args:
        df_adf: Tabla ADF larga (serie, fuente, transf, p_valor).
        cols: Series a resolver.

    Returns:
        DataFrame con serie, fuente_decision, transf_elegida,
        p_valor, n, veredicto y nota.
    """
    filas: list[dict] = []
    for col in cols:
        etiqueta = ETIQUETAS.get(col, col)
        sub = df_adf[df_adf["serie"] == etiqueta]
        if sub.empty:
            filas.append({
                "serie": col,
                "etiqueta": etiqueta,
                "fuente_decision": "s/d",
                "transf_elegida": "s/d",
                "p_valor": None,
                "n": 0,
                "nota": "sin resultados ADF",
            })
            continue
        elegida: dict | None = None
        for fuente in ("larga", "ventana"):
            sub_f = sub[sub["fuente"] == fuente]
            if sub_f.empty:
                continue
            for transf in ORDEN_TRANSF:
                cand = sub_f[sub_f["transf"] == transf]
                if cand.empty:
                    continue
                fila_c = cand.iloc[0]
                if _es_estacionaria(fila_c):
                    elegida = {
                        "fuente_decision": fuente,
                        "transf_elegida": transf,
                        "p_valor": fila_c["p_valor"],
                        "n": fila_c["n"],
                        "nota": str(fila_c["potencia"]),
                    }
                    break
            if elegida is not None:
                break
        if elegida is None:
            filas.append({
                "serie": col,
                "etiqueta": etiqueta,
                "fuente_decision": "—",
                "transf_elegida": "ninguna",
                "p_valor": None,
                "n": 0,
                "nota": (
                    "ninguna transformación resultó estacionaria "
                    f"a alpha={ADF_ALPHA}: no usar sin "
                    "tratamiento adicional"
                ),
            })
            continue
        filas.append({
            "serie": col,
            "etiqueta": etiqueta,
            **elegida,
        })
    return pd.DataFrame(filas)


def construir_tabla_adf_ancha(
    df_adf: pd.DataFrame,
    df_transf: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """
    Pivotea los resultados ADF largos a una fila por serie, con
    las 3 transformaciones como columnas, usando la fuente
    elegida por elegir_transformacion() (serie larga si existe,
    ventana si no — CAMBIO I3-1/I3-b).

    También compara la transformación elegida por ADF contra
    TRANSFORM_JOB8 (CAMBIO I3-2/I3-d): no modifica TRANSFORM, solo
    reporta si coinciden.

    Args:
        df_adf: Tabla ADF larga (salida de adf_serie, con el
            diagnóstico de ceros/atípicos/NaN de I3-3).
        df_transf: Salida de elegir_transformacion().
        cols: Columnas (nombres reales) a incluir, una fila cada
            una.

    Returns:
        DataFrame con una fila por columna de `cols`.
    """
    filas: list[dict] = []
    for col in cols:
        etiqueta = ETIQUETAS.get(col, col)
        fila_transf = df_transf[df_transf["serie"] == col]
        fuente = (
            str(fila_transf.iloc[0]["fuente_decision"])
            if not fila_transf.empty else "—"
        )
        transf_elegida = (
            str(fila_transf.iloc[0]["transf_elegida"])
            if not fila_transf.empty else "s/d"
        )
        sub = df_adf[
            (df_adf["serie"] == etiqueta)
            & (df_adf["fuente"] == fuente)
        ]
        fila: dict = {
            "serie": col,
            "etiqueta": etiqueta,
            "fuente_decision": fuente,
            "transf_elegida": transf_elegida,
        }
        for transf in ("nivel", "1a_dif", "log_dif"):
            cand = sub[sub["transf"] == transf]
            if cand.empty:
                fila[f"adf_stat_{transf}"] = None
                fila[f"p_valor_{transf}"] = None
                fila[f"n_{transf}"] = None
                fila[f"veredicto_{transf}"] = "s/d"
                fila[f"n_ceros_nivel_{transf}"] = None
                fila[f"n_atipicos_{transf}"] = None
                fila[f"n_nan_{transf}"] = None
            else:
                r = cand.iloc[0]
                fila[f"adf_stat_{transf}"] = r["adf_stat"]
                fila[f"p_valor_{transf}"] = r["p_valor"]
                fila[f"n_{transf}"] = r["n"]
                fila[f"veredicto_{transf}"] = r["veredicto"]
                fila[f"n_ceros_nivel_{transf}"] = r.get(
                    "n_ceros_nivel"
                )
                fila[f"n_atipicos_{transf}"] = r.get(
                    "n_atipicos_transf"
                )
                fila[f"n_nan_{transf}"] = r.get("n_nan_transf")

        # CAMBIO I3-2/I3-d: comparación contra TRANSFORM (Job 8).
        transf_job8_raw = TRANSFORM_JOB8.get(col)
        transf_job8 = _MAPA_TRANSF_JOB8.get(
            transf_job8_raw, "s/d"
        )
        fila["transf_job8"] = transf_job8
        fila["coincide_job8"] = (
            transf_elegida == transf_job8
            if transf_elegida not in ("s/d", "ninguna", "—")
            else False
        )
        filas.append(fila)
    return pd.DataFrame(filas)


def detectar_saltos(
    serie: pd.Series,
    nombre: str,
    umbral_z: float = UMBRAL_SALTO_Z,
) -> pd.DataFrame:
    """
    Detecta variaciones diarias anómalas en una serie de TVL.

    Mismo criterio que la detección de saltos de USDY en la
    Fase 3: z sobre la variación diaria absoluta. Sirve para
    identificar mint/redención y altas de emisores.

    Trabaja ÚNICAMENTE sobre la Series numérica (diff /
    pct_change / z-score); la fecha queda como índice compartido
    y se convierte en columna al final con reset_index(). Así se
    evita la mezcla DateTime64/Float64 que dispara el error de
    dtype promotion en pandas >= 2.x.

    Args:
        serie: Serie de nivel con DatetimeIndex.
        nombre: Nombre de la serie (columna 'serie').
        umbral_z: Umbral de |z| para marcar salto.

    Returns:
        DataFrame con fecha, serie, valor, valor_prev,
        delta_abs, delta_pct, z_score. Vacío si no hay saltos.
    """
    # Serie numérica pura: sin fechas dentro de los cálculos
    s = pd.to_numeric(serie, errors="coerce").dropna()
    s = s.astype("float64")
    if len(s) < 3:
        return pd.DataFrame()
    delta = s.diff()                       # variación absoluta
    delta_pct = s.pct_change() * 100.0     # variación %
    media_d = float(delta.mean())
    std_d = float(delta.std())
    if std_d == 0 or pd.isna(std_d):
        return pd.DataFrame()
    z = (delta - media_d) / std_d
    mask = z.abs() > umbral_z
    if not bool(mask.any()):
        return pd.DataFrame()
    # Todas las columnas son float64 y comparten el DatetimeIndex
    df_sal = pd.DataFrame({
        "valor": s[mask].astype("float64"),
        "valor_prev": s.shift(1)[mask].astype("float64"),
        "delta_abs": delta[mask].astype("float64"),
        "delta_pct": delta_pct[mask].astype("float64"),
        "z_score": z[mask].astype("float64"),
    })
    df_sal.index = pd.DatetimeIndex(df_sal.index, name="fecha")
    # La fecha pasa a columna recién acá, ya fuera del cálculo
    df_sal = df_sal.reset_index()
    df_sal.insert(1, "serie", nombre)
    return df_sal


def umbral_texto(umbral: float) -> str:
    """Formatea el umbral de z para las leyendas."""
    return f"{umbral:.0f}"


def plot_serie_con_saltos(
    serie: pd.Series,
    saltos: pd.DataFrame,
    titulo: str,
    fname: str,
) -> None:
    """
    Trayectoria de una serie con los saltos destacados.

    Dibuja con arrays explícitos (index.to_numpy() y
    to_numpy(dtype="float64")) en lugar de Series.plot(): así
    matplotlib recibe fechas y valores por separado y no
    intenta promover DateTime64 contra Float64.

    Args:
        serie: Serie de nivel con DatetimeIndex.
        saltos: Salida de detectar_saltos para esa serie.
        titulo: Título de la figura.
        fname: Nombre del archivo PNG.
    """
    s = pd.to_numeric(serie, errors="coerce")
    fechas = s.index.to_numpy()
    valores = s.to_numpy(dtype="float64")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        fechas, valores,
        color="steelblue", lw=1.0, label="Nivel",
    )
    if not saltos.empty and "valor" in saltos.columns:
        x_saltos = pd.to_datetime(saltos["fecha"]).to_numpy()
        y_saltos = saltos["valor"].to_numpy(dtype="float64")
        ax.scatter(
            x_saltos, y_saltos,
            color="red", zorder=5, s=40,
            label=f"Saltos |z|>{umbral_texto(UMBRAL_SALTO_Z)}",
        )
    ax.set_title(titulo, fontsize=11)
    ax.set_ylabel("USD", fontsize=9)
    ax.set_xlabel("Fecha", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    fig.autofmt_xdate()
    plt.tight_layout()
    _guardar_fig(os.path.join(DIR_FIGURAS, fname))
    plt.show()
    plt.close()


def matriz_correlacion(
    df: pd.DataFrame,
    filas: list[str],
    columnas: list[str],
    titulo: str,
    fname: str,
) -> pd.DataFrame:
    """
    Correlación de Pearson entre dos bloques de series y su
    heatmap. Admite matriz cuadrada (filas == columnas) o
    rectangular (RWA × on-chain).

    Excluye series ausentes o de varianza cero. La correlación
    se calcula por pares (pairwise), sin descartar filas
    completas por un NaN aislado.

    Args:
        df: Dataset con las series.
        filas: Series del eje vertical.
        columnas: Series del eje horizontal.
        titulo: Título de la figura.
        fname: Nombre del archivo PNG.

    Returns:
        DataFrame de correlación (filas × columnas).
    """
    def _validas(cols: list[str]) -> list[str]:
        return [
            c for c in cols
            if c in df.columns
            and df[c].notna().sum() > 2
            and float(df[c].std()) > 0
        ]

    f_ok = _validas(filas)
    c_ok = _validas(columnas)
    if not f_ok or not c_ok:
        if DEBUG:
            print(f"[DEBUG] {titulo}: sin columnas válidas")
        return pd.DataFrame()
    corr_full = df[list(dict.fromkeys(f_ok + c_ok))].corr(
        method="pearson"
    )
    corr = corr_full.loc[f_ok, c_ok]
    ancho = max(6.0, 1.0 * len(c_ok) + 3.0)
    alto = max(4.0, 0.8 * len(f_ok) + 2.0)
    fig, ax = plt.subplots(figsize=(ancho, alto))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1, ax=ax,
        annot_kws={"size": 7}, linewidths=0.4,
    )
    ax.set_title(titulo, fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    plt.tight_layout()
    _guardar_fig(os.path.join(DIR_FIGURAS, fname))
    plt.show()
    plt.close()
    return corr


def corr_a_largo(
    corr: pd.DataFrame,
    bloque: str,
    base: str,
) -> pd.DataFrame:
    """
    Pasa una matriz de correlación a formato largo, para poder
    apilar bloques y bases distintas en un único CSV.

    Args:
        corr: Matriz de correlación.
        bloque: "rwa_vs_rwa" | "rwa_vs_onchain".
        base: "nivel" | "log_dif".

    Returns:
        DataFrame con bloque, base, serie_a, serie_b, corr.
    """
    if corr.empty:
        return pd.DataFrame()
    largo = (
        corr.stack()
        .rename("corr")
        .reset_index()
    )
    largo.columns = ["serie_a", "serie_b", "corr"]
    largo.insert(0, "base", base)
    largo.insert(0, "bloque", bloque)
    largo["corr"] = largo["corr"].round(4)
    return largo


def log_diferencias(
    df: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """
    Construye el DataFrame de log-diferencias de las columnas
    estrictamente positivas.

    Las columnas con ceros o negativos se omiten (no se fuerza
    el logaritmo ni se desplaza la serie: sería inventar dato).

    Args:
        df: Dataset de niveles.
        cols: Columnas a transformar.

    Returns:
        DataFrame de log-diferencias.
    """
    datos: dict[str, pd.Series] = {}
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty or not bool((s > 0).all()):
            if DEBUG:
                print(
                    f"[DEBUG] {col}: no positiva — sin log_dif"
                )
            continue
        datos[col] = np.log(df[col]).diff()
    return pd.DataFrame(datos, index=df.index)


"""# 5. Proceso"""

# CAMBIO B1-4 (Bloque 1): hash de las entradas vs. manifest.
verificar_entradas_snapshot([ARCHIVO_DIARIAS, ARCHIVO_LARGAS], "Job 5")

# ── Carga de datos ──────────────────────────────────────────
# CAMBIO E2E-05-01: las dos entradas son productos de Jobs
# anteriores; se comprueban antes de calcular o guardar figuras.
df = validar_entrada_eda(ARCHIVO_DIARIAS, "Job 4")
df_lrg = validar_entrada_eda(ARCHIVO_LARGAS, "Job 3")

# CAMBIO I2-1: cols_finales_presentes reemplaza a
# cols_rwa_presentes/cols_onchain_presentes por separado —
# COLUMNAS_FINALES (23) es la única lista de columnas del Job.
cols_finales_presentes: list[str] = [
    c for c in COLUMNAS_FINALES if c in df.columns
]
cols_rwa_presentes: list[str] = [
    c for c in COLS_RWA if c in df.columns
]
cols_macro_presentes: list[str] = [
    c for c in COLS_MACRO if c in df.columns
]

# CAMBIO E2E-05-01: detectar un CSV con columnas de otro Job
# antes de producir reportes vacios que parezcan validos.
if not cols_rwa_presentes or not cols_macro_presentes:
    raise RuntimeError(
        "[DEBUG][ERROR] Job 5: series_completas.csv debe tener "
        "al menos una columna RWA y una macro; "
        f"RWA={len(cols_rwa_presentes)}, macro={len(cols_macro_presentes)}. "
        "Revisar la salida del Job 4."
    )

if DEBUG:
    print(f"[DEBUG] dataset: {df.shape}")
    print(
        f"[DEBUG] ventana: {df.index.min().date()} → "
        f"{df.index.max().date()}"
    )
    _faltan = [c for c in COLUMNAS_FINALES if c not in df.columns]
    print(
        f"[DEBUG] columnas finales presentes: "
        f"{len(cols_finales_presentes)}/{len(COLUMNAS_FINALES)}"
    )
    if _faltan:
        print(f"[DEBUG][ALERTA] AUSENTES: {_faltan}")

# CAMBIO E2E-05-01: si el CSV largo falta o es ilegible, el
# preflight ya detuvo el Job; no se publica un ADF parcial.
_cols_largas_disponibles = [
    c for c in (COLS_RWA + [COL_CISS, COL_STLFSI])
    if c in df_lrg.columns
]
if not _cols_largas_disponibles:
    raise RuntimeError(
        "[DEBUG][ERROR] Job 5: series_largas_ext.csv no contiene "
        "columnas RWA, CISS ni STLFSI; revisar la salida del Job 3."
    )

# ── Bloque 1: descriptiva ───────────────────────────────────
if DEBUG:
    print("\n[DEBUG] === BLOQUE 1: descriptiva (23 series) ===")
df_desc = pd.DataFrame()
try:
    df_desc = estadistica_descriptiva(df, COLUMNAS_FINALES)
    if DEBUG:
        print(df_desc.to_string(index=False))
        print(
            "[DEBUG] CV alto + skew alto = serie con nivel "
            "creciente, coherente con adopción incipiente "
            "(típico de las RWA; no esperado en macro)."
        )
except Exception as _exc:
    # CAMBIO E2E-05-01: no ocultar errores que dejen CSV parciales.
    raise RuntimeError(
        "[DEBUG][ERROR] Job 5: fallo el bloque 1 (descriptiva): "
        f"{_exc}"
    ) from _exc

# ── Bloque 2: series con saltos |z|>3 marcados ──────────────
# CAMBIO I2-3: se fusiona con el antiguo Bloque 4 (saltos). Una
# sola figura por serie, serie_<columna>.png, con los saltos
# |z|>3 ya marcados — no se guarda más una versión "simple" sin
# marcar por separado.
if DEBUG:
    print(
        "\n[DEBUG] === BLOQUE 2: series + saltos |z|>3 "
        "(serie_<col>.png) ==="
    )
df_saltos = pd.DataFrame()
try:
    _bloques_saltos: list[pd.DataFrame] = []
    for _col in cols_finales_presentes:
        try:
            _sal = detectar_saltos(
                df[_col], _col, UMBRAL_SALTO_Z
            )
        except Exception as _e:
            _sal = pd.DataFrame()
            if DEBUG:
                print(f"[DEBUG] saltos {_col}: {_e}")
        if not _sal.empty:
            _bloques_saltos.append(_sal)
        plot_serie_con_saltos(
            df[_col],
            _sal,
            f"{ETIQUETAS.get(_col, _col)} — saltos "
            f"|z|>{umbral_texto(UMBRAL_SALTO_Z)}",
            f"serie_{_col}.png",
        )
        if DEBUG:
            print(f"[DEBUG] {_col}: {len(_sal)} saltos")
    if _bloques_saltos:
        df_saltos = pd.concat(
            _bloques_saltos, ignore_index=True
        )
        df_saltos = df_saltos.sort_values(
            ["serie", "fecha"]
        ).reset_index(drop=True)
        if DEBUG:
            print(df_saltos.to_string(index=False))
            print(
                "[DEBUG] Los saltos son candidatos a eventos de "
                "mint/redención, alta de emisor o anuncios "
                "macro — NO necesariamente respuesta a estrés: "
                "verificar caso por caso."
            )
    elif DEBUG:
        print("[DEBUG] sin saltos por encima del umbral")
except Exception as _exc:
    # CAMBIO E2E-05-01: no ocultar errores que dejen CSV parciales.
    raise RuntimeError(
        "[DEBUG][ERROR] Job 5: fallo el bloque 2 (saltos): "
        f"{_exc}"
    ) from _exc

# ── Bloque 3: ADF (ventana + serie larga) ───────────────────
if DEBUG:
    print("\n[DEBUG] === BLOQUE 3: ADF y transformación ===")
df_adf = pd.DataFrame()
df_transf = pd.DataFrame()
df_adf_ancha = pd.DataFrame()
try:
    resultados_adf: list[dict] = []

    # 3.a Ventana diaria (n corto ⇒ poca potencia)
    for _col in cols_finales_presentes:
        _nombre = ETIQUETAS.get(_col, _col)
        try:
            resultados_adf.extend(
                adf_serie(
                    df[_col], _nombre, "ventana", col_real=_col,
                )
            )
        except Exception as _e:
            if DEBUG:
                print(f"[DEBUG] ADF ventana {_col}: {_e}")

    # 3.b Serie larga (más observaciones ⇒ más potencia). Solo
    # cubre STLFSI, CISS y las 7 RWA: las 14 FRED no tienen hoy
    # un archivo de historia larga propio en el pipeline (ver
    # nota de alcance).
    if df_lrg is not None:
        for _col in COLUMNAS_FINALES:
            if _col not in df_lrg.columns:
                if DEBUG:
                    print(
                        f"[DEBUG] {_col} sin historia larga: "
                        "solo ADF de ventana"
                    )
                continue
            _serie_lrg = df_lrg[_col].dropna()
            if _serie_lrg.empty:
                continue
            _nombre = ETIQUETAS.get(_col, _col)
            if DEBUG:
                print(
                    f"[DEBUG] {_col} larga: {len(_serie_lrg)} "
                    f"obs ({_serie_lrg.index.min().date()} → "
                    f"{_serie_lrg.index.max().date()})"
                )
            try:
                resultados_adf.extend(
                    adf_serie(
                        _serie_lrg, _nombre, "larga",
                        col_real=_col,
                    )
                )
            except Exception as _e:
                if DEBUG:
                    print(f"[DEBUG] ADF larga {_col}: {_e}")

    df_adf = pd.DataFrame(resultados_adf)
    if DEBUG and not df_adf.empty:
        _adf_cols = [
            "serie", "fuente", "transf", "n",
            "adf_stat", "p_valor", "veredicto", "potencia",
        ]
        print(df_adf[_adf_cols].to_string(index=False))

    # 3.c Transformación elegida por serie
    df_transf = elegir_transformacion(df_adf, COLUMNAS_FINALES)
    if DEBUG and not df_transf.empty:
        print("\n[DEBUG] Transformación elegida por serie:")
        print(df_transf.to_string(index=False))

    # 3.d CAMBIO I3-2/I3-b: tabla ancha (1 fila/serie, 3
    # transformaciones + comparación contra TRANSFORM de Job 8).
    df_adf_ancha = construir_tabla_adf_ancha(
        df_adf, df_transf, COLUMNAS_FINALES,
    )
    if DEBUG and not df_adf_ancha.empty:
        print("\n[DEBUG] Tabla ADF ancha (eda_adf.csv):")
        print(df_adf_ancha.to_string(index=False))
        _no_coincide = df_adf_ancha.loc[
            ~df_adf_ancha["coincide_job8"], "serie"
        ].tolist()
        if _no_coincide:
            print(
                "[DEBUG][ALERTA] series donde la transformación "
                "elegida por ADF no coincide con TRANSFORM "
                f"(Job 8): {_no_coincide}"
            )

except Exception as _exc:
    # CAMBIO E2E-05-01: no ocultar errores que dejen CSV parciales.
    raise RuntimeError(
        "[DEBUG][ERROR] Job 5: fallo el bloque 3 (ADF): "
        f"{_exc}"
    ) from _exc

# ── Bloques 4 y 5: correlaciones ────────────────────────────
# CAMBIO I2-4: se renumeran (el antiguo Bloque 4 de saltos pasó
# al Bloque 2); se saca on-chain y entra macro (COLS_MACRO).
if DEBUG:
    print("\n[DEBUG] === BLOQUES 4-5: correlaciones ===")
df_corr = pd.DataFrame()
try:
    _piezas_corr: list[pd.DataFrame] = []

    # Log-diferencias: lectura interpretable (sin tendencia)
    df_ld = log_diferencias(
        df, cols_rwa_presentes + cols_macro_presentes
    )
    _ld_rwa = [c for c in cols_rwa_presentes if c in df_ld]
    _ld_macro = [c for c in cols_macro_presentes if c in df_ld]

    # Bloque 4: RWA × RWA
    _corr_rwa_nivel = matriz_correlacion(
        df, cols_rwa_presentes, cols_rwa_presentes,
        "RWA × RWA — nivel (dominado por tendencia común)",
        "corr_rwa_nivel.png",
    )
    _piezas_corr.append(
        corr_a_largo(_corr_rwa_nivel, "rwa_vs_rwa", "nivel")
    )
    _corr_rwa_ld = matriz_correlacion(
        df_ld, _ld_rwa, _ld_rwa,
        "RWA × RWA — log-diferencia (co-movimiento)",
        "corr_rwa_logdif.png",
    )
    _piezas_corr.append(
        corr_a_largo(_corr_rwa_ld, "rwa_vs_rwa", "log_dif")
    )

    # Bloque 5: RWA × macro (16 series FRED/ECB; antes on-chain)
    _corr_mix_nivel = matriz_correlacion(
        df, cols_rwa_presentes, cols_macro_presentes,
        "RWA × macro — nivel",
        "corr_rwa_macro_nivel.png",
    )
    _piezas_corr.append(
        corr_a_largo(_corr_mix_nivel, "rwa_vs_macro", "nivel")
    )
    _corr_mix_ld = matriz_correlacion(
        df_ld, _ld_rwa, _ld_macro,
        "RWA × macro — log-diferencia",
        "corr_rwa_macro_logdif.png",
    )
    _piezas_corr.append(
        corr_a_largo(_corr_mix_ld, "rwa_vs_macro", "log_dif")
    )

    _piezas_corr = [p for p in _piezas_corr if not p.empty]
    if _piezas_corr:
        df_corr = pd.concat(_piezas_corr, ignore_index=True)
    if DEBUG and not df_corr.empty:
        print(
            "[DEBUG] correlaciones registradas: "
            f"{len(df_corr)} pares"
        )
        _fuera_diag = df_corr[
            df_corr["serie_a"] != df_corr["serie_b"]
        ]
        _top = _fuera_diag.reindex(
            _fuera_diag["corr"].abs().sort_values(
                ascending=False
            ).index
        ).head(12)
        print("[DEBUG] pares de mayor |corr|:")
        print(_top.to_string(index=False))
        print(
            "[DEBUG] RECORDATORIO: la correlación alta entre "
            "tipos y total es esperable por construcción (tags "
            "no exclusivos), no es un hallazgo."
        )
except Exception as _exc:
    # CAMBIO E2E-05-01: no ocultar errores que dejen CSV parciales.
    raise RuntimeError(
        "[DEBUG][ERROR] Job 5: fallo el bloque 4-5 (correlaciones): "
        f"{_exc}"
    ) from _exc

# ── Resumen: insumo para el modelo ──────────────────────────
print("\n[EDA] === Transformación por serie (insumo modelo) ===")
if df_transf.empty:
    print("[EDA] sin decisión de transformación disponible")
else:
    for _, _f in df_transf.iterrows():
        print(
            f"[EDA] {_f['serie']}: {_f['transf_elegida']} "
            f"(fuente={_f['fuente_decision']}, "
            f"p={_f['p_valor']}, n={_f['n']}) {_f['nota']}"
        )
    _sin_transf = df_transf[
        df_transf["transf_elegida"].isin(["ninguna", "s/d"])
    ]["serie"].tolist()
    if _sin_transf:
        print(
            f"[EDA][ALERTA] SIN transformación estacionaria: "
            f"{_sin_transf} — no usar sin tratamiento adicional."
        )
print(
    "[EDA] NOTA: en las RWA, la dinámica está dominada por "
    "adopción/crecimiento estructural, no por respuesta a "
    "estrés. Interpretar en consecuencia."
)
# CAMBIO I3-2/I3-d: no se modifica TRANSFORM (Job 8), solo se
# reporta el desacuerdo.
if not df_adf_ancha.empty:
    _desacuerdo = df_adf_ancha[~df_adf_ancha["coincide_job8"]]
    if not _desacuerdo.empty:
        print(
            "\n[EDA] Series donde TRANSFORM (Job 8) no coincide "
            "con la transformación elegida por ADF (no se "
            "modificó TRANSFORM):"
        )
        for _, _d in _desacuerdo.iterrows():
            print(
                f"[EDA]   {_d['serie']}: ADF eligió "
                f"'{_d['transf_elegida']}', TRANSFORM tiene "
                f"'{_d['transf_job8']}'"
            )

"""# 6. Save"""

if SAVE_DATA:
    os.makedirs(DIR_REPORTES, exist_ok=True)
    os.makedirs(DIR_FIGURAS, exist_ok=True)

    # CAMBIO E2E-05-01: contratos CSV explicitos para cero filas.
    # Con datos, se escribe el DataFrame ORIGINAL, sin modificar
    # columnas, orden ni calculos. Sin resultados, se escriben solo
    # encabezados para no dejar en Drive un CSV de otra corrida.
    _cols_desc = [
        "serie", "etiqueta", "n", "n_nulo", "media", "desvio",
        "min", "max", "q25", "q50_mediana", "q75",
        "n_atipicos_iqr", "cv", "skew", "pct_repetido_ffill",
        "min_fecha", "max_fecha",
    ]
    _cols_adf = [
        "serie", "etiqueta", "fuente_decision", "transf_elegida",
    ]
    for _transf in ("nivel", "1a_dif", "log_dif"):
        _cols_adf.extend([
            f"adf_stat_{_transf}", f"p_valor_{_transf}",
            f"n_{_transf}", f"veredicto_{_transf}",
            f"n_ceros_nivel_{_transf}", f"n_atipicos_{_transf}",
            f"n_nan_{_transf}",
        ])
    _cols_adf.extend(["transf_job8", "coincide_job8"])

    _salidas_eda = [
        ("eda_descriptiva.csv", df_desc, _cols_desc),
        ("eda_adf.csv", df_adf_ancha, _cols_adf),
        (
            "eda_rwa_saltos.csv", df_saltos,
            ["fecha", "serie", "valor", "valor_prev",
             "delta_abs", "delta_pct", "z_score"],
        ),
        (
            "eda_rwa_corr.csv", df_corr,
            ["bloque", "base", "serie_a", "serie_b", "corr"],
        ),
    ]
    for _nombre, _tabla, _columnas_vacias in _salidas_eda:
        _salida = (
            _tabla if not _tabla.empty
            else pd.DataFrame(columns=_columnas_vacias)
        )
        _ruta = os.path.join(DIR_REPORTES, _nombre)
        _salida.to_csv(_ruta, index=False)
        if DEBUG:
            print(f"[DEBUG] guardado: {_ruta} ({len(_salida)} filas)")
elif DEBUG:
    print("\n[DEBUG] SAVE_DATA=False — datos solo en memoria")

