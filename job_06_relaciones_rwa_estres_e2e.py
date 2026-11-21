"""# Job 6 - Relaciones temporales: RWA vs. estrés financiero"""

# 1. Configuración

DEBUG: bool = True
SAVE_DATA: bool = True

"""# 2. Imports"""

import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

from google.colab import drive

drive.mount("/content/drive")

"""# 3. Paths y Constantes"""

# CAMBIO E2E-06-01: ruta configurable; conserva el destino original por defecto.
BASE_DIR: str = os.environ.get(
    "TESIS_BASE_DIR", "/content/drive/MyDrive/Tesis/expansion"
)
DIR_PROCESSED: str = os.path.join(BASE_DIR, "data", "processed")
DIR_REPORTES: str = os.path.join(
    BASE_DIR, "outputs", "reportes"
)
DIR_FIGURAS: str = os.path.join(
    DIR_REPORTES, "figuras_relaciones_rwa"
)

ARCHIVO_DIARIAS: str = os.path.join(
    DIR_PROCESSED, "series_diarias_ext.csv"
)
ARCHIVO_LARGAS: str = os.path.join(
    DIR_PROCESSED, "series_largas_ext.csv"
)

# CAMBIO I4-1: se elimina el recorte fijo al 2026-07-31 (era un
# workaround puntual para un artefacto de DeFiLlama en Ripple
# que ya no se verifica) y el archivo cacheado
# series_diarias_ext_31jul.csv. FECHA_CORTE ahora es None por
# defecto: se usa la ventana completa del dataset. Si se
# completa con una fecha ISO, recortar_ventana() sigue
# recortando TODAS las columnas a la vez.
FECHA_CORTE: str | None = None

# ── Series RWA: set final de 8 (12_inventario_rwa) ───────────
# Todas son STOCKS (TVL en USD): en el resample semanal se toma
# last(), nunca sum() — sumar un stock contaría siete veces el
# mismo capital.
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

COL_CISS: str = "macro_euro_ciss"
COL_STLFSI: str = "macro_eeuu_stlfsi"

K_DIARIO: int = 14      # lags máximos Pista A (días)
K_SEMANAL: int = 8      # lags máximos Pista B (semanas)
UMBRAL_R: float = 0.20  # piso exploratorio |r|
ALFA_BH: float = 0.05   # nivel FDR Benjamini-Hochberg
N_MIN_SEMANAS: int = 40  # bajo esto se avisa potencia baja

# Etiquetas cortas (* = caveat de adopción incipiente)
ETIQUETAS: dict[str, str] = {
    "rwa_red_stellar": "RWA-Stellar*",
    "rwa_red_ripple": "RWA-Ripple*",
    "rwa_red_algorand": "RWA-Algorand*",
    "rwa_tipo_treasurybills": "RWA-Treasuries*",
    "rwa_tipo_commodities": "RWA-Commodities*",
    "rwa_tipo_privatecredit": "RWA-PrivCredit*",
    "rwa_tipo_stocksetfs": "RWA-StocksETFs*",
    "rwa_total_global": "RWA-Total*",
}

# Series cuya relación con el estrés tiene dirección teórica
# previa (activo refugio): en un episodio de estrés se espera
# flujo HACIA oro y deuda soberana tokenizados. Se las reporta
# aparte para trazabilidad, marcadas como indicio NO confirmado
# cuando no sobreviven el FDR global.
SERIES_DIRECCION_TEORICA: list[str] = [
    "rwa_tipo_treasurybills",
    "rwa_tipo_commodities",
]

# Nota de adopción (se añade a todas las filas RWA de los CSVs)
NOTA_ADOPCION: str = (
    "RWA de adopción incipiente (2025-2026): dinámica dominada "
    "por crecimiento estructural, no por respuesta a estrés. "
    "Relación posiblemente por tendencia común; exploratorio."
)

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


def validar_entrada_relaciones(ruta: str, job_origen: str) -> pd.DataFrame:
    """CAMBIO E2E-06-01: valida existencia, CSV y fechas, sin alterar valores.

    Cada entrada del Job 6 es una salida obligatoria del Job 3.
    No ordena, rellena, filtra, agrega ni sustituye datos.
    """
    if not os.path.isfile(ruta):
        raise FileNotFoundError(
            f"[DEBUG][ERROR] Job 6: falta {ruta}. "
            f"Debe generarlo el {job_origen}."
        )
    try:
        df = cargar_series(ruta)
    except Exception as exc:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 6: no se pudo leer {ruta} "
            f"(salida del {job_origen}); revisar CSV y columna 'fecha': {exc}"
        ) from exc
    if df.empty:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 6: entrada vacia: {ruta} "
            f"(salida del {job_origen})."
        )
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 6: fechas invalidas en {ruta}. "
            f"Revisar {job_origen}."
        )
    if df.index.has_duplicates:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 6: fechas duplicadas en {ruta}. "
            "No se puede usar un indice temporal ambiguo."
        )
    if not df.index.is_monotonic_increasing:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 6: fechas desordenadas en {ruta}. "
            "No se reordena automaticamente."
        )
    return df


def recortar_ventana(
    df: pd.DataFrame,
    fecha_corte: str | None,
) -> pd.DataFrame:
    """
    Recorta el dataset a fecha <= fecha_corte, si se indica.

    El recorte se aplica a TODAS las columnas a la vez para que
    on-chain, macro y RWA cubran exactamente el mismo período.

    CAMBIO I4-1: si fecha_corte es None (default), no se recorta
    — se usa la ventana completa del dataset.

    Args:
        df: Dataset diario indexado por fecha.
        fecha_corte: Última fecha incluida (ISO), o None para no
            recortar.

    Returns:
        DataFrame recortado, o el mismo dataset si fecha_corte
        es None.
    """
    if fecha_corte is None:
        if DEBUG:
            print(f"[DEBUG] sin recorte: ventana completa ({len(df)} filas)")
        return df
    corte = pd.Timestamp(fecha_corte)
    recortado = df.loc[df.index <= corte].copy()
    recortado.index.name = "fecha"
    if DEBUG:
        n_fuera = len(df) - len(recortado)
        print(
            f"[DEBUG] recorte <= {fecha_corte}: "
            f"{len(df)} → {len(recortado)} filas "
            f"({n_fuera} descartadas)"
        )
    return recortado


def _log_diff(serie: pd.Series) -> pd.Series:
    """
    Log-diferencia segura: los valores <= 0 pasan a NaN antes
    del logaritmo, evitando -inf en el cálculo.

    Args:
        serie: Serie de nivel.

    Returns:
        Serie de log-diferencias.
    """
    return np.log(serie.where(serie > 0)).diff()


def _par_corr(
    x: pd.Series,
    y: pd.Series,
    lag: int,
    metodo: str = "pearson",
) -> tuple[float, float, int]:
    """
    Correlación de x adelantado `lag` períodos con y.

    lag>0: x[t-lag] correlaciona con y[t] → x precede a y.
    Alinea por índice (concat → dropna), así que funciona
    aunque x e y tengan ventanas distintas.

    Args:
        x: Serie explicativa (RWA transformada).
        y: Serie objetivo (índice de estrés transformado).
        lag: Desplazamiento en períodos.
        metodo: "pearson" o "spearman".

    Returns:
        (r, p, n). r y p son NaN si n < 10.
    """
    x_lead = x.shift(lag) if lag > 0 else x
    xy = pd.concat(
        [x_lead.rename("x"), y.rename("y")], axis=1
    ).dropna()
    n = len(xy)
    if n < 10:
        return float("nan"), float("nan"), n
    xi, yi = xy["x"].values, xy["y"].values
    try:
        if metodo == "pearson":
            r, p = pearsonr(xi, yi)
        else:
            r, p = spearmanr(xi, yi)
        return float(r), float(p), n
    except Exception:
        return float("nan"), float("nan"), n


def transformar_pista_a(
    df: pd.DataFrame,
) -> tuple[dict[str, pd.Series], pd.Series]:
    """
    Pista A — diaria:
      · Las 8 RWA → log-diferencia diaria.
      · CISS → primera diferencia diaria.

    Args:
        df: Dataset diario ya recortado.

    Returns:
        (series_x, serie_y) con las RWA transformadas y ΔCISS.
    """
    series_x: dict[str, pd.Series] = {}
    for col in COLS_RWA:
        if col not in df.columns:
            if DEBUG:
                print(f"[DEBUG] A: {col} ausente — se omite")
            continue
        series_x[col] = _log_diff(df[col]).dropna()

    if COL_CISS not in df.columns:
        raise ValueError(f"{COL_CISS} no está en el dataset")
    serie_y = df[COL_CISS].diff().dropna()

    if DEBUG:
        for col, s in series_x.items():
            print(
                f"[DEBUG] A-x {ETIQUETAS.get(col, col)}: "
                f"{len(s)} obs"
            )
        print(f"[DEBUG] A-y ΔCISS: {len(serie_y)} obs")
    return series_x, serie_y


def transformar_pista_b(
    df: pd.DataFrame,
    df_lrg: pd.DataFrame | None,
) -> tuple[dict[str, pd.Series], pd.Series]:
    """
    Pista B — semanal:
      · Las 8 RWA → resample W last() (son stocks: el TVL de
        cierre de la semana) → log-diferencia.
      · STLFSI → nivel semanal nativo (de series_largas_ext si
        está disponible; si no, del dataset diario).

    Args:
        df: Dataset diario ya recortado.
        df_lrg: Dataset de series largas (o None).

    Returns:
        (series_x, serie_y) semanales.
    """
    series_x: dict[str, pd.Series] = {}
    for col in COLS_RWA:
        if col not in df.columns:
            continue
        series_x[col] = _log_diff(
            df[col].resample("W").last()
        ).dropna()

    if df_lrg is not None and COL_STLFSI in df_lrg.columns:
        fuente = df_lrg[COL_STLFSI]
        origen = "series_largas_ext"
    elif COL_STLFSI in df.columns:
        fuente = df[COL_STLFSI]
        origen = "series_diarias_ext (ffill)"
    else:
        raise ValueError(f"{COL_STLFSI} no disponible")
    # El recorte también aplica al target, para comparar el
    # mismo período en ambas pistas.
    # CAMBIO B2-2 (Bloque 2): con FECHA_CORTE=None, pd.Timestamp(None)
    # es NaT y la comparación daba todo False: Pista B quedaba con 0
    # semanas. Se recorta solo si hay fecha de corte (como en
    # recortar_ventana).
    if FECHA_CORTE is not None:
        fuente = fuente.loc[fuente.index <= pd.Timestamp(FECHA_CORTE)]
    stlfsi_sem = fuente.resample("W").last().dropna()

    if DEBUG:
        for col, s in series_x.items():
            print(
                f"[DEBUG] B-x {ETIQUETAS.get(col, col)}: "
                f"{len(s)} obs"
            )
        print(
            f"[DEBUG] B-y STLFSI (nivel sem., {origen}): "
            f"{len(stlfsi_sem)} obs"
        )
    return series_x, stlfsi_sem


def analizar_contemporaneo(
    series_x: dict[str, pd.Series],
    serie_y: pd.Series,
    nombre_macro: str,
    pista: str,
) -> pd.DataFrame:
    """
    Correlación contemporánea (lag=0): Pearson y Spearman para
    cada par (RWA, macro).

    Args:
        series_x: Series RWA transformadas.
        serie_y: Serie macro transformada.
        nombre_macro: Nombre del índice ("CISS"/"STLFSI").
        pista: Etiqueta de pista.

    Returns:
        DataFrame con una fila por par.
    """
    filas: list[dict] = []
    for col, sx in series_x.items():
        rp, pp, n = _par_corr(sx, serie_y, 0, "pearson")
        rs, ps, _ = _par_corr(sx, serie_y, 0, "spearman")
        filas.append({
            "pista": pista,
            "serie_x": col,
            "etiqueta": ETIQUETAS.get(col, col),
            "macro": nombre_macro,
            "n": n,
            "pearson_r": (
                round(rp, 4) if np.isfinite(rp) else None
            ),
            "pearson_p": (
                round(pp, 4) if np.isfinite(pp) else None
            ),
            "spearman_r": (
                round(rs, 4) if np.isfinite(rs) else None
            ),
            "spearman_p": (
                round(ps, 4) if np.isfinite(ps) else None
            ),
            "nota": NOTA_ADOPCION,
        })
    return pd.DataFrame(filas)


def analizar_lags(
    series_x: dict[str, pd.Series],
    serie_y: pd.Series,
    nombre_macro: str,
    K: int,
    pista: str,
) -> pd.DataFrame:
    """
    Correlación cruzada Pearson para lag k=0..K.

    En cada fila: serie_x[t-k] correlaciona con macro[t]; k>0
    significa que la RWA precede al índice de estrés.

    Args:
        series_x: Series RWA transformadas.
        serie_y: Serie macro transformada.
        nombre_macro: Nombre del índice.
        K: Lag máximo.
        pista: Etiqueta de pista.

    Returns:
        DataFrame con una fila por (serie, lag).
    """
    filas: list[dict] = []
    for col, sx in series_x.items():
        for k in range(K + 1):
            r, p, n = _par_corr(sx, serie_y, k, "pearson")
            filas.append({
                "pista": pista,
                "serie_x": col,
                "etiqueta": ETIQUETAS.get(col, col),
                "macro": nombre_macro,
                "lag": k,
                "n": n,
                "pearson_r": (
                    round(r, 4) if np.isfinite(r) else None
                ),
                "pearson_p": (
                    round(p, 4) if np.isfinite(p) else None
                ),
                "nota": NOTA_ADOPCION,
            })
    return pd.DataFrame(filas)


def lag_optimo_por_par(df_lags: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada (pista, serie_x, macro): el lag de máxima |r| y
    sus estadísticos asociados.

    IMPORTANTE: debe recibir la tabla de lags YA CONSOLIDADA y
    con el FDR global aplicado. Las columnas bh_q y bh_rechaza
    del lag elegido se arrastran tal cual, de modo que la
    significancia de cada par es siempre la del ajuste sobre el
    conjunto completo de tests, nunca la de un subconjunto.

    Args:
        df_lags: Tabla consolidada de correlaciones por lag,
            con bh_q / bh_rechaza globales.

    Returns:
        DataFrame con una fila por par.
    """
    filas: list[dict] = []
    grupos = df_lags.groupby(
        ["pista", "serie_x", "macro"], sort=False
    )
    for (pista, sx, macro), grp in grupos:
        grp_v = grp.dropna(subset=["pearson_r"])
        if grp_v.empty:
            continue
        idx_max = grp_v["pearson_r"].abs().idxmax()
        fila = grp_v.loc[idx_max]
        filas.append({
            "pista": pista,
            "serie_x": sx,
            "etiqueta": ETIQUETAS.get(sx, sx),
            "macro": macro,
            "lag_optimo": int(fila["lag"]),
            "r_max": round(float(fila["pearson_r"]), 4),
            "p_lag_opt": round(float(fila["pearson_p"]), 4),
            "n": int(fila["n"]),
            # BH global del lag elegido (no se recalcula)
            "bh_q": fila.get("bh_q", np.nan),
            "bh_rechaza": bool(fila.get("bh_rechaza", False)),
            "ambito_bh": str(fila.get("ambito_bh", "")),
            "nota": str(fila.get("nota", "")),
        })
    return pd.DataFrame(filas)


def aplicar_bh(
    df: pd.DataFrame,
    col_p: str,
    alfa: float = ALFA_BH,
    col_q: str = "bh_q",
    col_rechaza: str = "bh_rechaza",
) -> pd.DataFrame:
    """
    Corrección Benjamini-Hochberg (FDR) sobre col_p.

    Los nombres de las columnas de salida son parametrizables
    para poder guardar en la misma tabla el ajuste por pista y
    el ajuste global sobre todos los tests.

    Args:
        df: Tabla con p-values.
        col_p: Columna de p-values.
        alfa: Nivel de FDR.
        col_q: Nombre de la columna de q-valores.
        col_rechaza: Nombre de la columna booleana.

    Returns:
        Copia del DataFrame con las dos columnas añadidas.
    """
    df = df.copy().reset_index(drop=True)
    df[col_q] = np.nan
    df[col_rechaza] = False
    pvals = pd.to_numeric(df[col_p], errors="coerce")
    validos = pvals.notna()
    if validos.sum() < 2:
        return df
    rechaza, adj, _, _ = multipletests(
        pvals[validos].values, alpha=alfa, method="fdr_bh"
    )
    df.loc[validos, col_q] = np.round(adj, 4)
    df.loc[validos, col_rechaza] = rechaza
    return df


def clasificar_precedencia(df_opt: pd.DataFrame) -> pd.DataFrame:
    """
    Clasifica cada par según su señal de precedencia.

    La significancia se lee SIEMPRE de bh_rechaza, que proviene
    del FDR aplicado sobre el conjunto completo de tests de
    ambas pistas. Un par no puede clasificarse como señal por
    haber sobrevivido un ajuste sobre un subconjunto.

    Clases:
      · 'precede (sig BH global)' — lag>0, |r|>umbral y
        sobrevive el FDR global.
      · 'contemporaneo (sig)'     — lag=0, |r|>umbral, p crudo
        <alfa. Sin BH: es una única comparación por par.
      · 'indicio no confirmado'   — lag>0, |r|>umbral, en una
        serie con dirección teórica previa, pero NO sobrevive
        el FDR global. Se reporta para trazabilidad; NO es
        señal.
      · 'precede (no sig)'        — lag>0, |r|>umbral, sin
        significancia ni dirección teórica previa.
      · 'no_sig'                  — el resto.

    Args:
        df_opt: Tabla de lags óptimos con el BH global
            arrastrado desde la tabla de lags consolidada.

    Returns:
        DataFrame con la columna 'precedencia'.
    """
    df = df_opt.copy()
    lag = df["lag_optimo"]
    r_abs = df["r_max"].abs()
    bh_ok = df["bh_rechaza"].fillna(False).astype(bool)
    p_raw = pd.to_numeric(df["p_lag_opt"], errors="coerce")
    teorica = df["serie_x"].isin(SERIES_DIRECCION_TEORICA)

    cond_pre_sig = (lag > 0) & (r_abs > UMBRAL_R) & bh_ok
    cond_cont = (
        (lag == 0) & (r_abs > UMBRAL_R) & (p_raw < ALFA_BH)
    )
    cond_pre_deb = (lag > 0) & (r_abs > UMBRAL_R) & ~bh_ok
    cond_indicio = cond_pre_deb & teorica

    estado = pd.Series("no_sig", index=df.index)
    estado.loc[cond_pre_deb] = "precede (no sig)"
    estado.loc[cond_indicio] = "indicio no confirmado"
    estado.loc[cond_cont] = "contemporaneo (sig)"
    estado.loc[cond_pre_sig] = "precede (sig BH global)"
    df["precedencia"] = estado
    return df


def _top_pares(
    df_opt: pd.DataFrame,
    series_dict: dict[str, pd.Series],
    n: int = 3,
) -> list[str]:
    """
    Series RWA con mayor |r_max| entre las que tienen datos.

    Args:
        df_opt: Tabla de lags óptimos.
        series_dict: Series disponibles de esa pista.
        n: Cuántas devolver.

    Returns:
        Lista de nombres de columna.
    """
    if df_opt.empty:
        return []
    sub = (
        df_opt[df_opt["serie_x"].isin(series_dict)]
        .dropna(subset=["r_max"])
        .sort_values(
            "r_max", key=lambda x: x.abs(), ascending=False
        )
    )
    return sub["serie_x"].head(n).tolist()


def plot_ccf(
    serie_x: pd.Series,
    serie_y: pd.Series,
    K: int,
    nombre_x: str,
    nombre_y: str,
    pista: str,
    ruta: str,
) -> None:
    """
    Correlograma cruzado: r(k) para k=0..K.

    k>0 → serie_x precede a serie_y. Bandas: ±1.96/√n
    (asintótica) y ±UMBRAL_R (referencia exploratoria).

    Args:
        serie_x: Serie RWA transformada.
        serie_y: Serie macro transformada.
        K: Lag máximo.
        nombre_x: Etiqueta de la RWA.
        nombre_y: Etiqueta del macro.
        pista: Etiqueta de pista (define la unidad del eje).
        ruta: Ruta del PNG.
    """
    lags = list(range(K + 1))
    corrs: list[float] = []
    n0 = 0
    for k in lags:
        r, _, n = _par_corr(serie_x, serie_y, k, "pearson")
        corrs.append(r if np.isfinite(r) else 0.0)
        if k == 0:
            n0 = n

    conf = 1.96 / np.sqrt(max(n0, 1))
    unidad = "días" if pista.startswith("A") else "semanas"

    fig, ax = plt.subplots(figsize=(10, 4))
    colores = [
        "crimson" if abs(c) > UMBRAL_R else "steelblue"
        for c in corrs
    ]
    ax.bar(lags, corrs, color=colores, alpha=0.75, width=0.6)
    ax.axhline(
        conf, color="grey", lw=0.9, ls=":",
        label=f"±1.96/√n  (n={n0})",
    )
    ax.axhline(-conf, color="grey", lw=0.9, ls=":")
    ax.axhline(
        UMBRAL_R, color="red", lw=0.9, ls="--", alpha=0.6,
        label=f"|r|={UMBRAL_R} (umbral)",
    )
    ax.axhline(
        -UMBRAL_R, color="red", lw=0.9, ls="--", alpha=0.6
    )
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(
        f"CCF — {nombre_x} → {nombre_y}  [{pista}]\n"
        f"lag>0: {nombre_x} precede a {nombre_y}",
        fontsize=10,
    )
    ax.set_xlabel(f"Lag ({unidad})", fontsize=9)
    ax.set_ylabel("Pearson r", fontsize=9)
    ax.set_xticks(lags)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    if SAVE_DATA:
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        plt.savefig(ruta, dpi=150, bbox_inches="tight")
        if DEBUG:
            print(f"[DEBUG] figura: {ruta}")
    plt.show()
    plt.close()


"""# 5. Proceso"""

# CAMBIO B1-4 (Bloque 1): hash de las entradas vs. manifest.
verificar_entradas_snapshot([ARCHIVO_DIARIAS, ARCHIVO_LARGAS], "Job 6")

# ── Carga de datos ──────────────────────────────────────────
# CAMBIO E2E-06-01: las dos entradas las produce el Job 3.
# No se continua con una entrada larga ausente o ilegitima,
# porque el fallback silencioso puede alterar la fuente de STLFSI.
df_full = validar_entrada_relaciones(ARCHIVO_DIARIAS, "Job 3")
df_lrg = validar_entrada_relaciones(ARCHIVO_LARGAS, "Job 3")

# Contrato de columnas: Ripple y otras RWA pueden estar ausentes;
# no se crean ni imputan columnas. Los dos targets deben existir
# en alguna de las fuentes que el codigo original consulta.
if COL_CISS not in df_full.columns:
    raise RuntimeError(
        f"[DEBUG][ERROR] Job 6: falta {COL_CISS} en "
        f"{ARCHIVO_DIARIAS} (Job 3)."
    )
# CAMBIO B2-3 (Bloque 2): rwa_red_ripple sale de COLS_RWA si no
# existe en los datos (columna ausente o sin ningún valor). Se
# decide por existencia, no por resultados; se informa en el log.
_COL_RIPPLE = "rwa_red_ripple"
if _COL_RIPPLE in COLS_RWA and (
    _COL_RIPPLE not in df_full.columns or df_full[_COL_RIPPLE].notna().sum() == 0
):
    for _lista in (COLS_RWA,):
        if _COL_RIPPLE in _lista:
            _lista.remove(_COL_RIPPLE)
    print(
        f"[DEBUG][ALERTA] Job 6: {_COL_RIPPLE} sin datos en "
        f"{os.path.basename(ARCHIVO_DIARIAS)}: se quita de COLS_RWA "
        f"({len(COLS_RWA)} series RWA)."
    )
elif DEBUG:
    print(f"[DEBUG] Job 6: {_COL_RIPPLE} presente, queda en COLS_RWA")

if not any(col in df_full.columns for col in COLS_RWA):
    raise RuntimeError(
        "[DEBUG][ERROR] Job 6: series_diarias_ext.csv no tiene "
        "ninguna de las columnas RWA configuradas. Revisar Job 3."
    )
if COL_STLFSI not in df_lrg.columns and COL_STLFSI not in df_full.columns:
    raise RuntimeError(
        f"[DEBUG][ERROR] Job 6: falta {COL_STLFSI} en "
        "series_largas_ext.csv y series_diarias_ext.csv. Revisar Job 3."
    )

# CAMBIO I4-1: se elimina la verificación del artefacto de
# Ripple (ya no se recorta por defecto — FECHA_CORTE=None).
if DEBUG:
    print("\n[DEBUG] ══ VENTANA DE ANÁLISIS ══")

df: pd.DataFrame = recortar_ventana(df_full, FECHA_CORTE)

# CAMBIO E2E-06-01: evitar .date() sobre NaT con ventana vacia.
if df.empty:
    raise RuntimeError(
        "[DEBUG][ERROR] Job 6: la ventana de analisis quedo vacia "
        f"tras aplicar FECHA_CORTE={FECHA_CORTE!r}."
    )

if DEBUG:
    print(
        f"[DEBUG] ventana de análisis: "
        f"{df.index.min().date()} → {df.index.max().date()} "
        f"({len(df)} días, {df.shape[1]} columnas)"
    )

# Inicializar DataFrames por si una pista falla
df_cont_a = df_lags_a = df_opt_a = pd.DataFrame()
df_cont_b = df_lags_b = df_opt_b = pd.DataFrame()
sx_a: dict[str, pd.Series] = {}
sx_b: dict[str, pd.Series] = {}
sy_a: pd.Series = pd.Series(dtype=float)
sy_b: pd.Series = pd.Series(dtype=float)

# ── PISTA A: diaria — RWA log-dif vs ΔCISS ──────────────────
if DEBUG:
    print("\n[DEBUG] ══ PISTA A: DIARIA (RWA vs ΔCISS) ══")
try:
    sx_a, sy_a = transformar_pista_a(df)

    df_cont_a = analizar_contemporaneo(
        sx_a, sy_a, "CISS", "A-diaria"
    )
    df_lags_a = analizar_lags(
        sx_a, sy_a, "CISS", K_DIARIO, "A-diaria"
    )
    # El FDR NO se aplica acá: se aplica una sola vez, más
    # abajo, sobre las dos pistas juntas. Un ajuste por pista
    # sobreestimaría la significancia.
    if DEBUG:
        _cols = [
            "etiqueta", "n", "pearson_r", "pearson_p",
            "spearman_r", "spearman_p",
        ]
        print("\n[DEBUG] Contemporánea Pista A:")
        print(df_cont_a[_cols].to_string(index=False))
        print(
            f"[DEBUG] tests de lags Pista A: {len(df_lags_a)}"
        )
except Exception as _exc:
    # CAMBIO E2E-06-01: una falla de calculo no se presenta como
    # "sin señal" ni habilita CSV parciales.
    raise RuntimeError(
        f"[DEBUG][ERROR] Job 6: fallo la Pista A: {_exc}"
    ) from _exc

# ── PISTA B: semanal — RWA log-dif sem. vs STLFSI ───────────
if DEBUG:
    print("\n[DEBUG] ══ PISTA B: SEMANAL (RWA vs STLFSI) ══")
try:
    sx_b, sy_b = transformar_pista_b(df, df_lrg)

    _n_sem = len(sy_b)
    if _n_sem < N_MIN_SEMANAS:
        print(
            f"[RWA] AVISO: Pista B con {_n_sem} semanas "
            f"(< {N_MIN_SEMANAS}) — potencia estadística baja; "
            "resultados orientativos únicamente."
        )

    df_cont_b = analizar_contemporaneo(
        sx_b, sy_b, "STLFSI", "B-semanal"
    )
    df_lags_b = analizar_lags(
        sx_b, sy_b, "STLFSI", K_SEMANAL, "B-semanal"
    )
    # Igual que en la Pista A: el FDR es global, no por pista.
    if DEBUG:
        _cols = [
            "etiqueta", "n", "pearson_r", "pearson_p",
            "spearman_r", "spearman_p",
        ]
        print("\n[DEBUG] Contemporánea Pista B:")
        print(df_cont_b[_cols].to_string(index=False))
        print(
            f"[DEBUG] tests de lags Pista B: {len(df_lags_b)}"
        )
except Exception as _exc:
    # CAMBIO E2E-06-01: no ocultar fallas en la Pista B.
    raise RuntimeError(
        f"[DEBUG][ERROR] Job 6: fallo la Pista B: {_exc}"
    ) from _exc

# ── Tablas consolidadas y FDR GLOBAL (único ajuste) ─────────
# El universo de tests es: 8 RWA × 2 macro × (K+1) lags de cada
# pista. La corrección Benjamini-Hochberg se aplica UNA sola vez
# sobre ese conjunto completo. Aplicarla por pista (8 tests) o
# por par inflaría la significancia: con ~200 tests, varios
# p<0.05 aparecen por azar.
df_cont: pd.DataFrame = pd.concat(
    [df_cont_a, df_cont_b], ignore_index=True
)
if not df_cont.empty:
    df_cont = aplicar_bh(df_cont, "pearson_p")
    df_cont["ambito_bh"] = "contemporaneas (lag=0, 2 pistas)"

df_lags: pd.DataFrame = pd.concat(
    [df_lags_a, df_lags_b], ignore_index=True
)

n_tests_global: int = 0
n_sig_crudo: int = 0
n_sig_bh_global: int = 0

if not df_lags.empty:
    df_lags = aplicar_bh(df_lags, "pearson_p")
    df_lags["ambito_bh"] = (
        "global: 2 pistas × todos los lags"
    )
    _p = pd.to_numeric(df_lags["pearson_p"], errors="coerce")
    n_tests_global = int(_p.notna().sum())
    n_sig_crudo = int((_p < ALFA_BH).sum())
    n_sig_bh_global = int(
        df_lags["bh_rechaza"].fillna(False).astype(bool).sum()
    )
    if DEBUG:
        print(
            f"\n[DEBUG] FDR GLOBAL — {n_tests_global} tests | "
            f"p<{ALFA_BH} crudo: {n_sig_crudo} | "
            f"sobreviven BH: {n_sig_bh_global}"
        )

# Lag óptimo por par: arrastra el BH GLOBAL del lag elegido
df_opt: pd.DataFrame = pd.DataFrame()
if not df_lags.empty:
    df_opt = lag_optimo_por_par(df_lags)
    df_opt = clasificar_precedencia(df_opt)

df_opt_a = (
    df_opt[df_opt["pista"] == "A-diaria"]
    if not df_opt.empty else pd.DataFrame()
)
df_opt_b = (
    df_opt[df_opt["pista"] == "B-semanal"]
    if not df_opt.empty else pd.DataFrame()
)

if DEBUG and not df_opt.empty:
    _cols2 = [
        "pista", "etiqueta", "macro", "lag_optimo", "r_max",
        "p_lag_opt", "bh_q", "bh_rechaza", "precedencia",
    ]
    print("\n[DEBUG] Lag óptimo por par (BH global):")
    print(df_opt[_cols2].to_string(index=False))

# ── Figuras: CCF top-3 por pista ────────────────────────────
if DEBUG:
    print("\n[DEBUG] ══ FIGURAS CCF ══")

try:
    for _col in _top_pares(df_opt_a, sx_a, n=3):
        plot_ccf(
            sx_a[_col], sy_a, K_DIARIO,
            ETIQUETAS.get(_col, _col), "ΔCISS", "A-diaria",
            os.path.join(DIR_FIGURAS, f"ccf_A_{_col}.png"),
        )
except Exception as _exc:
    if DEBUG:
        print(f"[DEBUG] CCF Pista A falló: {_exc}")

try:
    for _col in _top_pares(df_opt_b, sx_b, n=3):
        plot_ccf(
            sx_b[_col], sy_b, K_SEMANAL,
            ETIQUETAS.get(_col, _col), "STLFSI", "B-semanal",
            os.path.join(DIR_FIGURAS, f"ccf_B_{_col}.png"),
        )
except Exception as _exc:
    if DEBUG:
        print(f"[DEBUG] CCF Pista B falló: {_exc}")

# ── RESUMEN — ¿las RWA entran al modelo? ────────────────────
print("\n" + "─" * 62)
print("RESUMEN — SEÑAL RWA → ESTRÉS")
print("─" * 62)
try:
    print(
        f"\nVentana analizada: {df.index.min().date()} → "
        f"{df.index.max().date()} "
        f"({'recorte en ' + FECHA_CORTE if FECHA_CORTE else 'sin recorte'})"
    )
    # Línea de comparaciones múltiples: es el dato que decide.
    print(
        f"\nFDR (Benjamini-Hochberg) sobre el conjunto COMPLETO "
        f"de tests — 2 pistas, todos los lags:"
    )
    print(
        f"  {n_tests_global} tests totales | "
        f"p<{ALFA_BH} crudo: {n_sig_crudo} | "
        f"sobreviven BH global: {n_sig_bh_global}"
    )

    if df_opt.empty:
        print("  [!] Sin datos para resumir")
    else:
        _pre = df_opt[
            df_opt["precedencia"] == "precede (sig BH global)"
        ]
        _con = df_opt[
            df_opt["precedencia"] == "contemporaneo (sig)"
        ]
        _ind = df_opt[
            df_opt["precedencia"] == "indicio no confirmado"
        ]
        _deb = df_opt[
            df_opt["precedencia"] == "precede (no sig)"
        ]

        print(
            f"\nPares PRECEDEN (lag>0, |r|>{UMBRAL_R}, "
            f"BH GLOBAL q<{ALFA_BH}): {len(_pre)}"
        )
        if not _pre.empty:
            _c = [
                "pista", "etiqueta", "macro",
                "lag_optimo", "r_max", "bh_q", "n",
            ]
            print(_pre[_c].to_string(index=False))
        else:
            print(
                "  → Ninguno sobrevive la corrección global."
            )

        print(
            f"\nPares CONTEMPORÁNEOS (lag=0, |r|>{UMBRAL_R}, "
            f"p<{ALFA_BH}): {len(_con)}"
        )
        if not _con.empty:
            _c = [
                "pista", "etiqueta", "macro",
                "r_max", "p_lag_opt", "n",
            ]
            print(_con[_c].to_string(index=False))
        else:
            print("  → Ninguno contemporáneo.")

        # Trazabilidad: dirección teórica previa (refugio),
        # explícitamente NO presentados como señal.
        print(
            f"\nINDICIOS NO CONFIRMADOS — dirección teórica "
            f"previa (oro, treasuries), NO sobreviven el FDR "
            f"global: {len(_ind)}"
        )
        if not _ind.empty:
            _c = [
                "pista", "etiqueta", "macro", "lag_optimo",
                "r_max", "p_lag_opt", "bh_q", "n",
            ]
            print(_ind[_c].to_string(index=False))
            print(
                "  → Se registran para trazabilidad y para "
                "orientar el test de Granger. NO constituyen "
                "evidencia de señal."
            )
        else:
            print("  → Ninguno.")

        print(
            f"\nOtros pares con |r|>{UMBRAL_R} y lag>0, sin "
            f"significancia: {len(_deb)}"
        )
        if not _deb.empty:
            _c = [
                "pista", "etiqueta", "macro", "lag_optimo",
                "r_max", "p_lag_opt", "bh_q",
            ]
            print(_deb[_c].to_string(index=False))

        # ── VEREDICTO ───────────────────────────────────────
        # Se basa EXCLUSIVAMENTE en el FDR global. Si nada
        # sobrevive, no se afirma señal, por más que existan
        # pares con |r| alto o con dirección teórica esperada.
        # CAMBIO B3-1 (Bloque 3): cada frase sale de una condición
        # sobre las cifras. Antes: "Las RWA son candidatas" aunque
        # sobrevivieran 3/105 tests, la rama contemporánea era
        # inalcanzable y se afirmaba una comparación con las
        # on-chain que este Job no calcula.
        print("\n" + "─" * 62)
        _esperados_azar = n_tests_global * ALFA_BH
        if _pre.empty:
            print(
                "VEREDICTO: no hay precedencia RWA→estrés que "
                "sobreviva la corrección por comparaciones "
                "múltiples."
            )
            _frase_azar = (
                "en línea con lo esperable por azar"
                if n_sig_crudo <= _esperados_azar
                else "por encima de lo esperable por azar, pero "
                "ninguno sobrevive el FDR"
            )
            print(
                f"De {n_tests_global} tests, {n_sig_crudo} dan "
                f"p<{ALFA_BH} crudo (≈{_esperados_azar:.1f} "
                f"esperables por azar: {_frase_azar}) y "
                f"{n_sig_bh_global} sobreviven el FDR global."
            )
            if not _ind.empty:
                _etqs = ", ".join(sorted(set(_ind["etiqueta"])))
                print(
                    f"Los pares en dirección teórica ({_etqs}) "
                    "se reportan como INDICIOS NO CONFIRMADOS, a "
                    "validar con Granger; no como señal."
                )
            if not _con.empty:
                print(
                    f"Hay {len(_con)} par(es) con correlación "
                    "contemporánea significativa: aporte "
                    "descriptivo, no predictivo."
                )
        else:
            _etqs_pre = ", ".join(
                f"{_r.etiqueta}→{_r.macro} ({_r.pista})"
                for _r in _pre.itertuples()
            )
            _n_rwa_pre = _pre["etiqueta"].nunique()
            _n_rwa_tot = df_opt["etiqueta"].nunique()
            print(
                f"VEREDICTO: {len(_pre)} de {len(df_opt)} pares "
                f"({n_sig_bh_global} de {n_tests_global} tests) "
                "muestran precedencia RWA→estrés que sobrevive el "
                f"FDR GLOBAL: {_etqs_pre}."
            )
            print(
                f"Solo esas {_n_rwa_pre} de {_n_rwa_tot} series "
                "RWA son candidatas a entrar al modelo final; el "
                "resto no muestra precedencia."
            )
            print(
                "AVISO: el lag óptimo se eligió post-hoc → los "
                "p-values están sesgados a la baja. Confirmar "
                "con Granger multivariado antes de concluir."
            )
        print(
            "\nRECORDATORIO: series de adopción incipiente. Toda "
            "relación detectada puede reflejar tendencia común "
            "de crecimiento, no un mecanismo económico. "
            "Resultados EXPLORATORIOS."
        )
        print("─" * 62)
except Exception as _exc:
    print(f"  [DEBUG] Resumen falló: {_exc}")

"""# 6. Save"""

if SAVE_DATA:
    os.makedirs(DIR_PROCESSED, exist_ok=True)
    os.makedirs(DIR_REPORTES, exist_ok=True)
    os.makedirs(DIR_FIGURAS, exist_ok=True)

    # CAMBIO E2E-06-01: siempre publicar los CSV de ESTA corrida.
    # Una tabla sin filas se guarda con encabezados explicitos;
    # no se reciclan resultados anteriores ni se inventan datos.
    # Las tablas no vacias se guardan sin ninguna transformacion.
    _esquemas_vacios: list[tuple[str, pd.DataFrame, list[str]]] = [
        (
            "relaciones_rwa_contemporaneas.csv", df_cont,
            ["pista", "serie_x", "etiqueta", "macro", "n",
             "pearson_r", "pearson_p", "spearman_r", "spearman_p",
             "nota", "bh_q", "bh_rechaza", "ambito_bh"],
        ),
        (
            "relaciones_rwa_lags.csv", df_lags,
            ["pista", "serie_x", "etiqueta", "macro", "lag", "n",
             "pearson_r", "pearson_p", "nota", "bh_q",
             "bh_rechaza", "ambito_bh"],
        ),
        (
            "relaciones_rwa_lag_optimo.csv", df_opt,
            ["pista", "serie_x", "etiqueta", "macro", "lag_optimo",
             "r_max", "p_lag_opt", "n", "bh_q", "bh_rechaza",
             "ambito_bh", "nota", "precedencia"],
        ),
    ]
    for _nombre, _tabla, _columnas in _esquemas_vacios:
        _salida = _tabla if not _tabla.empty else pd.DataFrame(columns=_columnas)
        _ruta = os.path.join(DIR_REPORTES, _nombre)
        _salida.to_csv(_ruta, index=False)
        if DEBUG:
            print(f"[DEBUG] guardado: {_ruta} ({len(_salida)} filas)")
elif DEBUG:
    print("\n[DEBUG] SAVE_DATA=False — datos solo en memoria")

