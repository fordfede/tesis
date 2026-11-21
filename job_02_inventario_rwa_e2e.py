"""# Job 2 - Inventario y preselección de series RWA"""

# 1. Configuración

DEBUG: bool = True
SAVE_DATA: bool = True

# Límite de protocolos a descargar (None = todos los RWA).
# Solo para pruebas rápidas; en la corrida real dejar None.
LIMITE_PROTOCOLOS: int | None = None

# 2. Imports

import datetime
import os
import time
import warnings
from typing import Any

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

from google.colab import drive

drive.mount("/content/drive")

"""# 3. Paths y Constantes"""

# CAMBIO H2-3 (Job 1.b): ver nota en Job 0 (mismo patrón).
# Lectura: dataset base validado (carpeta 'expansion')
BASE_DIR_XP: str = os.environ.get(
    "TESIS_BASE_DIR", "/content/drive/MyDrive/Tesis/expansion"
)
ARCHIVO_DIARIAS_XP: str = os.path.join(
    BASE_DIR_XP, "data", "processed", "series_diarias.csv"
)

# Escritura: reportes de la expansión
BASE_DIR_EXP: str = os.environ.get(
    "TESIS_BASE_DIR", "/content/drive/MyDrive/Tesis/expansion"
)
DIR_REPORTES: str = os.path.join(
    BASE_DIR_EXP, "outputs", "reportes"
)
DIR_FIGURAS: str = os.path.join(
    DIR_REPORTES, "figuras_inventario_rwa"
)

TIMEOUT: int = 40
PAUSA: float = 1.0
VENTANA_INICIO: str = "2026-01-01"
HOY: str = datetime.date.today().isoformat()
FECHA_EXTRACCION: str = HOY

# CAMBIO I1-2 (Job 1.b): reintentos de get_json. No se agrega
# guardado de JSON crudo ni protocolos_sin_serie.csv en esta
# fase — ver nota de alcance en el cierre de F1 (colisión de
# nombre de archivo con Job 2, que escribe el mismo reporte).
MAX_REINTENTOS: int = 3

# --- Endpoints (validados en 00d) ---
LLAMA_BASE: str = "https://api.llama.fi"
LLAMA_PROTOCOLS: str = f"{LLAMA_BASE}/protocols"
LLAMA_PROTOCOL: str = f"{LLAMA_BASE}/protocol"
CATEGORIA_RWA: str = "RWA"

# ── Umbrales de decisión (ajustables y documentados) ─────────
# TVL mínimo para que una candidata sea considerada: por debajo
# de esto la serie es ruido estadístico, no señal.
UMBRAL_TVL_MIN: float = 100e6          # 100 M USD

# |corr| en log-dif por encima del cual dos candidatas se
# consideran redundantes (mismo cluster, una representante).
UMBRAL_COLINEAL: float = 0.85

# Cobertura mínima de la ventana 2026 para entrar al análisis
# de colinealidad (fracción de días de la grilla con dato).
COBERTURA_MIN: float = 0.80

# Tamaño objetivo de la preselección.
PRESEL_MIN: int = 8
PRESEL_MAX: int = 12

# Observaciones mínimas para que una correlación sea creíble.
N_MIN_CORR: int = 30

# Nivel protocolo×red: cuántos protocolos mayores se detallan.
N_PROTOCOLOS_DETALLE: int = 20

# Chains "sintéticas" de DeFiLlama a excluir del eje red: las
# claves de chainTvls que contienen "-" son agregados derivados
# (borrowed, staking, pool2), no redes.
MARCA_AGREGADO: str = "-"

# Nombre de la serie de contexto (total global).
COL_TOTAL: str = "rwa_total_global"

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


"""# 4. Funciones"""


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = TIMEOUT,
) -> Any:
    """
    GET HTTP con reintentos y manejo explícito de errores.

    CAMBIO I1-2 (Job 1.b): antes devolvía None en silencio ante
    cualquier falla. Ahora reintenta hasta MAX_REINTENTOS veces
    con espera creciente y registra con [DEBUG] el estado HTTP
    (o tipo de error) de cada intento.

    Args:
        url: URL del endpoint.
        params: Query parameters opcionales.
        timeout: Segundos de espera máxima por intento.

    Returns:
        JSON parseado, o None si los MAX_REINTENTOS intentos
        fallan.
    """
    # CAMBIO B1-2 (Bloque 1): en modo snapshot no hay red.
    if MODO_DATOS == "snapshot":
        raise RuntimeError(
            "[DEBUG][ERROR] MODO_DATOS=snapshot: llamada de red "
            f"bloqueada ({url[:55]})."
        )
    _error: str = "sin intentos"
    for _intento in range(1, MAX_REINTENTOS + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError as e:
                    _error = f"JSON inválido: {e}"
                    if DEBUG:
                        print(
                            f"[DEBUG] intento {_intento}/"
                            f"{MAX_REINTENTOS} — {_error}: "
                            f"{url[:60]}"
                        )
            else:
                _error = f"HTTP {r.status_code}"
                if DEBUG:
                    print(
                        f"[DEBUG] intento {_intento}/"
                        f"{MAX_REINTENTOS} — {_error}: "
                        f"{url[:60]}"
                    )
        except requests.Timeout:
            _error = "timeout"
            if DEBUG:
                print(
                    f"[DEBUG] intento {_intento}/"
                    f"{MAX_REINTENTOS} — timeout: {url[:60]}"
                )
        except requests.RequestException as e:
            _error = f"request error: {type(e).__name__}"  # CAMBIO B1-2: sin URL/clave
            if DEBUG:
                print(
                    f"[DEBUG] intento {_intento}/"
                    f"{MAX_REINTENTOS} — {_error}: {url[:60]}"
                )
        if _intento < MAX_REINTENTOS:
            time.sleep(PAUSA * _intento)
    if DEBUG:
        print(
            f"[DEBUG][ALERTA] get_json agotó "
            f"{MAX_REINTENTOS} intentos ({_error}): {url[:60]}"
        )
    return None


def _a_fecha(valor: Any) -> str | None:
    """
    Convierte un epoch (segundos o ms) a fecha ISO UTC.

    Args:
        valor: Epoch numérico o cadena numérica.

    Returns:
        Fecha ISO (YYYY-MM-DD) o None si no se pudo interpretar.
    """
    try:
        segundos = float(valor)
    except (TypeError, ValueError):
        return None
    if segundos > 1e11:
        segundos = segundos / 1000.0
    try:
        return datetime.datetime.fromtimestamp(
            segundos, tz=datetime.timezone.utc
        ).date().isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _slug_col(texto: str) -> str:
    """
    Normaliza un nombre de chain o tag a sufijo de columna.

    Args:
        texto: Nombre original (ej. "Treasury Bills").

    Returns:
        Sufijo en snake_case alfanumérico (ej. "treasurybills").
    """
    limpio = "".join(
        c.lower() if c.isalnum() else " " for c in texto
    )
    return "".join(limpio.split())


def listar_protocolos_rwa() -> pd.DataFrame:
    """
    Lista los protocolos de categoría RWA en DeFiLlama.

    Returns:
        DataFrame con slug, name, tags, chains, n_chains y tvl,
        ordenado por TVL descendente. Vacío si el endpoint falla.
    """
    if DEBUG:
        print(
            f"[DEBUG] /protocols → categoría '{CATEGORIA_RWA}'"
        )
    data = get_json(LLAMA_PROTOCOLS)
    if not isinstance(data, list) or not data:
        return pd.DataFrame()
    filas: list[dict[str, Any]] = []
    vistos: set[str] = set()
    for proto in data:
        if not isinstance(proto, dict):
            continue
        if proto.get("category") != CATEGORIA_RWA:
            continue
        slug = str(proto.get("slug") or "")
        if not slug or slug in vistos:
            continue                  # dedupe defensivo
        vistos.add(slug)
        chains = [str(c) for c in (proto.get("chains") or [])]
        filas.append({
            "slug": slug,
            "name": str(proto.get("name") or slug),
            "tags": [str(t) for t in (proto.get("tags") or [])],
            "chains": chains,
            "n_chains": len(chains),
            "tvl_actual": float(proto.get("tvl") or 0.0),
        })
    df = pd.DataFrame(filas)
    if not df.empty:
        df = df.sort_values("tvl_actual", ascending=False)
        df = df.reset_index(drop=True)
    return df


def descargar_series_protocolo(
    slug: str,
) -> dict[str, pd.Series]:
    """
    Descarga la serie histórica de TVL de un protocolo,
    desglosada por chain.

    Toma chainTvls[chain]["tvl"] (lista de {date,
    totalLiquidityUSD}) y excluye las claves derivadas. Si un
    día trae más de una observación, queda la última.

    Args:
        slug: Slug del protocolo en DeFiLlama.

    Returns:
        Dict chain → pd.Series (DatetimeIndex, float64).
    """
    data = get_json(f"{LLAMA_PROTOCOL}/{slug}")
    if not isinstance(data, dict):
        return {}
    chain_tvls = data.get("chainTvls")
    if not isinstance(chain_tvls, dict):
        return {}
    series: dict[str, pd.Series] = {}
    for chain, contenido in chain_tvls.items():
        nombre_chain = str(chain)
        if MARCA_AGREGADO in nombre_chain:
            continue
        if not isinstance(contenido, dict):
            continue
        observaciones = contenido.get("tvl")
        if not isinstance(observaciones, list):
            continue
        por_fecha: dict[str, float] = {}
        for obs in observaciones:
            if not isinstance(obs, dict):
                continue
            fecha = _a_fecha(obs.get("date"))
            valor = obs.get("totalLiquidityUSD")
            if fecha is None or valor is None:
                continue
            try:
                por_fecha[fecha] = float(valor)
            except (TypeError, ValueError):
                continue
        if not por_fecha:
            continue
        serie = pd.Series(por_fecha, dtype="float64")
        serie.index = pd.to_datetime(serie.index)
        series[nombre_chain] = serie.sort_index()
    return series


def _densificar(serie: pd.Series) -> pd.Series:
    """
    Lleva una serie a diaria continua dentro de SU rango
    observado, propagando con ffill. No extrapola.

    Args:
        serie: Serie con DatetimeIndex ordenado.

    Returns:
        Serie diaria entre su primera y última observación.
    """
    if serie.empty:
        return serie
    grilla = pd.date_range(
        start=serie.index.min(), end=serie.index.max(), freq="D"
    )
    return serie.reindex(grilla).ffill()


def agregar_series(componentes: list[pd.Series]) -> pd.Series:
    """
    Suma varias series de TVL en una serie agregada diaria.

    Cada componente se densifica dentro de su propio rango
    (ffill) y aporta 0 fuera de él: un protocolo que todavía no
    existía no tiene valor tokenizado. El agregado arranca el
    primer día con dato de algún componente; antes no existe
    (no se hace back-fill).

    Args:
        componentes: Series de TVL a sumar.

    Returns:
        Serie agregada diaria (float64). Vacía si no hay datos.
    """
    validos = [s for s in componentes if not s.empty]
    if not validos:
        return pd.Series(dtype="float64")
    inicio = min(s.index.min() for s in validos)
    fin = max(s.index.max() for s in validos)
    grilla = pd.date_range(start=inicio, end=fin, freq="D")
    total = pd.Series(0.0, index=grilla, dtype="float64")
    for serie in validos:
        densa = _densificar(serie).reindex(grilla).fillna(0.0)
        total = total.add(densa, fill_value=0.0)
    total.index.name = "fecha"
    return total.astype("float64")


def construir_candidatas(
    series_protocolos: dict[str, dict[str, pd.Series]],
    tags_por_slug: dict[str, list[str]],
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """
    Construye TODAS las series agregadas candidatas: por red,
    por tipo de activo y total global.

    Args:
        series_protocolos: slug → {chain → serie de TVL}.
        tags_por_slug: slug → lista de tags (tipo de activo).

    Returns:
        (series, meta):
          series: nombre_columna → serie agregada.
          meta  : DataFrame con dimension, categoria y
                  n_protocolos de cada candidata.
    """
    series: dict[str, pd.Series] = {}
    meta: list[dict[str, Any]] = []

    # --- Eje RED: todas las chains con presencia RWA ---
    chains = sorted({
        chain
        for por_chain in series_protocolos.values()
        for chain in por_chain
    })
    for chain in chains:
        componentes = [
            por_chain[chain]
            for por_chain in series_protocolos.values()
            if chain in por_chain
        ]
        agregada = agregar_series(componentes)
        if agregada.empty:
            continue
        col = f"rwa_red_{_slug_col(chain)}"
        series[col] = agregada
        meta.append({
            "serie": col,
            "dimension": "red",
            "categoria": chain,
            "n_protocolos": len(componentes),
        })

    # --- Eje TIPO: todos los tags declarados ---
    tags = sorted({
        t
        for slug in series_protocolos
        for t in tags_por_slug.get(slug, [])
    })
    for tag in tags:
        componentes = [
            serie
            for slug, por_chain in series_protocolos.items()
            if tag in tags_por_slug.get(slug, [])
            for serie in por_chain.values()
        ]
        n_proto = len([
            slug for slug in series_protocolos
            if tag in tags_por_slug.get(slug, [])
        ])
        agregada = agregar_series(componentes)
        if agregada.empty:
            continue
        col = f"rwa_tipo_{_slug_col(tag)}"
        series[col] = agregada
        meta.append({
            "serie": col,
            "dimension": "tipo",
            "categoria": tag,
            "n_protocolos": n_proto,
        })

    # --- TOTAL global (contexto/control) ---
    componentes_total = [
        serie
        for por_chain in series_protocolos.values()
        for serie in por_chain.values()
    ]
    agregada_total = agregar_series(componentes_total)
    if not agregada_total.empty:
        series[COL_TOTAL] = agregada_total
        meta.append({
            "serie": COL_TOTAL,
            "dimension": "total",
            "categoria": "global",
            "n_protocolos": len(series_protocolos),
        })
    return series, pd.DataFrame(meta)


def construir_grilla(inicio: str, fin: str) -> pd.DatetimeIndex:
    """
    Grilla diaria continua (calendario, UTC).

    Args:
        inicio: Fecha ISO de inicio.
        fin: Fecha ISO de fin.

    Returns:
        DatetimeIndex con freq="D" y name="fecha".
    """
    return pd.date_range(
        start=inicio, end=fin, freq="D", name="fecha"
    )


def alinear_a_ventana(
    serie: pd.Series,
    grilla: pd.DatetimeIndex,
) -> pd.Series:
    """
    Alinea una serie a la grilla de la ventana con ffill.

    Sin interpolación y sin back-fill: si la serie nace después
    del inicio de la ventana, esos días quedan NaN.

    Args:
        serie: Serie de historia completa.
        grilla: Grilla diaria destino.

    Returns:
        Serie sobre la grilla (float64).
    """
    if serie.empty:
        return pd.Series(dtype="float64", index=grilla)
    inicio_ext = min(serie.index.min(), grilla.min())
    grilla_ext = pd.date_range(
        start=inicio_ext, end=grilla.max(), freq="D"
    )
    alineada = serie.reindex(grilla_ext).ffill().reindex(grilla)
    alineada.index.name = "fecha"
    return alineada.astype("float64")


def atributos_candidata(
    col: str,
    serie_full: pd.Series,
    serie_ventana: pd.Series,
    grilla: pd.DatetimeIndex,
) -> dict[str, Any]:
    """
    Calcula los atributos de inventario de una candidata.

    Args:
        col: Nombre de la serie.
        serie_full: Historia completa.
        serie_ventana: Serie alineada a la ventana.
        grilla: Grilla de la ventana.

    Returns:
        Dict con TVL actual, historia, cobertura y fechas.
    """
    con_dato = serie_ventana.dropna()
    cobertura = (
        len(con_dato) / len(grilla) if len(grilla) else 0.0
    )
    return {
        "serie": col,
        "tvl_actual": (
            float(serie_full.iloc[-1])
            if not serie_full.empty else 0.0
        ),
        "tvl_max": (
            float(serie_full.max())
            if not serie_full.empty else 0.0
        ),
        "min_fecha": (
            serie_full.index.min().date().isoformat()
            if not serie_full.empty else "s/d"
        ),
        "max_fecha": (
            serie_full.index.max().date().isoformat()
            if not serie_full.empty else "s/d"
        ),
        "n_dias_historia": int(len(serie_full)),
        "n_dias_ventana": int(len(con_dato)),
        "cobertura_ventana": round(float(cobertura), 4),
        "inicia_en_ventana": bool(
            not serie_full.empty
            and serie_full.index.min() > grilla.min()
        ),
        "fecha_extraccion": FECHA_EXTRACCION,
    }


def log_diferencias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log-diferencia de las columnas estrictamente positivas.

    Las columnas con ceros o negativos se omiten: no se fuerza
    el logaritmo ni se desplaza la serie (sería inventar dato).

    Args:
        df: Dataset de niveles.

    Returns:
        DataFrame de log-diferencias.
    """
    datos: dict[str, pd.Series] = {}
    for col in df.columns:
        s = df[col].dropna()
        if s.empty or not bool((s > 0).all()):
            if DEBUG:
                print(f"[DEBUG] {col}: no positiva — sin log_dif")
            continue
        datos[col] = np.log(df[col]).diff()
    return pd.DataFrame(datos, index=df.index)


def pares_colineales(
    corr: pd.DataFrame,
    umbral: float,
) -> pd.DataFrame:
    """
    Lista los pares de series con |corr| por encima del umbral.

    Args:
        corr: Matriz de correlación.
        umbral: Umbral de |corr|.

    Returns:
        DataFrame con serie_a, serie_b y corr, de mayor a menor.
    """
    filas: list[dict[str, Any]] = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            valor = corr.loc[a, b]
            if pd.isna(valor):
                continue
            if abs(float(valor)) > umbral:
                filas.append({
                    "serie_a": a,
                    "serie_b": b,
                    "corr": round(float(valor), 4),
                })
    df = pd.DataFrame(filas)
    if not df.empty:
        df = df.reindex(
            df["corr"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)
    return df


def clusters_colinealidad(
    corr: pd.DataFrame,
    umbral: float,
) -> dict[str, int]:
    """
    Agrupa series en clusters de redundancia.

    Dos series quedan en el mismo cluster si |corr| > umbral;
    la relación se propaga transitivamente (union-find), de modo
    que un grupo redundante queda representado por un único id.

    Args:
        corr: Matriz de correlación en log-diferencia.
        umbral: Umbral de |corr|.

    Returns:
        Dict serie → id de cluster (int).
    """
    cols = list(corr.columns)
    padre: dict[str, str] = {c: c for c in cols}

    def buscar(x: str) -> str:
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            valor = corr.loc[a, b]
            if pd.isna(valor) or abs(float(valor)) <= umbral:
                continue
            ra, rb = buscar(a), buscar(b)
            if ra != rb:
                padre[ra] = rb

    raices = sorted({buscar(c) for c in cols})
    id_por_raiz = {r: i for i, r in enumerate(raices)}
    return {c: id_por_raiz[buscar(c)] for c in cols}


def elegir_representante(
    candidatas: pd.DataFrame,
) -> str:
    """
    Elige la representante de un cluster.

    Criterio: mayor TVL actual; a igualdad de TVL, mayor
    historia (más días de serie).

    Args:
        candidatas: Filas del inventario de un mismo cluster.

    Returns:
        Nombre de la serie representante.
    """
    ordenadas = candidatas.sort_values(
        ["tvl_actual", "n_dias_historia"], ascending=False
    )
    return str(ordenadas.iloc[0]["serie"])


def preseleccionar(
    inventario: pd.DataFrame,
    clusters: dict[str, int],
    presel_min: int = PRESEL_MIN,
    presel_max: int = PRESEL_MAX,
) -> pd.DataFrame:
    """
    Propone el subconjunto diverso y no redundante.

    Algoritmo:
      1. De cada cluster de colinealidad se conserva UNA
         representante (mayor TVL, desempate por historia); las
         demás quedan marcadas REDUNDANTE con su representante.
      2. Entre las representantes se alterna la selección entre
         el eje TIPO y el eje RED, tomando en cada turno la de
         mayor TVL disponible. Así el subconjunto cubre
         dinámicas distintas (refugio soberano, oro, crédito
         privado, equity) y varias redes, en vez de llenarse con
         las mayores de un solo eje.
      3. El total global entra primero como serie de contexto,
         salvo que haya quedado en el mismo cluster que una
         componente: en ese caso el paso 1 ya lo resolvió y no
         se duplica el anidamiento.
      4. El corte es presel_max; si hay menos representantes que
         presel_min se reporta el faltante.

    Args:
        inventario: Inventario filtrado (pasa tamaño+cobertura).
        clusters: Serie → id de cluster.
        presel_min: Tamaño mínimo deseado.
        presel_max: Tamaño máximo del subconjunto.

    Returns:
        DataFrame con estado ("SELECCIONADA"/"REDUNDANTE"/
        "NO_SELECCIONADA"), representada_por y justificacion.
    """
    base = inventario.copy()
    base["cluster"] = base["serie"].map(clusters)
    base["estado"] = "NO_SELECCIONADA"
    base["representada_por"] = ""
    base["justificacion"] = ""

    # 1. Representante por cluster
    representantes: list[str] = []
    for cid, grupo in base.groupby("cluster", dropna=False):
        rep = elegir_representante(grupo)
        representantes.append(rep)
        for _, fila in grupo.iterrows():
            nombre = str(fila["serie"])
            if nombre == rep:
                continue
            idx = index_serie(base, nombre)
            base.loc[idx, "estado"] = "REDUNDANTE"
            base.loc[idx, "representada_por"] = rep
            base.loc[idx, "justificacion"] = (
                f"|corr| log-dif > {UMBRAL_COLINEAL} con el "
                f"cluster {cid}; representada por {rep}"
            )

    # 2. Selección alternando ejes, por TVL descendente
    reps = base[base["serie"].isin(representantes)]
    pool_total = [
        s for s in reps.loc[
            reps["dimension"] == "total", "serie"
        ]
    ]
    def _pool(dim: str) -> list[str]:
        sub = reps[reps["dimension"] == dim]
        sub = sub.sort_values("tvl_actual", ascending=False)
        return [str(s) for s in sub["serie"]]

    pool_tipo = _pool("tipo")
    pool_red = _pool("red")
    seleccion: list[str] = []
    seleccion.extend(pool_total[:1])
    turno_tipo = True
    while len(seleccion) < presel_max and (pool_tipo or pool_red):
        pool = pool_tipo if turno_tipo else pool_red
        if not pool:
            pool = pool_red if turno_tipo else pool_tipo
        if not pool:
            break
        seleccion.append(pool.pop(0))
        turno_tipo = not turno_tipo

    # 3. Marcado y justificación
    for nombre in seleccion:
        idx = index_serie(base, nombre)
        fila = base.loc[idx]
        base.loc[idx, "estado"] = "SELECCIONADA"
        base.loc[idx, "justificacion"] = (
            f"representante de su cluster; eje="
            f"{fila['dimension']}:{fila['categoria']}; "
            f"TVL={fila['tvl_actual']:,.0f} USD; "
            f"historia={fila['n_dias_historia']} días; "
            f"cobertura ventana="
            f"{fila['cobertura_ventana'] * 100:.0f}%"
        )
    no_sel = (
        (base["estado"] == "NO_SELECCIONADA")
        & (base["serie"].isin(representantes))
    )
    base.loc[no_sel, "justificacion"] = (
        f"no redundante, pero fuera del corte de {presel_max} "
        "por menor TVL en su eje"
    )
    return base


def index_serie(df: pd.DataFrame, nombre: str) -> Any:
    """
    Devuelve el índice de la fila de una serie del inventario.

    Args:
        df: Inventario.
        nombre: Nombre de la serie.

    Returns:
        Índice de pandas de esa fila.
    """
    return df.index[df["serie"] == nombre][0]


def heatmap_colinealidad(
    corr: pd.DataFrame,
    titulo: str,
    fname: str,
) -> None:
    """
    Heatmap de la matriz de colinealidad.

    Args:
        corr: Matriz de correlación.
        titulo: Título de la figura.
        fname: Nombre del archivo PNG.
    """
    if corr.empty:
        if DEBUG:
            print(f"[DEBUG] {titulo}: matriz vacía")
        return
    lado = max(6.0, 0.9 * len(corr.columns) + 3.0)
    fig, ax = plt.subplots(figsize=(lado, lado * 0.8))
    mask = np.zeros_like(corr.values, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax,
        annot_kws={"size": 7}, linewidths=0.4,
    )
    ax.set_title(titulo, fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    plt.tight_layout()
    if SAVE_DATA:
        os.makedirs(DIR_FIGURAS, exist_ok=True)
        ruta = os.path.join(DIR_FIGURAS, fname)
        plt.savefig(ruta, dpi=150, bbox_inches="tight")
        if DEBUG:
            print(f"[DEBUG] figura: {ruta}")
    plt.show()
    plt.close()


"""# 5. Proceso"""

# CAMBIO B1-2 (Bloque 1): en modo snapshot no se consulta
# DeFiLlama: se restauran los reportes congelados y el Job termina.
if MODO_DATOS == "snapshot":
    restaurar_desde_snapshot("Job 2", [])
    print("[DEBUG] Job 2: MODO_DATOS=snapshot — 0 llamadas de red. Fin.")
    raise SystemExit(0)

# ── 5.0 Ventana de análisis (del dataset base de 'expansion') ─────
if DEBUG:
    print("\n[DEBUG] === 5.0 Ventana de análisis ===")

# CAMBIO E2E-02-01: validar la entrada antes de las llamadas a DeFiLlama.
# Si falta o está dañada, abortar con el nombre del archivo y el Job
# productor. No sustituir silenciosamente su ventana por VENTANA_INICIO-HOY.
if not os.path.isfile(ARCHIVO_DIARIAS_XP):
    raise FileNotFoundError(
        f"[DEBUG][ERROR] Job 2: falta {ARCHIVO_DIARIAS_XP}. "
        "Ejecutar Job 1 y verificar series_diarias.csv antes del Job 2."
    )
try:
    df_base = pd.read_csv(
        ARCHIVO_DIARIAS_XP,
        parse_dates=["fecha"],
        index_col="fecha",
    )
except (OSError, ValueError, TypeError, pd.errors.ParserError) as _exc:
    raise RuntimeError(
        f"[DEBUG][ERROR] Job 2: no se puede leer {ARCHIVO_DIARIAS_XP}. "
        f"Debe ser un CSV válido con columna 'fecha'. Detalle: {_exc}"
    ) from _exc

if df_base.empty:
    raise RuntimeError(
        f"[DEBUG][ERROR] Job 2: {ARCHIVO_DIARIAS_XP} está vacío. "
        "Revisar la salida del Job 1; no se consultó DeFiLlama."
    )
if (
    not isinstance(df_base.index, pd.DatetimeIndex)
    or df_base.index.isna().any()
):
    raise RuntimeError(
        f"[DEBUG][ERROR] Job 2: columna 'fecha' inválida en "
        f"{ARCHIVO_DIARIAS_XP}. Se requieren fechas parseables "
        "sin valores nulos; no se consultó DeFiLlama."
    )

ventana_inicio: str = df_base.index.min().date().isoformat()
ventana_fin: str = df_base.index.max().date().isoformat()
if DEBUG:
    print(
        f"[DEBUG] dataset base: {df_base.shape} — "
        f"{ventana_inicio} → {ventana_fin}"
    )

grilla: pd.DatetimeIndex = construir_grilla(
    ventana_inicio, ventana_fin
)

# ── 5.1 PARTE 1: inventario de protocolos ────────────────────
if DEBUG:
    print("\n[DEBUG] === 5.1 PARTE 1: catálogo de protocolos ===")

protocolos: pd.DataFrame = listar_protocolos_rwa()
if protocolos.empty:
    raise RuntimeError(
        "No se pudo listar protocolos RWA en DeFiLlama: "
        "sin catálogo no hay inventario posible."
    )
if LIMITE_PROTOCOLOS is not None:
    protocolos = protocolos.head(LIMITE_PROTOCOLOS)

if DEBUG:
    print(f"[DEBUG] protocolos RWA: {len(protocolos)}")
    print(
        f"[DEBUG] TVL catálogo: "
        f"{protocolos['tvl_actual'].sum():,.0f} USD"
    )

series_protocolos: dict[str, dict[str, pd.Series]] = {}
for _, _fila in protocolos.iterrows():
    _slug = str(_fila["slug"])
    try:
        _series = descargar_series_protocolo(_slug)
    except Exception as _e:
        _series = {}
        if DEBUG:
            print(f"[DEBUG] error en {_slug}: {_e}")
    if _series:
        series_protocolos[_slug] = _series
    time.sleep(PAUSA)

if DEBUG:
    print(
        f"[DEBUG] protocolos con serie: "
        f"{len(series_protocolos)}/{len(protocolos)}"
    )

# Nivel PROTOCOLO: atributos de historia
filas_proto: list[dict[str, Any]] = []
for _, _fila in protocolos.iterrows():
    _slug = str(_fila["slug"])
    _por_chain = series_protocolos.get(_slug, {})
    _todas = list(_por_chain.values())
    _inicio = (
        min(s.index.min() for s in _todas) if _todas else None
    )
    _fin = (
        max(s.index.max() for s in _todas) if _todas else None
    )
    filas_proto.append({
        "slug": _slug,
        "name": str(_fila["name"]),
        "tags": ",".join(_fila["tags"]),
        "chains": ",".join(_fila["chains"]),
        "n_chains": int(_fila["n_chains"]),
        "tvl_actual": float(_fila["tvl_actual"]),
        "con_serie": bool(_todas),
        "min_fecha": (
            _inicio.date().isoformat() if _inicio else "s/d"
        ),
        "max_fecha": (
            _fin.date().isoformat() if _fin else "s/d"
        ),
        "n_dias_historia": (
            int((_fin - _inicio).days) + 1
            if _inicio is not None and _fin is not None else 0
        ),
        "fecha_extraccion": FECHA_EXTRACCION,
    })
df_protocolos: pd.DataFrame = pd.DataFrame(filas_proto)

# Nivel PROTOCOLO × RED (solo los mayores, para no inflar)
filas_pr: list[dict[str, Any]] = []
for _, _fila in df_protocolos.head(N_PROTOCOLOS_DETALLE).iterrows():
    _slug = str(_fila["slug"])
    for _chain, _serie in series_protocolos.get(_slug, {}).items():
        filas_pr.append({
            "slug": _slug,
            "name": _fila["name"],
            "chain": _chain,
            "tags": _fila["tags"],
            "tvl_ultimo": float(_serie.iloc[-1]),
            "min_fecha": _serie.index.min().date().isoformat(),
            "max_fecha": _serie.index.max().date().isoformat(),
            "n_obs": int(len(_serie)),
            "fecha_extraccion": FECHA_EXTRACCION,
        })
df_proto_red: pd.DataFrame = pd.DataFrame(filas_pr)

if DEBUG:
    print(f"[DEBUG] nivel protocolo: {df_protocolos.shape}")
    print(f"[DEBUG] nivel protocolo×red: {df_proto_red.shape}")

# ── 5.2 Series agregadas candidatas ──────────────────────────
if DEBUG:
    print("\n[DEBUG] === 5.2 Series candidatas ===")

tags_por_slug: dict[str, list[str]] = {
    str(f["slug"]): list(f["tags"])
    for _, f in protocolos.iterrows()
}

series_candidatas, df_meta = construir_candidatas(
    series_protocolos, tags_por_slug
)

if DEBUG:
    print(
        f"[DEBUG] candidatas construidas: "
        f"{len(series_candidatas)} "
        f"(red={int((df_meta['dimension'] == 'red').sum())}, "
        f"tipo={int((df_meta['dimension'] == 'tipo').sum())}, "
        f"total=1)"
    )

# Alineación a la ventana y atributos de inventario
series_ventana: dict[str, pd.Series] = {}
filas_inv: list[dict[str, Any]] = []
for _col, _serie in series_candidatas.items():
    _alineada = alinear_a_ventana(_serie, grilla)
    series_ventana[_col] = _alineada
    filas_inv.append(
        atributos_candidata(_col, _serie, _alineada, grilla)
    )

inventario: pd.DataFrame = pd.DataFrame(filas_inv)
inventario = inventario.merge(df_meta, on="serie", how="left")
inventario = inventario.sort_values(
    "tvl_actual", ascending=False
).reset_index(drop=True)

df_niveles: pd.DataFrame = pd.DataFrame(
    series_ventana, index=grilla
)

if DEBUG:
    _inv_cols = [
        "serie", "dimension", "categoria", "tvl_actual",
        "n_dias_historia", "cobertura_ventana", "min_fecha",
    ]
    print(inventario[_inv_cols].head(30).to_string(index=False))

# ── 5.3 PARTE 2: filtro por tamaño y cobertura ───────────────
if DEBUG:
    print("\n[DEBUG] === 5.3 PARTE 2: filtro tamaño/cobertura ===")

inventario["pasa_tamano"] = (
    inventario["tvl_actual"] >= UMBRAL_TVL_MIN
)
inventario["pasa_cobertura"] = (
    inventario["cobertura_ventana"] >= COBERTURA_MIN
)
inventario["motivo_descarte"] = ""
inventario.loc[~inventario["pasa_tamano"], "motivo_descarte"] = (
    f"TVL < {UMBRAL_TVL_MIN:,.0f} USD (ruido estadístico)"
)
_sin_cob = (
    inventario["pasa_tamano"] & ~inventario["pasa_cobertura"]
)
inventario.loc[_sin_cob, "motivo_descarte"] = (
    f"cobertura de ventana < {COBERTURA_MIN:.0%} "
    "(serie nace dentro de la ventana; no se rellena)"
)

candidatas_ok: list[str] = inventario.loc[
    inventario["pasa_tamano"] & inventario["pasa_cobertura"],
    "serie",
].tolist()

if DEBUG:
    print(
        f"[DEBUG] pasan tamaño: "
        f"{int(inventario['pasa_tamano'].sum())}/"
        f"{len(inventario)}"
    )
    print(
        f"[DEBUG] pasan tamaño+cobertura: {len(candidatas_ok)}"
    )
    _desc = inventario[inventario["motivo_descarte"] != ""]
    if not _desc.empty:
        print("[DEBUG] descartadas (primeras 20):")
        print(
            _desc[["serie", "tvl_actual", "motivo_descarte"]]
            .head(20).to_string(index=False)
        )

# ── 5.4 PARTE 3: colinealidad en log-diferencia ──────────────
if DEBUG:
    print("\n[DEBUG] === 5.4 PARTE 3: colinealidad log-dif ===")

corr_logdif: pd.DataFrame = pd.DataFrame()
df_pares: pd.DataFrame = pd.DataFrame()
clusters: dict[str, int] = {}
try:
    df_ld = log_diferencias(df_niveles[candidatas_ok])
    # Solo series con suficientes observaciones para correlar
    _validas = [
        c for c in df_ld.columns
        if df_ld[c].notna().sum() >= N_MIN_CORR
        and float(df_ld[c].std()) > 0
    ]
    _descartadas_n = [
        c for c in df_ld.columns if c not in _validas
    ]
    if DEBUG and _descartadas_n:
        print(
            f"[DEBUG] sin n suficiente para correlacionar "
            f"(<{N_MIN_CORR}): {_descartadas_n}"
        )
    corr_logdif = df_ld[_validas].corr(method="pearson")
    df_pares = pares_colineales(corr_logdif, UMBRAL_COLINEAL)
    clusters = clusters_colinealidad(
        corr_logdif, UMBRAL_COLINEAL
    )
    if DEBUG:
        print(
            f"[DEBUG] matriz log-dif: {corr_logdif.shape} | "
            f"pares |corr|>{UMBRAL_COLINEAL}: {len(df_pares)}"
        )
        if not df_pares.empty:
            print(df_pares.head(20).to_string(index=False))
        _n_clusters = len(set(clusters.values()))
        print(
            f"[DEBUG] clusters de redundancia: {_n_clusters} "
            f"sobre {len(clusters)} series"
        )
except Exception as _exc:
    if DEBUG:
        print(f"[DEBUG] PARTE 3 falló: {_exc}")

# ── 5.5 PARTE 4: preselección por diversidad ─────────────────
if DEBUG:
    print("\n[DEBUG] === 5.5 PARTE 4: preselección ===")

preseleccion: pd.DataFrame = pd.DataFrame()
try:
    inv_ok = inventario[
        inventario["serie"].isin(clusters.keys())
    ].reset_index(drop=True)
    preseleccion = preseleccionar(
        inv_ok, clusters, PRESEL_MIN, PRESEL_MAX
    )
    _sel = preseleccion[
        preseleccion["estado"] == "SELECCIONADA"
    ]["serie"].tolist()
    if len(_sel) < PRESEL_MIN and DEBUG:
        print(
            f"[DEBUG] AVISO: solo {len(_sel)} series no "
            f"redundantes disponibles (mínimo deseado "
            f"{PRESEL_MIN}): revisar UMBRAL_TVL_MIN o "
            "UMBRAL_COLINEAL"
        )
    if _sel and not corr_logdif.empty:
        heatmap_colinealidad(
            corr_logdif.loc[_sel, _sel],
            "Colinealidad en log-diferencia — preselección",
            "colinealidad_preseleccion_logdif.png",
        )
except Exception as _exc:
    if DEBUG:
        print(f"[DEBUG] PARTE 4 falló: {_exc}")

# ── 5.6 Reporte de decisión ──────────────────────────────────
print("\n[RWA] === Preselección propuesta (a validar) ===")
print(f"[RWA] fecha de extracción: {FECHA_EXTRACCION}")
print(f"[RWA] ventana: {ventana_inicio} → {ventana_fin}")
print(
    f"[RWA] umbrales — TVL mín: {UMBRAL_TVL_MIN:,.0f} USD | "
    f"colinealidad: |corr| log-dif > {UMBRAL_COLINEAL} | "
    f"cobertura mín: {COBERTURA_MIN:.0%}"
)
if preseleccion.empty:
    print("[RWA] sin preselección disponible")
else:
    _sel = preseleccion[preseleccion["estado"] == "SELECCIONADA"]
    _red = preseleccion[preseleccion["estado"] == "REDUNDANTE"]
    _no = preseleccion[
        preseleccion["estado"] == "NO_SELECCIONADA"
    ]
    print(f"\n[RWA] SELECCIONADAS ({len(_sel)}):")
    for _, _f in _sel.iterrows():
        print(f"  - {_f['serie']}: {_f['justificacion']}")
    print(f"\n[RWA] REDUNDANTES ({len(_red)}):")
    for _, _f in _red.iterrows():
        print(
            f"  - {_f['serie']} → representada por "
            f"{_f['representada_por']}"
        )
    if not _no.empty:
        print(f"\n[RWA] FUERA DEL CORTE ({len(_no)}):")
        for _, _f in _no.iterrows():
            print(f"  - {_f['serie']}: {_f['justificacion']}")
    _tipos = sorted(set(_sel.loc[
        _sel["dimension"] == "tipo", "categoria"
    ]))
    _redes = sorted(set(_sel.loc[
        _sel["dimension"] == "red", "categoria"
    ]))
    print(
        f"\n[RWA] diversidad — tipos: {_tipos or 'ninguno'}"
    )
    print(f"[RWA] diversidad — redes: {_redes or 'ninguna'}")
print(
    "\n[RWA] NOTA: los tags de tipo NO son exclusivos; las "
    "series por tipo no particionan el total global. El "
    "anidamiento se controla por correlación empírica en "
    "log-dif, no por aritmética de sumas."
)
print(
    "[RWA] Esta es una PROPUESTA: la decisión del set final "
    "queda a criterio del autor sobre este reporte."
)

"""# 6. Save"""

if SAVE_DATA:
    os.makedirs(DIR_REPORTES, exist_ok=True)
    os.makedirs(DIR_FIGURAS, exist_ok=True)

    _guardados: list[str] = []

    if not inventario.empty:
        _ruta = os.path.join(DIR_REPORTES, "inventario_rwa.csv")
        inventario.to_csv(_ruta, index=False)
        _guardados.append(_ruta)

    if not df_protocolos.empty:
        _ruta = os.path.join(
            DIR_REPORTES, "inventario_rwa_protocolos.csv"
        )
        df_protocolos.to_csv(_ruta, index=False)
        _guardados.append(_ruta)

    if not df_proto_red.empty:
        _ruta = os.path.join(
            DIR_REPORTES, "inventario_rwa_proto_red.csv"
        )
        df_proto_red.to_csv(_ruta, index=False)
        _guardados.append(_ruta)

    if not corr_logdif.empty:
        _ruta = os.path.join(
            DIR_REPORTES, "colinealidad_rwa_logdif.csv"
        )
        corr_logdif.to_csv(_ruta)
        _guardados.append(_ruta)

    if not df_pares.empty:
        _ruta = os.path.join(
            DIR_REPORTES, "colinealidad_rwa_pares.csv"
        )
        df_pares.to_csv(_ruta, index=False)
        _guardados.append(_ruta)

    if not preseleccion.empty:
        _ruta = os.path.join(
            DIR_REPORTES, "preseleccion_rwa.csv"
        )
        preseleccion.to_csv(_ruta, index=False)
        _guardados.append(_ruta)

    if DEBUG:
        for _g in _guardados:
            print(f"[DEBUG] guardado: {_g}")
    congelar_snapshot(_guardados, "Job 2")  # CAMBIO B1-2
elif DEBUG:
    print("\n[DEBUG] SAVE_DATA=False — datos solo en memoria")

