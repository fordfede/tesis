"""# Job 3 - Adquisición de series RWA (tokenización por red y tipo)"""

# 1. Configuración

DEBUG: bool = True
SAVE_DATA: bool = True

# Une las series RWA al dataset diario existente (left join).
UNIR_A_DATASET: bool = True
# Límite de protocolos a descargar (None = todos los RWA).
# Solo para pruebas rápidas; en la corrida real dejar None.
LIMITE_PROTOCOLOS: int | None = None

# 2. Imports

import datetime
import hashlib  # CAMBIO H3-3 (Job 2): huellas SHA-256
import json  # CAMBIO I1-3 (Job 2): guardado de JSON crudo
import os
import time
from typing import Any

import pandas as pd
import requests

from google.colab import drive
drive.mount('/content/drive')

# 3. Paths y Constantes

# CAMBIO H2-4 (Job 2): ver nota en Job 0 (mismo patrón).
# Lectura: dataset base validado (carpeta 'expansion')
BASE_DIR_XP: str = os.environ.get(
    "TESIS_BASE_DIR", "/content/drive/MyDrive/Tesis/expansion"
)
DIR_PROCESSED_XP: str = os.path.join(
    BASE_DIR_XP, "data", "processed"
)

# Escritura: todo lo nuevo va a la expansión
BASE_DIR: str = os.environ.get(
    "TESIS_BASE_DIR", "/content/drive/MyDrive/Tesis/expansion"
)
DIR_PROCESSED: str = os.path.join(BASE_DIR, "data", "processed")
DIR_REPORTES: str = os.path.join(BASE_DIR, "outputs", "reportes")

ARCHIVO_DIARIAS: str = os.path.join(
    DIR_PROCESSED_XP, "series_diarias.csv"
)
ARCHIVO_LARGAS: str = os.path.join(
    DIR_PROCESSED_XP, "series_largas.csv"
)
ARCHIVO_RWA: str = os.path.join(
    DIR_PROCESSED, "series_rwa.csv"
)
ARCHIVO_RWA_LARGAS: str = os.path.join(
    DIR_PROCESSED, "series_rwa_largas.csv"
)
ARCHIVO_DIARIAS_EXT: str = os.path.join(
    DIR_PROCESSED, "series_diarias_ext.csv"
)
ARCHIVO_LARGAS_EXT: str = os.path.join(
    DIR_PROCESSED, "series_largas_ext.csv"
)
ARCHIVO_COBERTURA: str = os.path.join(
    DIR_REPORTES, "cobertura_rwa.csv"
)
# CAMBIO H3-3 (Job 2): huellas SHA-256 de los CSV guardados.
ARCHIVO_HUELLAS: str = os.path.join(
    DIR_REPORTES, "huellas_datos.csv"
)
# CAMBIO I1-3 (Job 2): diagnóstico de la descarga por protocolo.
ARCHIVO_PROTOCOLOS_SIN_SERIE: str = os.path.join(
    DIR_REPORTES, "protocolos_sin_serie.csv"
)
ARCHIVO_RESUMEN_CATALOGO: str = os.path.join(
    DIR_REPORTES, "resumen_catalogo_rwa.csv"
)

TIMEOUT: int = 40
PAUSA: float = 1.0            # segundos entre requests
VENTANA_INICIO: str = "2026-01-01"
HOY: str = datetime.date.today().isoformat()
FECHA_EXTRACCION: str = HOY

# CAMBIO I1-3 (Job 2): reintentos de get_json y carpeta donde se
# congela cada respuesta cruda de DeFiLlama antes de procesarla.
MAX_REINTENTOS: int = 3
DIR_RAW: str = os.path.join(
    BASE_DIR, "data", "raw", FECHA_EXTRACCION
)
# Proporción mínima de protocolos RWA con serie descargable.
# Si la corrida queda por debajo, no se corta: se deja un aviso
# fuerte en [DEBUG] y en ARCHIVO_RESUMEN_CATALOGO.
MIN_PROP_PROTOCOLOS_CON_SERIE: float = 0.90

# --- Endpoints base (validados en 00d) ---
LLAMA_BASE: str = "https://api.llama.fi"
LLAMA_PROTOCOLS: str = f"{LLAMA_BASE}/protocols"
LLAMA_PROTOCOL: str = f"{LLAMA_BASE}/protocol"       # /{slug}

CATEGORIA_RWA: str = "RWA"

# GRUPO 1 — columna → chains DeFiLlama que la componen.
# "XRPL EVM" es la sidechain EVM de XRPL: se suma a Ripple por
# ser la misma red del modelo. Quitarla de la lista si se quiere
# el XRPL nativo puro.
REDES_OBJETIVO: dict[str, list[str]] = {
    "rwa_red_stellar": ["Stellar"],
    "rwa_red_ripple": ["Ripple", "XRPL EVM"],
    "rwa_red_algorand": ["Algorand"],
}

# GRUPO 2 — columna → tags DeFiLlama (nombres exactos del API).
TIPOS_OBJETIVO: dict[str, list[str]] = {
    "rwa_tipo_treasurybills": ["Treasury Bills"],
    "rwa_tipo_commodities": ["Commodities"],
    "rwa_tipo_privatecredit": ["Private Credit"],
    "rwa_tipo_stocksetfs": ["Stocks & ETFs"],
}

# GRUPO 3 — contexto.
COL_TOTAL: str = "rwa_total_global"

# Orden final de columnas del dataset RWA.
COLUMNAS_RWA: list[str] = (
    list(REDES_OBJETIVO) + list(TIPOS_OBJETIVO) + [COL_TOTAL]
)

# Sufijos de chainTvls que son agregados derivados, no chains
# reales (borrowed, staking, pool2, ...). Se excluyen.
MARCA_AGREGADO: str = "-"

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


def registrar_huella(ruta: str, archivo_huellas: str) -> None:
    """
    Calcula el SHA-256 de un CSV ya guardado y agrega una fila
    al archivo de huellas de datos (no sobrescribe filas previas).

    CAMBIO H3-3 (Job 2).

    Args:
        ruta: Ruta del CSV ya guardado en disco.
        archivo_huellas: Ruta del CSV de huellas (se crea si no
            existe; si existe, se agrega una fila al final).
    """
    if not os.path.exists(ruta):
        if DEBUG:
            print(f"[DEBUG] huella: no existe {ruta}, se omite")
        return
    try:
        with open(ruta, "rb") as _f:
            _sha256 = hashlib.sha256(_f.read()).hexdigest()
    except OSError as e:
        if DEBUG:
            print(
                f"[DEBUG][ALERTA] huella: error leyendo "
                f"{ruta}: {e}"
            )
        return
    _fila = pd.DataFrame([{
        "archivo": os.path.basename(ruta),
        "sha256": _sha256,
        "fecha": datetime.date.today().isoformat(),
    }])
    _existe = os.path.exists(archivo_huellas)
    _fila.to_csv(
        archivo_huellas, mode="a", header=not _existe,
        index=False,
    )
    if DEBUG:
        print(
            f"[DEBUG] huella registrada: {ruta} → "
            f"{_sha256[:12]}..."
        )


def _guardar_json_crudo(data: Any, ruta: str) -> None:
    """
    Guarda una respuesta JSON cruda en disco antes de procesarla.

    CAMBIO I1-3 (Job 2).

    Args:
        data: Objeto ya parseado (dict/list) a serializar.
        ruta: Ruta completa del archivo .json de destino.
    """
    try:
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        with open(ruta, "w", encoding="utf-8") as _f:
            json.dump(data, _f)
    except OSError as e:
        if DEBUG:
            print(
                f"[DEBUG][ALERTA] no se pudo guardar crudo "
                f"{ruta}: {e}"
            )


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = TIMEOUT,
    ruta_crudo: str | None = None,
) -> Any:
    """
    GET HTTP con reintentos y manejo explícito de errores.

    CAMBIO I1-3 (Job 2): antes devolvía None en silencio ante
    cualquier falla. Ahora reintenta hasta MAX_REINTENTOS veces
    con espera creciente, registra con [DEBUG] el estado HTTP (o
    tipo de error) de cada intento, y si se indica ruta_crudo
    guarda la respuesta cruda en disco antes de devolverla.

    Args:
        url: URL del endpoint.
        params: Query parameters opcionales.
        timeout: Segundos de espera máxima por intento.
        ruta_crudo: Si se indica, ruta donde guardar el JSON
            crudo (antes de cualquier procesamiento posterior).

    Returns:
        Objeto Python (dict/list) parseado del JSON, o None si
        los MAX_REINTENTOS intentos fallan.
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
                    data = r.json()
                except ValueError as e:
                    _error = f"JSON inválido: {e}"
                    if DEBUG:
                        print(
                            f"[DEBUG] intento {_intento}/"
                            f"{MAX_REINTENTOS} — {_error}: "
                            f"{url[:60]}"
                        )
                else:
                    if ruta_crudo:
                        _guardar_json_crudo(data, ruta_crudo)
                    return data
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
    Convierte un timestamp epoch (segundos o ms) a fecha ISO UTC.

    Args:
        valor: Epoch numérico o cadena numérica.

    Returns:
        Fecha ISO (YYYY-MM-DD) o None si no se pudo interpretar.
    """
    try:
        segundos = float(valor)
    except (TypeError, ValueError):
        return None
    if segundos > 1e11:           # epoch en milisegundos
        segundos = segundos / 1000.0
    try:
        return datetime.datetime.fromtimestamp(
            segundos, tz=datetime.timezone.utc
        ).date().isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def listar_protocolos_rwa() -> pd.DataFrame:
    """
    Lista los protocolos de categoría RWA en DeFiLlama.

    Returns:
        DataFrame con columnas slug, name, tags (list[str]),
        chains (list[str]) y tvl, ordenado por TVL descendente.
        Vacío si el endpoint falla.
    """
    if DEBUG:
        print(
            f"[DEBUG] /protocols → categoría '{CATEGORIA_RWA}'"
        )
    # CAMBIO I1-3 (Job 2): se congela la respuesta cruda
    # antes de procesarla.
    _ruta_crudo = os.path.join(DIR_RAW, "protocols.json")
    data = get_json(LLAMA_PROTOCOLS, ruta_crudo=_ruta_crudo)
    if not isinstance(data, list) or not data:
        if DEBUG:
            print("[DEBUG] /protocols: respuesta vacía o inválida")
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
            continue          # dedupe: no contar dos veces
        vistos.add(slug)
        filas.append({
            "slug": slug,
            "name": str(proto.get("name") or slug),
            "tags": [str(t) for t in (proto.get("tags") or [])],
            "chains": [
                str(c) for c in (proto.get("chains") or [])
            ],
            "tvl": float(proto.get("tvl") or 0.0),
        })
    df = pd.DataFrame(filas)
    if not df.empty:
        df = df.sort_values("tvl", ascending=False)
        df = df.reset_index(drop=True)
    return df


def descargar_series_protocolo(
    slug: str,
) -> tuple[dict[str, pd.Series], str]:
    """
    Descarga la serie histórica de TVL de un protocolo,
    desglosada por chain.

    Toma chainTvls[chain]["tvl"], que es una lista de
    {date, totalLiquidityUSD}. Excluye las claves derivadas
    (las que contienen "-": borrowed, staking, pool2, ...).
    Si un día trae más de una observación, queda la última.

    CAMBIO I1-3 (Job 2): antes devolvía solo el dict de series;
    ahora también el motivo de la ausencia de serie, para poder
    diagnosticar la descarga por protocolo (protocolos_sin_serie.csv).

    Args:
        slug: Slug del protocolo en DeFiLlama.

    Returns:
        (series, motivo): dict chain → pd.Series (vacío si no
        hay datos) y motivo ("" si se obtuvo al menos una serie,
        o una de: "HTTP", "JSON sin chainTvls", "chainTvls sin
        clave 'tvl'", "todas las claves descartadas por contener
        '-'", "sin observaciones utilizables en 'tvl'").
    """
    _ruta_crudo = os.path.join(DIR_RAW, f"protocol_{slug}.json")
    data = get_json(
        f"{LLAMA_PROTOCOL}/{slug}", ruta_crudo=_ruta_crudo
    )
    if not isinstance(data, dict):
        return {}, "HTTP"
    chain_tvls = data.get("chainTvls")
    if not isinstance(chain_tvls, dict):
        return {}, "JSON sin chainTvls"
    series: dict[str, pd.Series] = {}
    _hubo_clave_no_agregada = False
    _hubo_clave_tvl = False
    for chain, contenido in chain_tvls.items():
        nombre_chain = str(chain)
        if MARCA_AGREGADO in nombre_chain:
            continue
        _hubo_clave_no_agregada = True
        if not isinstance(contenido, dict):
            continue
        observaciones = contenido.get("tvl")
        if not isinstance(observaciones, list):
            continue
        _hubo_clave_tvl = True
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
        serie = pd.Series(por_fecha, dtype=float)
        serie.index = pd.to_datetime(serie.index)
        series[nombre_chain] = serie.sort_index()
    if series:
        return series, ""
    if not _hubo_clave_no_agregada:
        return {}, "todas las claves descartadas por contener '-'"
    if not _hubo_clave_tvl:
        return {}, "chainTvls sin clave 'tvl'"
    return {}, "sin observaciones utilizables en 'tvl'"


def _densificar(serie: pd.Series) -> pd.Series:
    """
    Lleva una serie a frecuencia diaria continua dentro de su
    propio rango observado, propagando con ffill.

    NO extrapola fuera del rango observado: antes de la primera
    y después de la última observación la serie no existe (NaN).

    Args:
        serie: Serie con DatetimeIndex ordenado.

    Returns:
        Serie diaria continua entre su min y su max observados.
    """
    if serie.empty:
        return serie
    grilla_propia = pd.date_range(
        start=serie.index.min(), end=serie.index.max(), freq="D"
    )
    return serie.reindex(grilla_propia).ffill()


def agregar_series(
    componentes: list[pd.Series],
) -> tuple[pd.Series, pd.DatetimeIndex]:
    """
    Suma varias series de TVL (una por protocolo/chain) en una
    serie agregada diaria.

    Criterio de agregación:
      - Cada componente se densifica a diaria dentro de SU rango
        observado (ffill), sin extrapolar.
      - Fuera de ese rango el componente aporta 0: un protocolo
        que todavía no existía (o que dejó de reportar) no tiene
        valor tokenizado. Esto NO es relleno inventado.
      - El agregado empieza el primer día en que ALGÚN componente
        tiene observación; antes de esa fecha no existe (se deja
        fuera de la serie, sin back-fill).

    Args:
        componentes: Series de TVL a sumar.

    Returns:
        (serie_agregada, fechas_reales):
          serie_agregada : suma diaria desde el primer día con
                           datos hasta el último.
          fechas_reales  : fechas con observación DIRECTA de al
                           menos un componente (para la máscara
                           de ffill posterior).
    """
    validos = [s for s in componentes if not s.empty]
    if not validos:
        return pd.Series(dtype=float), pd.DatetimeIndex([])
    inicio = min(s.index.min() for s in validos)
    fin = max(s.index.max() for s in validos)
    grilla = pd.date_range(start=inicio, end=fin, freq="D")
    total = pd.Series(0.0, index=grilla)
    for serie in validos:
        densa = _densificar(serie)
        total = total.add(
            densa.reindex(grilla).fillna(0.0), fill_value=0.0
        )
    fechas_reales = pd.DatetimeIndex(
        sorted({fecha for s in validos for fecha in s.index})
    )
    total.index.name = "fecha"
    return total, fechas_reales


def registrar_serie(
    columna: str,
    componentes: list[pd.Series],
    n_protocolos: int,
    descripcion: str,
    destino_series: dict[str, pd.Series],
    destino_fechas: dict[str, pd.DatetimeIndex],
    destino_detalle: list[dict[str, Any]],
) -> None:
    """
    Agrega los componentes de una serie y la registra en los
    contenedores del proceso.

    Si la agregación no devuelve datos, la serie NO se registra
    (no se rellena con ceros) y queda anotada como fallida en el
    detalle.

    Args:
        columna: Nombre de la columna destino.
        componentes: Series de TVL a sumar.
        n_protocolos: Cuántos protocolos aportan a la serie.
        descripcion: Evidencia de cómo se construyó la serie.
        destino_series: Dict columna → serie agregada.
        destino_fechas: Dict columna → fechas con dato directo.
        destino_detalle: Lista de registros de detalle.
    """
    agregada, reales = agregar_series(componentes)
    if agregada.empty:
        if DEBUG:
            # CAMBIO H5-1 (Job 2)
            print(f"[DEBUG][ALERTA] {columna}: SIN DATOS — se omite")
        destino_detalle.append({
            "serie": columna,
            "n_protocolos": n_protocolos,
            "ok": False,
            "descripcion": descripcion,
            "min_fecha_origen": "s/d",
            "max_fecha_origen": "s/d",
            "fecha_extraccion": FECHA_EXTRACCION,
        })
        return
    destino_series[columna] = agregada
    destino_fechas[columna] = reales
    destino_detalle.append({
        "serie": columna,
        "n_protocolos": n_protocolos,
        "ok": True,
        "descripcion": descripcion,
        "min_fecha_origen": (
            agregada.index.min().date().isoformat()
        ),
        "max_fecha_origen": (
            agregada.index.max().date().isoformat()
        ),
        "fecha_extraccion": FECHA_EXTRACCION,
    })
    if DEBUG:
        print(
            f"[DEBUG] {columna}: {n_protocolos} protocolos, "
            f"{len(agregada)} días "
            f"({agregada.index.min().date()} → "
            f"{agregada.index.max().date()}), "
            f"último = {agregada.iloc[-1]:,.0f} USD"
        )


def construir_grilla(inicio: str, fin: str) -> pd.DatetimeIndex:
    """
    Construye un índice de fechas DIARIO continuo (calendario).

    Incluye fines de semana. Convención UTC.

    Args:
        inicio: Fecha de inicio ISO.
        fin: Fecha de fin ISO.

    Returns:
        DatetimeIndex con freq="D" y name="fecha".
    """
    return pd.date_range(
        start=inicio, end=fin, freq="D", name="fecha"
    )


def alinear_serie(
    serie_full: pd.Series,
    fechas_reales: pd.DatetimeIndex,
    grilla: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series]:
    """
    Alinea una serie de historia completa a la grilla de la
    ventana de análisis usando ffill documentado.

    NO se interpola y NO se hace back-fill: si la serie empieza
    después del inicio de la ventana, los días previos quedan
    NaN (el activo no existía; rellenarlos sería inventar dato).

    Usa la historia previa a la ventana, si existe, para que el
    ffill funcione desde el primer día de la grilla.

    Args:
        serie_full: Serie con historia completa, DatetimeIndex.
        fechas_reales: Fechas con observación directa de origen.
        grilla: Grilla diaria destino (ventana de análisis).

    Returns:
        (serie_alineada, mascara_ffill):
          serie_alineada : valores sobre la grilla.
          mascara_ffill  : True donde el valor vino de ffill.
    """
    vacia = pd.Series(dtype=float, index=grilla)
    sin_mask = pd.Series(False, index=grilla, dtype=bool)
    if serie_full.empty:
        return vacia, sin_mask

    inicio_ext = min(serie_full.index.min(), grilla.min())
    grilla_ext = pd.date_range(
        start=inicio_ext, end=grilla.max(), freq="D"
    )
    s_ext = serie_full.reindex(grilla_ext).ffill()
    s_ventana = s_ext.reindex(grilla)
    s_ventana.index.name = "fecha"

    # Directo = la fecha de la grilla tiene observación de origen
    directo = pd.Series(
        grilla.isin(fechas_reales), index=grilla, dtype=bool
    )
    mascara_ffill: pd.Series = (~directo) & s_ventana.notna()
    return s_ventana, mascara_ffill


def calcular_cobertura(
    df: pd.DataFrame,
    mascaras: dict[str, pd.Series],
    fecha_extraccion: str,
) -> pd.DataFrame:
    """
    Calcula la cobertura de cada columna del dataset alineado.

    Métricas por columna:
      n_total  : días en la grilla.
      n_real   : observaciones directas (sin ffill).
      n_ffill  : días rellenados por ffill.
      n_nulo   : días sin valor (NaN, típicamente al inicio si
                 la serie nace después de VENTANA_INICIO).
      pct_ffill: porcentaje de días rellenados.

    Args:
        df: Dataset alineado a la grilla diaria.
        mascaras: Dict columna → máscara_ffill booleana.
        fecha_extraccion: Fecha de extracción (ISO).

    Returns:
        DataFrame con una fila por columna.
    """
    filas: list[dict[str, Any]] = []
    for col in df.columns:
        serie = df[col]
        mask = mascaras.get(
            col, pd.Series(False, index=df.index)
        )
        n_total = len(serie)
        n_nulo = int(serie.isna().sum())
        n_real = int((~mask & serie.notna()).sum())
        n_ffill = int(mask.sum())
        pct_ffill = (
            round(n_ffill / n_total * 100, 1)
            if n_total > 0 else 0.0
        )
        con_valor = serie.dropna()
        if len(con_valor) > 0:
            min_f = con_valor.index.min().date().isoformat()
            max_f = con_valor.index.max().date().isoformat()
        else:
            min_f, max_f = "s/d", "s/d"
        filas.append({
            "serie": col,
            "n_total": n_total,
            "n_real": n_real,
            "n_ffill": n_ffill,
            "n_nulo": n_nulo,
            "pct_ffill": pct_ffill,
            "min_fecha": min_f,
            "max_fecha": max_f,
            "fecha_extraccion": fecha_extraccion,
        })
    return pd.DataFrame(filas)


def leer_dataset(ruta: str) -> pd.DataFrame:
    """
    Lee un dataset existente indexado por "fecha".

    Args:
        ruta: Ruta del CSV.

    Returns:
        DataFrame con DatetimeIndex "fecha", o vacío si el
        archivo no existe o no se puede leer.
    """
    if not os.path.exists(ruta):
        if DEBUG:
            print(f"[DEBUG] no existe: {ruta}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(ruta, index_col="fecha")
        df.index = pd.to_datetime(df.index)
        df.index.name = "fecha"
        return df
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG] error leyendo {ruta}: {e}")
        return pd.DataFrame()


def unir_por_fecha(
    df_base: pd.DataFrame,
    df_nuevo: pd.DataFrame,
    como: str = "left",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Une columnas nuevas a un dataset existente por índice fecha,
    sin tocar las columnas ya presentes.

    Las columnas que ya existen en df_base se OMITEN (no se
    sobrescribe el dataset validado) y se devuelven como
    conflictos para su reporte.

    Args:
        df_base: Dataset existente (índice fecha).
        df_nuevo: Columnas nuevas (índice fecha).
        como: Tipo de join ("left" para la grilla del base,
            "outer" para conservar toda la historia).

    Returns:
        (df_unido, columnas_omitidas).
    """
    conflictos = [
        c for c in df_nuevo.columns if c in df_base.columns
    ]
    aportadas = [
        c for c in df_nuevo.columns if c not in conflictos
    ]
    df_unido = df_base.join(df_nuevo[aportadas], how=como)
    df_unido.index.name = "fecha"
    return df_unido, conflictos


"""# 5. Proceso"""

# CAMBIO B1-2 (Bloque 1): en modo snapshot no se consulta
# DeFiLlama: se restauran las series congeladas y el Job termina.
if MODO_DATOS == "snapshot":
    restaurar_desde_snapshot(
        "Job 3", [ARCHIVO_DIARIAS_EXT, ARCHIVO_LARGAS_EXT]
    )
    print("[DEBUG] Job 3: MODO_DATOS=snapshot — 0 llamadas de red. Fin.")
    raise SystemExit(0)

# CAMBIO E2E-03-01: preflight del contrato de archivos del Job 3.
# Estas salidas se escriben en 5.b, ANTES del bloque "6. Save".
os.makedirs(DIR_REPORTES, exist_ok=True)

# El Job 1 produce ambas bases. Validar ANTES de descargar protocolos
# para no gastar llamadas externas ni reutilizar CSV ext de otra corrida.
df_diarias_base: pd.DataFrame = pd.DataFrame()
df_largas_base: pd.DataFrame = pd.DataFrame()
if UNIR_A_DATASET:
    for _ruta_base in (ARCHIVO_DIARIAS, ARCHIVO_LARGAS):
        if not os.path.isfile(_ruta_base):
            raise FileNotFoundError(
                f"[DEBUG][ERROR] Job 3: falta {_ruta_base}. "
                "El Job 1 debe generar ese archivo antes del Job 3."
            )
    df_diarias_base = leer_dataset(ARCHIVO_DIARIAS)
    df_largas_base = leer_dataset(ARCHIVO_LARGAS)
    for _ruta_base, _df_base in (
        (ARCHIVO_DIARIAS, df_diarias_base),
        (ARCHIVO_LARGAS, df_largas_base),
    ):
        if _df_base.empty:
            raise RuntimeError(
                f"[DEBUG][ERROR] Job 3: entrada vacia o ilegible: "
                f"{_ruta_base}. Verificar salida del Job 1 "
                "(CSV con columna 'fecha' y datos)."
            )
        if _df_base.index.isna().any():
            raise RuntimeError(
                f"[DEBUG][ERROR] Job 3: fechas nulas o invalidas "
                f"en {_ruta_base}. Revisar el Job 1."
            )

# ── 5.a Catálogo de protocolos RWA ────────────────────────────────

if DEBUG:
    print("\n[DEBUG] === 5.a Catálogo de protocolos RWA ===")
    print(f"[DEBUG] fecha de extracción: {FECHA_EXTRACCION}")

protocolos: pd.DataFrame = listar_protocolos_rwa()

if protocolos.empty:
    raise RuntimeError(
        "No se pudo listar protocolos RWA en DeFiLlama: "
        "sin catálogo no hay adquisición posible."
    )

if LIMITE_PROTOCOLOS is not None:
    protocolos = protocolos.head(LIMITE_PROTOCOLOS)

if DEBUG:
    print(f"[DEBUG] protocolos RWA: {len(protocolos)}")
    print(
        f"[DEBUG] TVL total catálogo: "
        f"{protocolos['tvl'].sum():,.0f} USD"
    )

# CAMBIO I1-3 (Job 2): conteo de protocolos RWA y TVL total del
# catálogo de esta corrida. Se escriben junto con el chequeo de
# umbral de 5.b, en una sola fila de resumen (ver más abajo).
_n_protocolos_rwa: int = len(protocolos)
_tvl_total_catalogo: float = float(protocolos["tvl"].sum())

# ── 5.b Descarga de series por protocolo (una pasada) ─────────────

if DEBUG:
    print("\n[DEBUG] === 5.b Descarga de series por protocolo ===")

# slug → {chain → serie}
series_protocolos: dict[str, dict[str, pd.Series]] = {}
log_descarga: list[dict[str, Any]] = []

for _, _fila in protocolos.iterrows():
    _slug = str(_fila["slug"])
    _nombre = str(_fila["name"])
    try:
        _series, _motivo = descargar_series_protocolo(_slug)
    except Exception as _e:
        _series, _motivo = {}, f"excepción: {_e}"
        if DEBUG:
            print(f"[DEBUG] error en {_slug}: {_e}")
    if _series:
        series_protocolos[_slug] = _series
    log_descarga.append({
        "slug": _slug,
        "name": _nombre,
        "ok": bool(_series),
        "n_chains": len(_series),
        "motivo": _motivo,  # CAMBIO I1-3 (Job 2)
        "tags": ",".join(_fila["tags"]),
        "fecha_extraccion": FECHA_EXTRACCION,
    })
    time.sleep(PAUSA)

df_log: pd.DataFrame = pd.DataFrame(log_descarga)

if DEBUG:
    _n_ok = int(df_log["ok"].sum())
    print(
        f"[DEBUG] protocolos con serie: {_n_ok}/{len(df_log)}"
    )
    _fallidos = df_log.loc[~df_log["ok"], "slug"].tolist()
    if _fallidos:
        print(
            f"[DEBUG] sin serie ({len(_fallidos)}): "
            f"{', '.join(_fallidos[:12])}"
        )

# CAMBIO I1-3 (Job 2): slugs sin serie y su motivo, a CSV.
_df_sin_serie = df_log.loc[
    ~df_log["ok"], ["slug", "name", "motivo", "fecha_extraccion"]
]
_df_sin_serie.to_csv(ARCHIVO_PROTOCOLOS_SIN_SERIE, index=False)
if DEBUG:
    print(f"[DEBUG] guardado: {ARCHIVO_PROTOCOLOS_SIN_SERIE}")

# CAMBIO I1-3 (Job 2): umbral de proporción con serie. No corta
# la ejecución — deja el aviso en [DEBUG][ALERTA] y en el
# resumen.
_prop_con_serie = (
    float(df_log["ok"].mean()) if len(df_log) > 0 else 0.0
)
_alerta_umbral = _prop_con_serie < MIN_PROP_PROTOCOLOS_CON_SERIE
if _alerta_umbral:
    print(
        f"[DEBUG][ALERTA] proporción de protocolos con serie "
        f"({_prop_con_serie:.1%}) por debajo del umbral "
        f"({MIN_PROP_PROTOCOLOS_CON_SERIE:.0%})"
    )

# CAMBIO I1-3 (Job 2): una fila de resumen por corrida, con el
# tamaño/TVL del catálogo y el resultado del chequeo de umbral.
_resumen_catalogo = pd.DataFrame([{
    "fecha_extraccion": FECHA_EXTRACCION,
    "n_protocolos_rwa": _n_protocolos_rwa,
    "tvl_total_catalogo_usd": _tvl_total_catalogo,
    "prop_protocolos_con_serie": _prop_con_serie,
    "umbral_min_prop_con_serie": MIN_PROP_PROTOCOLOS_CON_SERIE,
    "alerta_umbral": _alerta_umbral,
}])
_existe_resumen_catalogo = os.path.exists(ARCHIVO_RESUMEN_CATALOGO)
_resumen_catalogo.to_csv(
    ARCHIVO_RESUMEN_CATALOGO, mode="a",
    header=not _existe_resumen_catalogo, index=False,
)
if DEBUG:
    print(f"[DEBUG] guardado: {ARCHIVO_RESUMEN_CATALOGO}")

# ── 5.c Construcción de las 8 series ──────────────────────────────

if DEBUG:
    print("\n[DEBUG] === 5.c Construcción de las 8 series ===")

# Mapa slug → tags, para la agregación por tipo de activo
tags_por_slug: dict[str, list[str]] = {
    str(f["slug"]): list(f["tags"])
    for _, f in protocolos.iterrows()
}

series_full: dict[str, pd.Series] = {}
fechas_reales: dict[str, pd.DatetimeIndex] = {}
detalle_series: list[dict[str, Any]] = []


# GRUPO 1 — por red
for _columna, _chains in REDES_OBJETIVO.items():
    try:
        _componentes: list[pd.Series] = []
        _slugs_red: set[str] = set()
        for _slug, _por_chain in series_protocolos.items():
            for _chain in _chains:
                _serie_chain = _por_chain.get(_chain)
                if _serie_chain is not None:
                    _componentes.append(_serie_chain)
                    _slugs_red.add(_slug)
        registrar_serie(
            _columna, _componentes, len(_slugs_red),
            f"suma de chainTvls[{'/'.join(_chains)}] "
            "de los protocolos categoría RWA",
            series_full, fechas_reales, detalle_series,
        )
    except Exception as _e:
        if DEBUG:
            print(f"[DEBUG] error construyendo {_columna}: {_e}")

# GRUPO 2 — por tipo de activo (todas las chains del protocolo)
for _columna, _tags in TIPOS_OBJETIVO.items():
    try:
        _componentes = []
        _slugs_tipo: set[str] = set()
        for _slug, _por_chain in series_protocolos.items():
            _tags_proto = tags_por_slug.get(_slug, [])
            if not any(t in _tags_proto for t in _tags):
                continue
            _slugs_tipo.add(_slug)
            _componentes.extend(_por_chain.values())
        registrar_serie(
            _columna, _componentes, len(_slugs_tipo),
            f"suma de TVL (todas las chains) de protocolos con "
            f"tag {'/'.join(_tags)}",
            series_full, fechas_reales, detalle_series,
        )
    except Exception as _e:
        if DEBUG:
            print(f"[DEBUG] error construyendo {_columna}: {_e}")

# GRUPO 3 — total global
try:
    _componentes = [
        _s for _por_chain in series_protocolos.values()
        for _s in _por_chain.values()
    ]
    registrar_serie(
        COL_TOTAL, _componentes, len(series_protocolos),
        "suma de TVL de todos los protocolos categoría RWA "
        "en todas sus chains",
        series_full, fechas_reales, detalle_series,
    )
except Exception as _e:
    if DEBUG:
        print(f"[DEBUG] error construyendo {COL_TOTAL}: {_e}")

df_detalle: pd.DataFrame = pd.DataFrame(detalle_series)

# ── 5.d Alineación a la grilla diaria de la ventana ───────────────

if DEBUG:
    print("\n[DEBUG] === 5.d Alineación a grilla diaria ===")
    print(f"[DEBUG] ventana: {VENTANA_INICIO} → {HOY}")

grilla: pd.DatetimeIndex = construir_grilla(VENTANA_INICIO, HOY)
mascaras: dict[str, pd.Series] = {}
df_rwa: pd.DataFrame = pd.DataFrame(index=grilla)
df_rwa.index.name = "fecha"

for _columna in COLUMNAS_RWA:
    # CAMBIO H5-1 (Job 2): COLUMNAS_RWA es la lista declarada en
    # un único lugar de las series RWA esperadas (sección 3). Si
    # una de ellas no se puede construir, se marca con
    # [DEBUG][ALERTA] y se sigue sin ella (no se corta la
    # ejecución ni se rellena).
    _serie_full = series_full.get(_columna)
    if _serie_full is None:
        if DEBUG:
            print(
                f"[DEBUG][ALERTA] {_columna}: sin datos de "
                "origen — columna OMITIDA (no se rellena)"
            )
        continue
    try:
        _alineada, _mask = alinear_serie(
            _serie_full, fechas_reales[_columna], grilla
        )
    except Exception as _e:
        if DEBUG:
            print(
                f"[DEBUG][ALERTA] error alineando {_columna}: "
                f"{_e}"
            )
        continue
    if _alineada.notna().sum() == 0:
        if DEBUG:
            print(
                f"[DEBUG][ALERTA] {_columna}: sin valores "
                "dentro de la ventana — columna OMITIDA"
            )
        continue
    df_rwa[_columna] = _alineada
    mascaras[_columna] = _mask
    if DEBUG:
        _n_nulo = int(_alineada.isna().sum())
        print(
            f"[DEBUG] {_columna}: "
            f"{int((~_mask & _alineada.notna()).sum())} reales, "
            f"{int(_mask.sum())} ffill, {_n_nulo} NaN"
        )
        if _n_nulo > 0:
            _primer_valor = _alineada.dropna().index.min()
            print(
                f"[DEBUG]   ATENCIÓN: la serie nace el "
                f"{_primer_valor.date()}, después de "
                f"{VENTANA_INICIO}. Se dejan {_n_nulo} NaN al "
                "inicio (sin back-fill)."
            )

# ── 5.e Historia completa (series largas) ─────────────────────────

df_rwa_largas: pd.DataFrame = (
    pd.DataFrame(
        {c: series_full[c] for c in COLUMNAS_RWA
         if c in series_full}
    )
    if series_full else pd.DataFrame()
)
if not df_rwa_largas.empty:
    df_rwa_largas.index.name = "fecha"

if DEBUG:
    if df_rwa_largas.empty:
        print("\n[DEBUG] sin historia RWA para series largas")
    else:
        print(
            f"\n[DEBUG] series largas RWA: "
            f"{df_rwa_largas.shape} "
            f"({df_rwa_largas.index.min().date()} → "
            f"{df_rwa_largas.index.max().date()})"
        )

# ── 5.f Reporte de cobertura ──────────────────────────────────────

if DEBUG:
    print("\n[DEBUG] === 5.f Cobertura de las series RWA ===")

cobertura: pd.DataFrame = calcular_cobertura(
    df_rwa, mascaras, FECHA_EXTRACCION
)

if DEBUG and not cobertura.empty:
    pd.set_option("display.max_colwidth", 30)
    pd.set_option("display.width", 130)
    _cob_cols = [
        "serie", "n_total", "n_real", "n_ffill", "n_nulo",
        "pct_ffill", "min_fecha", "max_fecha",
    ]
    print(cobertura[_cob_cols].to_string(index=False))

# ── 5.g Unión al dataset existente ────────────────────────────────

df_diarias_ext: pd.DataFrame = pd.DataFrame()
df_largas_ext: pd.DataFrame = pd.DataFrame()

if UNIR_A_DATASET:
    # CAMBIO E2E-03-01: si falta un lado RWA no se crea un CSV
    # incompleto ni se deja que otro Job consuma un ext antiguo.
    if df_rwa.empty or df_rwa_largas.empty:
        _rwa_faltantes = [
            _nombre for _nombre, _df in (
                ("series_rwa.csv", df_rwa),
                ("series_rwa_largas.csv", df_rwa_largas),
            ) if _df.empty
        ]
        raise RuntimeError(
            "[DEBUG][ERROR] Job 3: faltan datos RWA para "
            "generar series_diarias_ext.csv y series_largas_ext.csv: "
            f"{', '.join(_rwa_faltantes)}. "
            "No se reutilizan archivos ext de corridas anteriores."
        )
    if DEBUG:
        print("\n[DEBUG] === 5.g Unión al dataset existente ===")
    # CAMBIO E2E-03-01: base diaria leida y validada en preflight.
    if df_diarias_base.empty:
        if DEBUG:
            print(
                "[DEBUG] series_diarias.csv no disponible: no se "
                "genera series_diarias_ext.csv (no se inventa base)"
            )
    else:
        df_diarias_ext, _conflictos = unir_por_fecha(
            df_diarias_base, df_rwa, como="left"
        )
        if DEBUG:
            print(
                f"[DEBUG] base {df_diarias_base.shape} + RWA "
                f"{df_rwa.shape} → ext {df_diarias_ext.shape}"
            )
            if _conflictos:
                print(
                    "[DEBUG] columnas ya existentes NO "
                    f"sobrescritas: {_conflictos}"
                )
            _fuera = int(
                (~df_rwa.index.isin(df_diarias_base.index)).sum()
            )
            if _fuera:
                print(
                    f"[DEBUG] {_fuera} días de la grilla RWA "
                    "quedan fuera del índice del dataset base "
                    "(left join)"
                )

    # CAMBIO E2E-03-01: base larga leida y validada en preflight.
    if not df_largas_base.empty and not df_rwa_largas.empty:
        df_largas_ext, _conf_largas = unir_por_fecha(
            df_largas_base, df_rwa_largas, como="outer"
        )
        if DEBUG:
            print(
                f"[DEBUG] largas {df_largas_base.shape} + RWA "
                f"{df_rwa_largas.shape} → ext "
                f"{df_largas_ext.shape}"
            )
            if _conf_largas:
                print(
                    "[DEBUG] columnas largas NO sobrescritas: "
                    f"{_conf_largas}"
                )

    # CAMBIO E2E-03-01: validar la produccion de AMBOS ext.
    if df_diarias_ext.empty or df_largas_ext.empty:
        raise RuntimeError(
            "[DEBUG][ERROR] Job 3: una o ambas uniones quedaron "
            "vacias; no se pueden generar los CSV ext. "
            f"diarias={df_diarias_ext.shape}, "
            f"largas={df_largas_ext.shape}."
        )

# ── 5.h Resumen ───────────────────────────────────────────────────

print("\n[RWA] === Resumen de adquisición ===")
print(f"[RWA] fecha de extracción: {FECHA_EXTRACCION}")
print(f"[RWA] ventana: {VENTANA_INICIO} → {HOY}")
print(
    f"[RWA] series solicitadas: {len(COLUMNAS_RWA)} | "
    f"construidas: {len(df_rwa.columns)}"
)
_faltantes = [c for c in COLUMNAS_RWA if c not in df_rwa.columns]
if _faltantes:
    # CAMBIO H5-1 (Job 2): resumen final de series esperadas que
    # no se pudieron construir.
    print(
        f"[DEBUG][ALERTA] RWA OMITIDAS (sin dato, no "
        f"rellenadas): {_faltantes}"
    )
if not df_detalle.empty:
    for _, _fila_det in df_detalle.iterrows():
        print(
            f"[RWA] {_fila_det['serie']}: "
            f"{_fila_det['n_protocolos']} protocolos — "
            f"{_fila_det['descripcion']}"
        )
print(
    "[RWA] NOTA: adopción incipiente (RWA en Stellar ~ago-2025, "
    "en Ripple ~jun-2025): dinámica dominada por crecimiento "
    "estructural, no por respuesta a estrés."
)
print(
    "[RWA] NOTA: los tags de tipo no son exclusivos; las series "
    "por tipo no suman rwa_total_global."
)

"""# 6. Save"""

if SAVE_DATA:
    os.makedirs(DIR_PROCESSED, exist_ok=True)
    os.makedirs(DIR_REPORTES, exist_ok=True)

    if not df_rwa.empty:
        df_rwa.to_csv(ARCHIVO_RWA)
        if DEBUG:
            print(f"\n[DEBUG] guardado: {ARCHIVO_RWA}")
        registrar_huella(ARCHIVO_RWA, ARCHIVO_HUELLAS)  # CAMBIO H3-3
        congelar_snapshot([ARCHIVO_RWA], "Job 3")  # CAMBIO B1-2
    elif DEBUG:
        print("\n[DEBUG] df_rwa vacío, no se guarda")

    if not df_rwa_largas.empty:
        df_rwa_largas.to_csv(ARCHIVO_RWA_LARGAS)
        if DEBUG:
            print(f"[DEBUG] guardado: {ARCHIVO_RWA_LARGAS}")
        registrar_huella(
            ARCHIVO_RWA_LARGAS, ARCHIVO_HUELLAS
        )  # CAMBIO H3-3
        congelar_snapshot([ARCHIVO_RWA_LARGAS], "Job 3")  # CAMBIO B1-2

    if not df_diarias_ext.empty:
        df_diarias_ext.to_csv(ARCHIVO_DIARIAS_EXT)
        if DEBUG:
            print(f"[DEBUG] guardado: {ARCHIVO_DIARIAS_EXT}")
        congelar_snapshot([ARCHIVO_DIARIAS_EXT], "Job 3")  # CAMBIO B1-2

    if not df_largas_ext.empty:
        df_largas_ext.to_csv(ARCHIVO_LARGAS_EXT)
        if DEBUG:
            print(f"[DEBUG] guardado: {ARCHIVO_LARGAS_EXT}")
        congelar_snapshot([ARCHIVO_LARGAS_EXT], "Job 3")  # CAMBIO B1-2

    if not cobertura.empty:
        cobertura.to_csv(ARCHIVO_COBERTURA, index=False)
        if DEBUG:
            print(f"[DEBUG] guardado: {ARCHIVO_COBERTURA}")
elif DEBUG:
    print("\n[DEBUG] SAVE_DATA=False — datos solo en memoria")