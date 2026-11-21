"""# Job 1 - Adquisición base"""

# 1. Configuración


DEBUG: bool = True
SAVE_DATA: bool = True

# 2. Imports

import datetime
import hashlib  # CAMBIO H3-2 (Job 1.a): huellas SHA-256
import json  # CAMBIO I1-1 (Job 1.a): guardado de JSON crudo
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

"""# 3. Paths y Constantes"""

# CAMBIO H2-2 (Job 1.a): ver nota en Job 0 (mismo patrón).
BASE_DIR: str = os.environ.get(
    "TESIS_BASE_DIR", "/content/drive/MyDrive/Tesis/expansion"
)
DIR_PROCESSED: str = os.path.join(BASE_DIR, "data", "processed")
DIR_REPORTES: str = os.path.join(BASE_DIR, "outputs", "reportes")

ARCHIVO_DIARIAS: str = os.path.join(
    DIR_PROCESSED, "series_diarias.csv"
)
ARCHIVO_LARGAS: str = os.path.join(
    DIR_PROCESSED, "series_largas.csv"
)
ARCHIVO_COBERTURA: str = os.path.join(
    DIR_REPORTES, "cobertura_series.csv"
)
# CAMBIO H3-2 (Job 1.a): huellas SHA-256 de los CSV guardados.
ARCHIVO_HUELLAS: str = os.path.join(
    DIR_REPORTES, "huellas_datos.csv"
)

# Ruta de main para verificación cruzada (solo lectura)
MAIN_DIR: str = "/content/drive/MyDrive/Tesis/main"
MAIN_DIARIAS: str = os.path.join(
    MAIN_DIR, "data", "processed", "series_diarias.csv"
)

TIMEOUT: int = 30
PAUSA: float = 1.5
VENTANA_INICIO: str = "2026-01-01"
HISTORIA_INICIO: str = "2020-01-01"

# ── Fecha de corte ──────────────────────────────────────────
# La corrida original de main produjo 220 filas (2026-01-01 →
# 2026-08-08). Para mantener comparabilidad, la grilla de
# expansion se corta en esta fecha, no en HOY.
FECHA_CORTE: str = "2026-08-08"

HOY: str = datetime.date.today().isoformat()
FECHA_EXTRACCION: str = HOY

# CAMBIO I1-1 (Job 1.a): reintentos de get_json y carpeta donde
# se congela cada respuesta cruda (FRED, ECB) antes de procesarla.
MAX_REINTENTOS: int = 3
DIR_RAW: str = os.path.join(
    BASE_DIR, "data", "raw", FECHA_EXTRACCION
)

# FRED API key (Colab userdata o variable de entorno)
try:
    from google.colab import userdata as _ud  # type: ignore
    FRED_API_KEY: str = _ud.get("FRED_API_KEY") or ""
except Exception:
    FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# --- Endpoints base ---
CM_BASE: str = (
    "https://community-api.coinmetrics.io"
    "/v4/timeseries/asset-metrics"
)
LLAMA_STABLE: str = "https://stablecoins.llama.fi"
FRED_OBS: str = (
    "https://api.stlouisfed.org/fred/series/observations"
)
ECB_BASE: str = (
    "https://data-api.ecb.europa.eu/service/data"
)
COINGECKO_BASE: str = "https://api.coingecko.com/api/v3"

# Redes de pago ENTRA en Fase 1 (asset CM → nombre legible)
CM_REDES: dict[str, str] = {
    "xlm": "Stellar",
    "xrp": "XRP",
    "algo": "Algorand",
}

# Métricas CoinMetrics → sufijo de columna
CM_METRICAS: dict[str, str] = {
    "TxCnt": "txcnt",
    "FeeTotNtv": "feetot",
    "AdrActCnt": "adract",
}

# DeFiLlama USDY (id=129, confirmado en Fase 1) dejó de resolver:
# el id ya no existe en /stablecoins y no hay símbolo "USDY" en
# el catálogo actual (verificado 2026-10-25). Se reemplaza por
# CoinGecko, que sí mantiene el token activo.
COINGECKO_USDY_ID: str = "ondo-us-dollar-yield"

# FRED y ECB
FRED_STLFSI4_ID: str = "STLFSI4"
ECB_CISS_KEY: str = "D.U2.Z0Z.4F.EC.SS_CIN.IDX"

# CAMBIO H5-2 (Job 1.a): series macro/estrés que este Job
# descarga (2 de las 16 totales del proyecto; las otras 14 son
# FRED_SERIES en Job 8, fuera del alcance de esta fase — ver
# nota de alcance en el cierre de F1).
SERIES_MACRO_ESTRES_JOB1A: list[str] = [
    "macro_eeuu_stlfsi",
    "macro_euro_ciss",
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

    CAMBIO H3-2 (Job 1.a).

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

    CAMBIO I1-1 (Job 1.a).

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

    CAMBIO I1-1 (Job 1.a): antes devolvía None en silencio ante
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
                            f"{url[:55]}"
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
                        f"{url[:55]}"
                    )
        except requests.Timeout:
            _error = "timeout"
            if DEBUG:
                print(
                    f"[DEBUG] intento {_intento}/"
                    f"{MAX_REINTENTOS} — timeout: {url[:55]}"
                )
        except requests.RequestException as e:
            _error = f"request error: {type(e).__name__}"  # CAMBIO B1-2: sin URL/clave
            if DEBUG:
                print(
                    f"[DEBUG] intento {_intento}/"
                    f"{MAX_REINTENTOS} — {_error}: {url[:55]}"
                )
        if _intento < MAX_REINTENTOS:
            time.sleep(PAUSA * _intento)
    if DEBUG:
        print(
            f"[DEBUG][ALERTA] get_json agotó "
            f"{MAX_REINTENTOS} intentos ({_error}): {url[:55]}"
        )
    return None


def adquirir_coinmetrics(
    asset: str,
    metricas: dict[str, str],
    inicio: str = VENTANA_INICIO,
    fin: str = FECHA_CORTE,
) -> pd.DataFrame:
    """
    Descarga series de métricas diarias de CoinMetrics
    Community API para un asset dado.

    Args:
        asset: Símbolo del asset en CoinMetrics (ej. "xlm").
        metricas: Dict {nombre_cm: sufijo_col}.
        inicio: Fecha de inicio ISO.
        fin: Fecha de fin ISO.

    Returns:
        DataFrame con DatetimeIndex y columnas float por métrica.
        Vacío si falla la request.
    """
    params: dict[str, Any] = {
        "assets": asset,
        "metrics": ",".join(metricas.keys()),
        "frequency": "1d",
        "start_time": inicio,
        "end_time": fin,
        "page_size": 5000,
    }
    data = get_json(CM_BASE, params)
    if data is None:
        if DEBUG:
            print(f"[DEBUG] CoinMetrics {asset}: request fallida")
        return pd.DataFrame()

    registros: list[dict] = data.get("data", [])
    if not registros:
        if DEBUG:
            print(
                f"[DEBUG] CoinMetrics {asset}: "
                "sin registros en la ventana"
            )
        return pd.DataFrame()

    filas: list[dict] = []
    for r in registros:
        fila: dict[str, Any] = {
            "fecha": r.get("time", "")[:10]
        }
        for m in metricas:
            val = r.get(m)
            try:
                fila[m] = (
                    float(val)
                    if val not in (None, "") else None
                )
            except (ValueError, TypeError):
                fila[m] = None
        filas.append(fila)

    df = pd.DataFrame(filas)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.set_index("fecha").sort_index()

    if DEBUG:
        for m in metricas:
            n = (
                int(df[m].notna().sum())
                if m in df.columns else 0
            )
            print(
                f"[DEBUG] CoinMetrics {asset}/{m}: {n} obs"
            )
    return df


def adquirir_usdy_supply(
    coingecko_id: str,
    dias: int = 365,
) -> pd.DataFrame:
    """
    Descarga la serie histórica de supply de USDY, estimada
    desde CoinGecko (market_cap / price) por día.

    DeFiLlama retiró/renombró el id de stablecoin que se usaba
    antes (129): ya no existe en /stablecoins y no hay ningún
    símbolo "USDY" en el catálogo actual (verificado 2026-10-25,
    HTTP 200 con body vacío para ?stablecoin=129). CoinGecko sí
    mantiene el token activo bajo el id "ondo-us-dollar-yield".

    Args:
        coingecko_id: ID de CoinGecko (ver COINGECKO_USDY_ID).
        dias: Días de historia a pedir (máx. 365 en plan free).

    Returns:
        DataFrame con DatetimeIndex y columna 'supply' (float).
        Vacío si falla.
    """
    url = (
        f"{COINGECKO_BASE}/coins/{coingecko_id}/market_chart"
        f"?vs_currency=usd&days={dias}&interval=daily"
    )
    data = get_json(url)
    if not isinstance(data, dict):
        if DEBUG:
            print(
                "[DEBUG] CoinGecko USDY: "
                "respuesta vacía o inválida"
            )
        return pd.DataFrame()

    precios: list = data.get("prices", [])
    caps: list = data.get("market_caps", [])
    if not precios or not caps:
        if DEBUG:
            print(
                "[DEBUG] CoinGecko USDY: "
                "sin 'prices' o 'market_caps' en la respuesta"
            )
        return pd.DataFrame()

    precio_por_ts: dict[int, float] = {
        int(ts): float(val) for ts, val in precios
    }

    filas: list[dict] = []
    for ts, mcap in caps:
        ts = int(ts)
        precio = precio_por_ts.get(ts)
        if not precio:
            continue
        fecha = datetime.datetime.fromtimestamp(
            ts / 1000, tz=datetime.timezone.utc
        ).date().isoformat()
        try:
            supply_val = float(mcap) / precio
        except (ValueError, TypeError, ZeroDivisionError):
            supply_val = None
        filas.append({"fecha": fecha, "supply": supply_val})

    if not filas:
        return pd.DataFrame()

    df = pd.DataFrame(filas)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = (
        df.set_index("fecha")
        .sort_index()
        .groupby(level=0)
        .last()
    )
    if DEBUG:
        n_ok = int(df["supply"].notna().sum())
        print(
            f"[DEBUG] CoinGecko USDY: {n_ok} obs "
            f"({df.index.min().date()} → "
            f"{df.index.max().date()})"
        )
    return df


def adquirir_fred(
    series_id: str,
    inicio: str = HISTORIA_INICIO,
) -> pd.DataFrame:
    """
    Descarga una serie de FRED desde 'inicio'.

    Args:
        series_id: ID de la serie FRED (ej. "STLFSI4").
        inicio: Fecha de inicio de la descarga (ISO).

    Returns:
        DataFrame con DatetimeIndex y columna 'value' (float).
        Vacío si falla o si FRED_API_KEY no está configurada.
    """
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
    # CAMBIO I1-1 (Job 1.a): se congela la respuesta cruda antes
    # de procesarla.
    _ruta_crudo = os.path.join(DIR_RAW, f"fred_{series_id}.json")
    data = get_json(FRED_OBS, params, ruta_crudo=_ruta_crudo)
    if data is None:
        if DEBUG:
            print(f"[DEBUG] FRED {series_id}: request fallida")
        return pd.DataFrame()

    obs: list[dict] = data.get("observations", [])
    if not obs:
        if DEBUG:
            print(
                f"[DEBUG] FRED {series_id}: "
                "sin observaciones (¿serie no existe?)"
            )
        return pd.DataFrame()

    filas: list[dict] = []
    for o in obs:
        val = o.get("value", ".")
        try:
            fval = (
                float(val) if val not in (".", "") else None
            )
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


def adquirir_ecb_ciss(clave: str) -> pd.DataFrame:
    """
    Descarga el CISS de la zona euro desde ECB Data Portal
    en formato SDMX-JSON.

    Args:
        clave: Clave SDMX de la serie.

    Returns:
        DataFrame con DatetimeIndex y columna 'value' (float).
        Vacío si falla el parseo.
    """
    url = f"{ECB_BASE}/CISS/{clave}"
    params: dict[str, Any] = {
        "format": "jsondata",
        "startPeriod": HISTORIA_INICIO,
    }
    # CAMBIO I1-1 (Job 1.a): se congela la respuesta cruda antes
    # de procesarla.
    _ruta_crudo = os.path.join(DIR_RAW, "ecb_ciss.json")
    data = get_json(url, params, ruta_crudo=_ruta_crudo)
    if data is None:
        if DEBUG:
            print("[DEBUG] ECB CISS: request fallida")
        return pd.DataFrame()

    try:
        dims_obs: list[dict] = (
            data["structure"]["dimensions"]["observation"]
        )
        time_dim: dict = next(
            d for d in dims_obs if d["id"] == "TIME_PERIOD"
        )
        time_values: list[dict] = time_dim["values"]
        series_dict: dict = data["dataSets"][0]["series"]
        serie_key: str = next(iter(series_dict))
        obs_dict: dict = (
            series_dict[serie_key]["observations"]
        )
        filas: list[dict] = []
        for k, v in obs_dict.items():
            idx = int(k)
            if idx >= len(time_values):
                continue
            fecha = time_values[idx]["id"]
            raw = v[0] if v else None
            try:
                fval = float(raw) if raw is not None else None
            except (ValueError, TypeError):
                fval = None
            filas.append({"fecha": fecha, "value": fval})

        if not filas:
            return pd.DataFrame()

        df = pd.DataFrame(filas)
        df["fecha"] = pd.to_datetime(df["fecha"])
        df = df.set_index("fecha").sort_index()
        if DEBUG:
            n_ok = int(df["value"].notna().sum())
            print(
                f"[DEBUG] ECB CISS: {n_ok} obs "
                f"({df.index.min().date()} → "
                f"{df.index.max().date()})"
            )
        return df
    except (
        KeyError, StopIteration, TypeError, IndexError
    ) as e:
        if DEBUG:
            print(f"[DEBUG] ECB CISS parse error: {e}")
        return pd.DataFrame()


def construir_grilla(
    inicio: str,
    fin: str,
) -> pd.DatetimeIndex:
    """
    Construye un índice de fechas DIARIO continuo (calendario).

    Args:
        inicio: Fecha de inicio ISO (ej. "2026-01-01").
        fin: Fecha de fin ISO (ej. "2026-08-08").

    Returns:
        DatetimeIndex con freq="D" y name="fecha".
    """
    return pd.date_range(
        start=inicio, end=fin, freq="D", name="fecha"
    )


def alinear_serie(
    serie_full: pd.Series,
    grilla: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series]:
    """
    Alinea una serie a la grilla diaria usando ffill documentado.

    Args:
        serie_full: Serie con historia completa, DatetimeIndex.
        grilla: Grilla diaria destino.

    Returns:
        (serie_alineada, mascara_ffill).
    """
    if serie_full.empty:
        return (
            pd.Series(dtype=float, index=grilla),
            pd.Series(False, index=grilla, dtype=bool),
        )

    inicio_ext = min(serie_full.index.min(), grilla.min())
    grilla_ext = pd.date_range(
        start=inicio_ext, end=grilla.max(), freq="D"
    )
    s_ext = serie_full.reindex(grilla_ext).ffill()

    s_ventana = s_ext.reindex(grilla)

    s_directo = serie_full.reindex(grilla)
    mascara_ffill: pd.Series = (
        s_directo.isna() & s_ventana.notna()
    )
    return s_ventana, mascara_ffill


def calcular_cobertura(
    df: pd.DataFrame,
    mascaras: dict[str, pd.Series],
    fecha_extraccion: str,
) -> pd.DataFrame:
    """
    Calcula cobertura de cada columna del dataset alineado.

    Args:
        df: Dataset alineado con grilla diaria.
        mascaras: Dict col → máscara_ffill booleana.
        fecha_extraccion: Fecha de extracción (ISO).

    Returns:
        DataFrame con una fila por columna.
    """
    filas: list[dict] = []
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
        fechas_val = serie.dropna()
        if len(fechas_val) > 0:
            min_f = fechas_val.index.min().date().isoformat()
            max_f = fechas_val.index.max().date().isoformat()
        else:
            min_f, max_f = "s/d", "s/d"
        filas.append({
            "columna": col,
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


def verificar_contra_main(
    df_expansion: pd.DataFrame,
) -> None:
    """
    Compara el dataset generado contra el de main (si existe).

    Solo lectura sobre main. Imprime un resumen de máximo 5
    líneas con shape, columnas, rango de fechas y diferencias
    de valores.

    Args:
        df_expansion: Dataset recién generado por este notebook.
    """
    if not os.path.exists(MAIN_DIARIAS):
        if DEBUG:
            print(
                "[DEBUG] Verificación: main/series_diarias.csv "
                "no encontrado — se omite"
            )
        return

    try:
        df_main = pd.read_csv(
            MAIN_DIARIAS, parse_dates=["fecha"],
        )
        df_main = df_main.set_index("fecha").sort_index()
    except Exception as e:
        if DEBUG:
            print(
                f"[DEBUG] Verificación: error leyendo main: {e}"
            )
        return

    print("\n[VERIF] Comparación expansion vs main:")
    print(
        f"[VERIF]   shape: expansion={df_expansion.shape} "
        f"main={df_main.shape}"
    )

    cols_exp = set(df_expansion.columns)
    cols_main = set(df_main.columns)
    solo_exp = cols_exp - cols_main
    solo_main = cols_main - cols_exp
    if solo_exp or solo_main:
        print(
            f"[VERIF]   cols solo expansion: "
            f"{sorted(solo_exp) or '—'} | "
            f"solo main: {sorted(solo_main) or '—'}"
        )
    else:
        print("[VERIF]   columnas: idénticas")

    rango_exp = (
        f"{df_expansion.index.min().date()} → "
        f"{df_expansion.index.max().date()}"
    )
    rango_main = (
        f"{df_main.index.min().date()} → "
        f"{df_main.index.max().date()}"
    )
    print(
        f"[VERIF]   rango: expansion={rango_exp} "
        f"main={rango_main}"
    )

    comunes = sorted(cols_exp & cols_main)
    if comunes:
        idx_comun = df_expansion.index.intersection(
            df_main.index
        )
        if len(idx_comun) > 0:
            diffs: list[str] = []
            for col in comunes:
                a = df_expansion.loc[idx_comun, col]
                b = df_main.loc[idx_comun, col]
                mask_ambos = a.notna() & b.notna()
                if not mask_ambos.any():
                    continue
                max_diff = float(
                    (a[mask_ambos] - b[mask_ambos]).abs().max()
                )
                if max_diff > 1e-6:
                    diffs.append(f"{col}(Δmax={max_diff:.6f})")
            if diffs:
                print(
                    f"[VERIF]   diffs (tol>1e-6): "
                    f"{', '.join(diffs[:5])}"
                )
            else:
                print(
                    "[VERIF]   valores: idénticos dentro de "
                    "tolerancia (1e-6)"
                )
        else:
            print("[VERIF]   sin fechas en común para comparar")


"""# 5. Proceso"""

# CAMBIO B1-2 (Bloque 1): en modo snapshot no se descarga nada:
# se restauran las salidas congeladas y el Job termina acá.
if MODO_DATOS == "snapshot":
    restaurar_desde_snapshot("Job 1", [ARCHIVO_DIARIAS, ARCHIVO_LARGAS])
    print("[DEBUG] Job 1: MODO_DATOS=snapshot — 0 llamadas de red. Fin.")
    raise SystemExit(0)

# Historia completa por columna (pre-alineación)
series_full: dict[str, pd.Series] = {}
# Log de descarga por columna
log_descarga: list[dict] = []

# ── 1.a CoinMetrics: Stellar, XRP, Algorand ──────────────────────

if DEBUG:
    print("\n[DEBUG] === Redes de pago (CoinMetrics) ===")

for _asset, _nombre in CM_REDES.items():
    try:
        _df_cm = adquirir_coinmetrics(_asset, CM_METRICAS)
        for _m_cm, _sufijo in CM_METRICAS.items():
            _col = f"onchain_{_asset}_{_sufijo}"
            if (
                not _df_cm.empty
                and _m_cm in _df_cm.columns
                and _df_cm[_m_cm].notna().any()
            ):
                series_full[_col] = _df_cm[_m_cm]
                _n = int(_df_cm[_m_cm].notna().sum())
                log_descarga.append({
                    "fuente": "CoinMetrics",
                    "columna": _col,
                    "ok": True,
                    "n_obs": _n,
                    "nota": "",
                })
            else:
                log_descarga.append({
                    "fuente": "CoinMetrics",
                    "columna": _col,
                    "ok": False,
                    "n_obs": 0,
                    "nota": "métrica no disponible, omitida",
                })
                if DEBUG:
                    print(
                        f"[DEBUG] {_col}: "
                        "sin datos → columna omitida"
                    )
    except Exception as _e:
        if DEBUG:
            print(
                f"[DEBUG] Error CoinMetrics/"
                f"{_asset}: {_e}"
            )
        for _sfx in CM_METRICAS.values():
            log_descarga.append({
                "fuente": "CoinMetrics",
                "columna": f"onchain_{_asset}_{_sfx}",
                "ok": False,
                "n_obs": 0,
                "nota": f"excepción: {_e}",
            })
    time.sleep(PAUSA)

# ── 1.b CoinGecko: Ondo USDY ────────────────────────────────────────

if DEBUG:
    print("\n[DEBUG] === RWA: Ondo USDY (CoinGecko) ===")

_col_usdy: str = "rwa_usdy_supply"
try:
    _df_usdy = adquirir_usdy_supply(COINGECKO_USDY_ID)
    if not _df_usdy.empty and _df_usdy["supply"].notna().any():
        series_full[_col_usdy] = _df_usdy["supply"]
        log_descarga.append({
            "fuente": "CoinGecko",
            "columna": _col_usdy,
            "ok": True,
            "n_obs": int(_df_usdy["supply"].notna().sum()),
            "nota": f"coingecko_id={COINGECKO_USDY_ID}",
        })
    else:
        log_descarga.append({
            "fuente": "CoinGecko",
            "columna": _col_usdy,
            "ok": False,
            "n_obs": 0,
            "nota": "respuesta vacía",
        })
except Exception as _e:
    if DEBUG:
        print(f"[DEBUG] Error CoinGecko USDY: {_e}")
    log_descarga.append({
        "fuente": "CoinGecko",
        "columna": _col_usdy,
        "ok": False,
        "n_obs": 0,
        "nota": f"excepción: {_e}",
    })
time.sleep(PAUSA)

# ── 1.c FRED: STLFSI4 (EE.UU.) ───────────────────────────────────

if DEBUG:
    print("\n[DEBUG] === Macro EE.UU.: FRED STLFSI4 ===")

_col_stlfsi: str = SERIES_MACRO_ESTRES_JOB1A[0]  # CAMBIO H5-2
try:
    _df_stlfsi = adquirir_fred(FRED_STLFSI4_ID)
    if (
        not _df_stlfsi.empty
        and _df_stlfsi["value"].notna().any()
    ):
        series_full[_col_stlfsi] = _df_stlfsi["value"]
        log_descarga.append({
            "fuente": "FRED",
            "columna": _col_stlfsi,
            "ok": True,
            "n_obs": int(_df_stlfsi["value"].notna().sum()),
            "nota": "semanal → ffill en grilla diaria",
        })
    else:
        if DEBUG:
            print(
                f"[DEBUG][ALERTA] {_col_stlfsi}: sin datos o "
                "FRED_API_KEY ausente"
            )
        log_descarga.append({
            "fuente": "FRED",
            "columna": _col_stlfsi,
            "ok": False,
            "n_obs": 0,
            "nota": "sin datos o FRED_API_KEY ausente",
        })
except Exception as _e:
    if DEBUG:
        print(f"[DEBUG][ALERTA] Error FRED STLFSI4: {_e}")
    log_descarga.append({
        "fuente": "FRED",
        "columna": _col_stlfsi,
        "ok": False,
        "n_obs": 0,
        "nota": f"excepción: {_e}",
    })
time.sleep(PAUSA)

# ── 1.d ECB: CISS (Zona Euro) ─────────────────────────────────────

if DEBUG:
    print("\n[DEBUG] === Macro Zona Euro: ECB CISS ===")

_col_ciss: str = SERIES_MACRO_ESTRES_JOB1A[1]  # CAMBIO H5-2
try:
    _df_ciss = adquirir_ecb_ciss(ECB_CISS_KEY)
    if (
        not _df_ciss.empty
        and _df_ciss["value"].notna().any()
    ):
        series_full[_col_ciss] = _df_ciss["value"]
        log_descarga.append({
            "fuente": "ECB",
            "columna": _col_ciss,
            "ok": True,
            "n_obs": int(_df_ciss["value"].notna().sum()),
            "nota": f"clave={ECB_CISS_KEY}",
        })
    else:
        if DEBUG:
            print(
                f"[DEBUG][ALERTA] {_col_ciss}: respuesta "
                "vacía o error de parseo"
            )
        log_descarga.append({
            "fuente": "ECB",
            "columna": _col_ciss,
            "ok": False,
            "n_obs": 0,
            "nota": "respuesta vacía o error de parseo",
        })
except Exception as _e:
    if DEBUG:
        print(f"[DEBUG][ALERTA] Error ECB CISS: {_e}")
    log_descarga.append({
        "fuente": "ECB",
        "columna": _col_ciss,
        "ok": False,
        "n_obs": 0,
        "nota": f"excepción: {_e}",
    })

# ── 2. Log de descarga ─────────────────────────────────────────────

df_log: pd.DataFrame = pd.DataFrame(log_descarga)

if DEBUG:
    _n_ok = int(df_log["ok"].sum()) if not df_log.empty else 0
    print(
        f"\n[DEBUG] Series descargadas OK: "
        f"{_n_ok}/{len(df_log)}"
    )

# ── 3. Grilla diaria y alineación ─────────────────────────────────

if DEBUG:
    print(
        f"\n[DEBUG] === Alineación a grilla diaria ==="
    )
    print(
        f"[DEBUG] Ventana: {VENTANA_INICIO} → {FECHA_CORTE} "
        f"(FECHA_CORTE fija, no HOY={HOY})"
    )

grilla: pd.DatetimeIndex = construir_grilla(
    VENTANA_INICIO, FECHA_CORTE
)
mascaras: dict[str, pd.Series] = {}
df_diarias: pd.DataFrame = pd.DataFrame(index=grilla)
df_diarias.index.name = "fecha"

for _col, _serie in series_full.items():
    try:
        _s_al, _mask = alinear_serie(_serie, grilla)
        df_diarias[_col] = _s_al
        mascaras[_col] = _mask
        if DEBUG:
            _nr = int((~_mask & _s_al.notna()).sum())
            _nf = int(_mask.sum())
            print(
                f"[DEBUG] {_col}: "
                f"{_nr} reales, {_nf} ffill"
            )
    except Exception as _e:
        if DEBUG:
            print(f"[DEBUG] Error alineando {_col}: {_e}")

if DEBUG:
    print(
        f"[DEBUG] df_diarias: {df_diarias.shape} — "
        f"{df_diarias.index.min().date()} → "
        f"{df_diarias.index.max().date()}"
    )

# ── 4. Series largas (historia completa pre-ventana) ──────────────

_ts_ventana = pd.Timestamp(VENTANA_INICIO)
_cols_largas: list[str] = [
    col for col, s in series_full.items()
    if not s[s.index < _ts_ventana].empty
]
df_largas: pd.DataFrame = (
    pd.DataFrame({c: series_full[c] for c in _cols_largas})
    if _cols_largas else pd.DataFrame()
)

if DEBUG:
    if not df_largas.empty:
        print(
            f"\n[DEBUG] Series largas: {len(_cols_largas)} cols, "
            f"{df_largas.shape}"
        )
    else:
        print("\n[DEBUG] Sin series con historia pre-ventana")

# ── 5. Cobertura ──────────────────────────────────────────────────

cobertura: pd.DataFrame = calcular_cobertura(
    df_diarias, mascaras, FECHA_EXTRACCION
)

if DEBUG:
    print(
        f"\n[DEBUG] Cobertura: {len(cobertura)} cols registradas"
    )

# ── 6. Verificación cruzada contra main ──────────────────────────

verificar_contra_main(df_diarias)

"""# 6. Save"""

if SAVE_DATA:
    # CAMBIO E2E-01-01: validar la dependencia de historia larga antes de
    # escribir cualquier salida. Si falta, abortar con el nombre del archivo
    # esperado; no crear un CSV vacío ni reutilizar uno de otra ejecución.
    _historia_previa_disponible = (
        not df_largas.empty
        and df_largas.loc[
            df_largas.index < _ts_ventana
        ].notna().to_numpy().any()
    )
    if not _historia_previa_disponible:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 1: no se puede generar {ARCHIVO_LARGAS}: "
            f"no hay valores históricos anteriores a {VENTANA_INICIO}. "
            "No se escribieron las salidas del Job 1. "
            "Revisar log_descarga y las fuentes FRED/ECB/CoinGecko "
            "antes de ejecutar el Job 3."
        )

    os.makedirs(DIR_PROCESSED, exist_ok=True)
    os.makedirs(DIR_REPORTES, exist_ok=True)

    df_diarias.to_csv(ARCHIVO_DIARIAS)
    if DEBUG:
        print(f"\n[DEBUG] guardado: {ARCHIVO_DIARIAS}")
    registrar_huella(ARCHIVO_DIARIAS, ARCHIVO_HUELLAS)  # CAMBIO H3-2
    congelar_snapshot([ARCHIVO_DIARIAS], "Job 1")  # CAMBIO B1-2

    if not df_largas.empty:
        df_largas.to_csv(ARCHIVO_LARGAS)
        if DEBUG:
            print(f"[DEBUG] guardado: {ARCHIVO_LARGAS}")
        registrar_huella(ARCHIVO_LARGAS, ARCHIVO_HUELLAS)  # CAMBIO H3-2
        congelar_snapshot([ARCHIVO_LARGAS], "Job 1")  # CAMBIO B1-2
    elif DEBUG:
        print("[DEBUG] df_largas vacío, no se guarda")

    cobertura.to_csv(ARCHIVO_COBERTURA, index=False)
    if DEBUG:
        print(f"[DEBUG] guardado: {ARCHIVO_COBERTURA}")
elif DEBUG:
    print(
        "\n[DEBUG] SAVE_DATA=False — "
        "datos solo en memoria"
    )

