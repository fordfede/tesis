"""# Job 0 - Verificación de Fuentes RWA (tokenización por red)"""

# 1. Configuración

DEBUG: bool = True
SAVE_DATA: bool = True

# Flags propios del notebook
PROBAR_DEFILLAMA: bool = True
PROBAR_RWA_XYZ: bool = True
PROBAR_COINGECKO: bool = True
PROBAR_COINMETRICS: bool = True


# 2. Imports

import datetime
import os
import time
from typing import Any

import pandas as pd
import requests


from google.colab import drive
drive.mount('/content/drive')

# CAMBIO H3-1 (Job 0): versiones del entorno al inicio del
# script, para poder reproducir una corrida exacta. xgboost es
# opcional en este Job (no se usa acá): si no está instalado se
# registra como tal con [DEBUG], no se corta la ejecución.
import sys

print(f"[DEBUG] Python: {sys.version.split()[0]}")
print(f"[DEBUG] pandas: {pd.__version__}")
for _lib_nombre in (
    "numpy", "scipy", "statsmodels", "sklearn", "xgboost"
):
    try:
        _lib = __import__(_lib_nombre)
        print(f"[DEBUG] {_lib_nombre}: {_lib.__version__}")
    except ImportError:
        print(f"[DEBUG] {_lib_nombre}: no instalado")

"""# 3. Paths y Constantes"""

# CAMBIO H2-1 (Job 0): BASE_DIR único, configurable por variable
# de entorno TESIS_BASE_DIR (valor actual como default). Mismo
# patrón repetido en Job 1.a, 1.b, 2 y 8 (cada Job sigue siendo
# ejecutable de forma independiente en Colab, por eso se repite
# la expresión en lugar de importar la variable entre celdas).
BASE_DIR: str = os.environ.get(
    "TESIS_BASE_DIR", "/content/drive/MyDrive/Tesis/expansion"
)
DIR_REPORTES: str = os.path.join(BASE_DIR, "outputs", "reportes")
ARCHIVO_SALIDA: str = os.path.join(
    DIR_REPORTES, "rwa_viabilidad.csv"
)
ARCHIVO_PROTOCOLOS: str = os.path.join(


    DIR_REPORTES, "rwa_protocolos_defillama.csv"
)

TIMEOUT: int = 40
PAUSA: float = 1.5          # segundos entre requests (rate limit)
VENTANA_INICIO: str = "2026-01-01"
HOY: str = datetime.date.today().isoformat()
FECHA_VERIFICACION: str = HOY
MAX_FECHA_RECIENTE: str = "2026-06-01"  # umbral mínimo de max_fecha

# --- Endpoints base ---
LLAMA_BASE: str = "https://api.llama.fi"
LLAMA_PROTOCOLS: str = f"{LLAMA_BASE}/protocols"
LLAMA_PROTOCOL: str = f"{LLAMA_BASE}/protocol"          # /{slug}
LLAMA_CHAIN_TVL: str = f"{LLAMA_BASE}/v2/historicalChainTvl"
LLAMA_STABLE: str = "https://stablecoins.llama.fi"
COINGECKO_BASE: str = "https://api.coingecko.com/api/v3"
CM_BASE: str = (
    "https://community-api.coinmetrics.io"
    "/v4/timeseries/asset-metrics"
)

# Candidatos de API pública de RWA.xyz (se prueban en orden).
# Si ninguno responde sin key → NO DISPONIBLE (no inventar).
RWA_XYZ_CANDIDATOS: list[str] = [
    "https://api.rwa.xyz/v1/assets",
    "https://api.rwa.xyz/assets",
    "https://app.rwa.xyz/api/assets",
    "https://www.rwa.xyz/api/assets",
]

# Categoría DeFiLlama que agrupa la tokenización de RWA
CATEGORIA_RWA: str = "RWA"

# Redes de interés del proyecto (nombres DeFiLlama)
REDES_INTERES: list[str] = [
    "Stellar",
    "XRPL",
    "Ripple",
    "Ethereum",
    "Solana",
    "Polygon",
    "Avalanche",
    "Aptos",
    "Arbitrum",
    "Algorand",
    "Hedera",
]

# Cuántos protocolos RWA (por TVL) se verifican en detalle
N_PROTOCOLOS_MUESTRA: int = 8

# Stablecoin RWA ya validada en fases previas (USDY)
USDY_STABLECOIN_ID: int = 129
CHAINS_STABLECOIN: list[str] = ["Ethereum", "Solana", "Aptos"]

# Categorías CoinGecko candidatas para RWA
CG_CATEGORIAS: list[str] = [
    "real-world-assets-rwa",
    "tokenized-treasury-bills",
    "tokenized-gold",
]

# Tokens RWA de muestra: símbolo → (pista de nombre, tipo activo)
CG_TOKENS_RWA: dict[str, tuple[str, str]] = {
    "USDY": ("ondo", "treasuries"),
    "OUSG": ("ondo", "treasuries"),
    "PAXG": ("paxos", "oro"),
    "XAUT": ("tether", "oro"),
}

# Assets RWA en CoinMetrics Community (símbolo → tipo activo)
CM_ASSETS_RWA: dict[str, str] = {
    "paxg": "oro",
    "usdy": "treasuries",
}
CM_METRICS_RWA: list[str] = ["SplyCur", "CapMrktCurUSD"]

"""# 4. Funciones"""


def probar_endpoint(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int = TIMEOUT,
) -> dict[str, Any]:
    """
    Realiza una request GET real y registra el resultado.

    Args:
        url: URL del endpoint.
        params: Query parameters opcionales.
        timeout: Segundos de espera máxima.

    Returns:
        Dict con claves: ok (bool), status (int|None),
        data (JSON parseado o None), error (str|None).
    """
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            try:
                data = r.json()
            except Exception:
                data = None
            return {
                "ok": True,
                "status": r.status_code,
                "data": data,
                "error": None,
            }
        return {
            "ok": False,
            "status": r.status_code,
            "data": None,
            "error": f"HTTP {r.status_code}",
        }
    except requests.Timeout:
        return {
            "ok": False, "status": None,
            "data": None, "error": "timeout",
        }
    except requests.RequestException as e:
        return {
            "ok": False, "status": None,
            "data": None, "error": str(e),
        }


def probar_fuente(
    nombre: str,
    url: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Prueba genérica de una fuente: hace la request y describe
    de forma superficial si la respuesta parece o no una serie.

    Se usa para fuentes sin parser dedicado (exploración).

    Args:
        nombre: Nombre legible de la fuente.
        url: Endpoint a probar.
        params: Query parameters opcionales.

    Returns:
        Dict con ok, status, n_obs, min_fecha, max_fecha,
        tiene_serie_temporal, granularidad, error.
    """
    if DEBUG:
        print(f"[DEBUG] probar_fuente: {nombre} -> {url}")
    res = probar_endpoint(url, params)
    salida: dict[str, Any] = {
        "nombre": nombre,
        "ok": res["ok"],
        "status": res["status"],
        "n_obs": 0,
        "min_fecha": "s/d",
        "max_fecha": "s/d",
        "tiene_serie_temporal": False,
        "granularidad": "s/d",
        "error": res["error"],
    }
    if not res["ok"]:
        return salida
    data = res.get("data")
    fechas = _extraer_fechas(data)
    salida["n_obs"] = len(fechas)
    if fechas:
        fechas_ord = sorted(fechas)
        salida["min_fecha"], salida["max_fecha"] = _rango_fechas(
            fechas_ord
        )
        salida["tiene_serie_temporal"] = len(fechas_ord) > 1
        salida["granularidad"] = _inferir_granularidad(fechas_ord)
    else:
        salida["granularidad"] = "puntual (sin serie)"
    return salida


def _extraer_fechas(data: Any) -> list[str]:
    """
    Busca en una respuesta JSON una lista de observaciones con
    campo de fecha y devuelve esas fechas en ISO.

    Reconoce claves 'date', 'time', 'timestamp' y valores
    epoch (int/str numérico) o fecha ISO.

    Args:
        data: JSON ya parseado.

    Returns:
        Lista de fechas ISO (YYYY-MM-DD); vacía si no aplica.
    """
    candidatos: list[Any] = []
    if isinstance(data, list):
        candidatos = data
    elif isinstance(data, dict):
        for clave in ("data", "results", "chart", "chart_data",
                      "items", "series"):
            valor = data.get(clave)
            if isinstance(valor, list) and valor:
                candidatos = valor
                break
    fechas: list[str] = []
    for obs in candidatos:
        if not isinstance(obs, dict):
            continue
        for clave in ("date", "time", "timestamp"):
            if clave in obs:
                fecha = _a_fecha_iso(obs[clave])
                if fecha:
                    fechas.append(fecha)
                break
    return fechas


def _a_fecha_iso(valor: Any) -> str | None:
    """
    Normaliza un valor de fecha (epoch seg/ms o string ISO) a
    fecha ISO YYYY-MM-DD.

    Args:
        valor: Epoch numérico o cadena de fecha.

    Returns:
        Fecha ISO o None si no se pudo interpretar.
    """
    try:
        if isinstance(valor, (int, float)):
            segundos = float(valor)
            if segundos > 1e11:      # epoch en milisegundos
                segundos = segundos / 1000.0
            return datetime.datetime.utcfromtimestamp(
                segundos
            ).date().isoformat()
        if isinstance(valor, str):
            if valor.isdigit():
                return _a_fecha_iso(int(valor))
            return valor[:10]
    except Exception:
        return None
    return None


def _rango_fechas(fechas: list[str]) -> tuple[str, str]:
    """Extrae min y max de una lista de fechas ISO (YYYY-MM-DD)."""
    validas = sorted(f for f in fechas if f and f != "nan")
    if not validas:
        return "s/d", "s/d"
    return validas[0], validas[-1]


def _inferir_granularidad(fechas: list[str]) -> str:
    """
    Infiere la granularidad de una serie por el espaciado modal
    entre observaciones consecutivas.

    Args:
        fechas: Fechas ISO ordenadas ascendentemente.

    Returns:
        "diaria" | "semanal" | "mensual" | "cada N días" |
        "puntual (sin serie)" | "desconocida"
    """
    if len(fechas) < 2:
        return "puntual (sin serie)"
    diffs: list[int] = []
    for anterior, actual in zip(fechas, fechas[1:]):
        try:
            d1 = datetime.date.fromisoformat(anterior)
            d2 = datetime.date.fromisoformat(actual)
        except Exception:
            continue
        delta = abs((d2 - d1).days)
        if delta > 0:
            diffs.append(delta)
    if not diffs:
        return "desconocida"
    modal = max(set(diffs), key=diffs.count)
    if modal == 1:
        return "diaria"
    if modal == 7:
        return "semanal"
    if 28 <= modal <= 31:
        return "mensual"
    return f"cada {modal} días"


def registro_estandar(
    fuente: str,
    endpoint: str,
    ok: bool,
    red: str = "TODAS",
    tipo_activo: str = "TODOS",
    desglose_red: bool = False,
    desglose_tipo: bool = False,
    granularidad: str = "s/d",
    min_fecha: str = "s/d",
    max_fecha: str = "s/d",
    n_obs: int = 0,
    motivo: str = "",
) -> dict[str, Any]:
    """
    Construye un registro de verificación en formato estándar.

    Args:
        fuente: Nombre de la fuente/serie verificada.
        endpoint: URL efectivamente probada.
        ok: Si la API respondió con datos utilizables.
        red: Red/chain de la fila ("TODAS" si es agregado,
            "N/D" si la fuente no desagrega).
        tipo_activo: Tipo de activo RWA de la fila.
        desglose_red: Si la fuente permite desagregar por red.
        desglose_tipo: Si permite desagregar por tipo de activo.
        granularidad: Granularidad observada de la serie.
        min_fecha: Primera fecha observada (ISO).
        max_fecha: Última fecha observada (ISO).
        n_obs: Número de observaciones.
        motivo: Evidencia o motivo de la limitación.

    Returns:
        Dict con el registro estándar.
    """
    return {
        "fuente": fuente,
        "red": red,
        "tipo_activo": tipo_activo,
        "endpoint": endpoint,
        "ok": ok,
        "desglose_red": desglose_red,
        "desglose_tipo": desglose_tipo,
        "granularidad": granularidad,
        "min_fecha": min_fecha,
        "max_fecha": max_fecha,
        "n_obs": n_obs,
        "motivo": motivo,
        "fecha_verificacion": FECHA_VERIFICACION,
    }


# ── DeFiLlama ─────────────────────────────────────────────────────


def listar_protocolos_rwa() -> pd.DataFrame:
    """
    Lista los protocolos de categoría RWA en DeFiLlama con su
    TVL actual, chains y tags (tipo de activo declarado).

    Returns:
        DataFrame con columnas: slug, name, category, tags,
        chains, tvl, n_chains. Vacío si el endpoint falla.
    """
    if DEBUG:
        print(f"[DEBUG] DeFiLlama /protocols (categoría "
              f"'{CATEGORIA_RWA}') ...")
    res = probar_endpoint(LLAMA_PROTOCOLS)
    if not res["ok"] or not isinstance(res.get("data"), list):
        if DEBUG:
            print(f"[DEBUG]   falló /protocols: {res['error']}")
        return pd.DataFrame()
    filas: list[dict[str, Any]] = []
    for proto in res["data"]:
        if not isinstance(proto, dict):
            continue
        if proto.get("category") != CATEGORIA_RWA:
            continue
        chains = proto.get("chains") or []
        tags = proto.get("tags") or []
        filas.append({
            "slug": proto.get("slug", ""),
            "name": proto.get("name", ""),
            "category": proto.get("category", ""),
            "tags": ",".join(str(t) for t in tags),
            "chains": ",".join(str(c) for c in chains),
            "n_chains": len(chains),
            "tvl": proto.get("tvl") or 0.0,
        })
    df = pd.DataFrame(filas)
    if not df.empty:
        df = df.sort_values("tvl", ascending=False)
        df = df.reset_index(drop=True)
    return df


def serie_tvl_protocolo(slug: str) -> dict[str, Any]:
    """
    Trae la serie histórica de TVL de un protocolo DeFiLlama,
    desglosada por chain.

    Args:
        slug: Slug del protocolo (ej. "blackrock-buidl").

    Returns:
        Dict con claves: ok (bool), endpoint (str),
        error (str|None), tags (list[str]),
        series (dict chain → dict fecha_iso → usd).
    """
    url = f"{LLAMA_PROTOCOL}/{slug}"
    res = probar_endpoint(url)
    salida: dict[str, Any] = {
        "ok": False,
        "endpoint": url,
        "error": res["error"],
        "tags": [],
        "series": {},
    }
    if not res["ok"] or not isinstance(res.get("data"), dict):
        return salida
    data: dict[str, Any] = res["data"]
    salida["tags"] = [
        str(t) for t in (data.get("tags") or [])
    ]
    chain_tvls = data.get("chainTvls") or {}
    if not isinstance(chain_tvls, dict):
        return salida
    series: dict[str, dict[str, float]] = {}
    for chain, contenido in chain_tvls.items():
        # Se excluyen agregados derivados (borrowed, staking, ...)
        if "-" in str(chain):
            continue
        if not isinstance(contenido, dict):
            continue
        observaciones = contenido.get("tvl") or []
        if not isinstance(observaciones, list):
            continue
        por_fecha: dict[str, float] = {}
        for obs in observaciones:
            if not isinstance(obs, dict):
                continue
            fecha = _a_fecha_iso(obs.get("date"))
            valor = obs.get("totalLiquidityUSD")
            if fecha is None or valor is None:
                continue
            # Si hay más de una obs por día, queda la última
            por_fecha[fecha] = float(valor)
        if por_fecha:
            series[str(chain)] = por_fecha
    salida["series"] = series
    salida["ok"] = bool(series)
    return salida


def verificar_protocolo_rwa(
    slug: str,
    nombre: str,
    tags: str,
) -> list[dict[str, Any]]:
    """
    Verifica la serie histórica de TVL por chain de un protocolo
    RWA concreto: una fila por (protocolo, chain).

    Args:
        slug: Slug del protocolo en DeFiLlama.
        nombre: Nombre legible del protocolo.
        tags: Tags declarados (tipo de activo), separados por coma.

    Returns:
        Lista de registros estándar (una por chain con datos).
    """
    fuente = f"DeFiLlama/protocolo/{nombre}"
    if DEBUG:
        print(f"[DEBUG] {fuente} (slug={slug}) ...")
    resultado = serie_tvl_protocolo(slug)
    tipo = tags or "s/d"
    if not resultado["ok"]:
        motivo = (
            f"sin serie por chain: {resultado['error'] or 'chainTvls vacío'}"
        )
        return [registro_estandar(
            fuente=fuente, endpoint=resultado["endpoint"],
            ok=False, red="N/D", tipo_activo=tipo,
            motivo=motivo,
        )]
    registros: list[dict[str, Any]] = []
    for chain, por_fecha in resultado["series"].items():
        fechas = sorted(por_fecha.keys())
        min_f, max_f = _rango_fechas(fechas)
        registros.append(registro_estandar(
            fuente=fuente,
            endpoint=resultado["endpoint"],
            ok=True,
            red=chain,
            tipo_activo=tipo,
            desglose_red=True,
            desglose_tipo=bool(resultado["tags"]),
            granularidad=_inferir_granularidad(fechas),
            min_fecha=min_f, max_fecha=max_f,
            n_obs=len(fechas),
            motivo=(
                "chainTvls[chain].tvl: serie de "
                "totalLiquidityUSD"
            ),
        ))
    return registros


def agregar_series_por_chain(
    series_protocolos: list[tuple[str, dict[str, dict[str, float]]]],
) -> dict[str, dict[str, float]]:
    """
    Suma las series de TVL de varios protocolos RWA por chain,
    para construir la señal agregada "valor RWA tokenizado por red".

    Args:
        series_protocolos: Lista de (nombre_protocolo, series),
            donde series es dict chain → dict fecha → usd.

    Returns:
        Dict chain → dict fecha_iso → usd agregado.
    """
    agregado: dict[str, dict[str, float]] = {}
    for _, series in series_protocolos:
        for chain, por_fecha in series.items():
            destino = agregado.setdefault(chain, {})
            for fecha, valor in por_fecha.items():
                destino[fecha] = destino.get(fecha, 0.0) + valor
    return agregado


def agregar_series_por_tipo(
    series_por_tipo: list[tuple[str, dict[str, dict[str, float]]]],
) -> dict[str, dict[str, float]]:
    """
    Suma las series de TVL de varios protocolos RWA por tipo de
    activo (tag declarado en DeFiLlama), agregando todas sus chains.

    Args:
        series_por_tipo: Lista de (tipo_activo, series), donde
            series es dict chain → dict fecha → usd.

    Returns:
        Dict tipo_activo → dict fecha_iso → usd agregado.
    """
    agregado: dict[str, dict[str, float]] = {}
    for tipo, series in series_por_tipo:
        destino = agregado.setdefault(tipo or "s/d", {})
        for _, por_fecha in series.items():
            for fecha, valor in por_fecha.items():
                destino[fecha] = destino.get(fecha, 0.0) + valor
    return agregado


def registros_desde_agregado(
    fuente: str,
    endpoint: str,
    agregado: dict[str, dict[str, float]],
    dimension: str,
    motivo: str,
) -> list[dict[str, Any]]:
    """
    Convierte un agregado (clave → serie) en registros estándar.

    Args:
        fuente: Nombre de la fuente/serie agregada.
        endpoint: Endpoint del que provienen los datos.
        agregado: Dict clave → dict fecha_iso → valor.
        dimension: "red" o "tipo"; define en qué columna va la
            clave del agregado.
        motivo: Evidencia asociada a la construcción.

    Returns:
        Lista de registros estándar, uno por clave.
    """
    registros: list[dict[str, Any]] = []
    for clave, por_fecha in agregado.items():
        fechas = sorted(por_fecha.keys())
        min_f, max_f = _rango_fechas(fechas)
        registros.append(registro_estandar(
            fuente=fuente,
            endpoint=endpoint,
            ok=bool(fechas),
            red=clave if dimension == "red" else "TODAS",
            tipo_activo=clave if dimension == "tipo" else "TODOS",
            desglose_red=(dimension == "red"),
            desglose_tipo=(dimension == "tipo"),
            granularidad=_inferir_granularidad(fechas),
            min_fecha=min_f, max_fecha=max_f,
            n_obs=len(fechas),
            motivo=motivo,
        ))
    return registros


def verificar_llama_chain_tvl(chain: str) -> dict[str, Any]:
    """
    Verifica el TVL histórico total de una chain en DeFiLlama.

    Es TVL DeFi total, NO específico de RWA: sirve como
    denominador/contexto, no como señal de tokenización.

    Args:
        chain: Nombre de la chain en DeFiLlama (ej. "Stellar").

    Returns:
        Registro estándar.
    """
    url = f"{LLAMA_CHAIN_TVL}/{chain}"
    fuente = f"DeFiLlama/chainTVL/{chain}"
    if DEBUG:
        print(f"[DEBUG] {fuente} ...")
    res = probar_endpoint(url)
    if not res["ok"] or not isinstance(res.get("data"), list):
        return registro_estandar(
            fuente=fuente, endpoint=url, ok=False, red=chain,
            tipo_activo="N/D",
            motivo=f"endpoint error: {res['error']}",
        )
    fechas = sorted(
        f for f in (
            _a_fecha_iso(o.get("date"))
            for o in res["data"] if isinstance(o, dict)
        ) if f
    )
    if not fechas:
        return registro_estandar(
            fuente=fuente, endpoint=url, ok=False, red=chain,
            tipo_activo="N/D", motivo="respuesta sin fechas",
        )
    min_f, max_f = _rango_fechas(fechas)
    return registro_estandar(
        fuente=fuente, endpoint=url, ok=True, red=chain,
        tipo_activo="N/D",
        desglose_red=True, desglose_tipo=False,
        granularidad=_inferir_granularidad(fechas),
        min_fecha=min_f, max_fecha=max_f, n_obs=len(fechas),
        motivo=(
            "TVL DeFi total de la chain (contexto/denominador); "
            "NO es específico de RWA"
        ),
    )


def verificar_stablecoin_rwa_por_chain(
    stablecoin_id: int,
    chain: str,
    etiqueta: str,
) -> dict[str, Any]:
    """
    Verifica la serie diaria de circulación de una stablecoin RWA
    en una chain (endpoint ya validado en fases previas).

    Args:
        stablecoin_id: Id de la stablecoin en DeFiLlama.
        chain: Chain a consultar (ej. "Ethereum").
        etiqueta: Nombre legible del token (ej. "USDY").

    Returns:
        Registro estándar.
    """
    url = f"{LLAMA_STABLE}/stablecoincharts/{chain}"
    params: dict[str, Any] = {"stablecoin": stablecoin_id}
    fuente = f"DeFiLlama/stablecoin/{etiqueta}"
    if DEBUG:
        print(f"[DEBUG] {fuente} en {chain} ...")
    res = probar_endpoint(url, params)
    if not res["ok"] or not isinstance(res.get("data"), list):
        return registro_estandar(
            fuente=fuente, endpoint=url, ok=False, red=chain,
            tipo_activo="treasuries",
            motivo=f"endpoint error: {res['error']}",
        )
    fechas: list[str] = []
    for obs in res["data"]:
        if not isinstance(obs, dict):
            continue
        fecha = _a_fecha_iso(obs.get("date"))
        if fecha:
            fechas.append(fecha)
    if not fechas:
        return registro_estandar(
            fuente=fuente, endpoint=url, ok=False, red=chain,
            tipo_activo="treasuries",
            motivo="sin observaciones para esa chain/stablecoin",
        )
    fechas = sorted(fechas)
    min_f, max_f = _rango_fechas(fechas)
    return registro_estandar(
        fuente=fuente, endpoint=url, ok=True, red=chain,
        tipo_activo="treasuries",
        desglose_red=True, desglose_tipo=True,
        granularidad=_inferir_granularidad(fechas),
        min_fecha=min_f, max_fecha=max_f, n_obs=len(fechas),
        motivo=(
            f"stablecoincharts/{chain}?stablecoin="
            f"{stablecoin_id}: circulación diaria"
        ),
    )


# ── RWA.xyz ───────────────────────────────────────────────────────


def verificar_rwa_xyz() -> dict[str, Any]:
    """
    Prueba los candidatos de API pública de RWA.xyz.

    Si ninguno responde 200 con datos (típicamente requiere key
    o no expone API abierta), se registra NO DISPONIBLE con el
    detalle de lo probado. No se inventan endpoints.

    Returns:
        Registro estándar.
    """
    fuente = "RWA.xyz"
    intentos: list[str] = []
    for url in RWA_XYZ_CANDIDATOS:
        prueba = probar_fuente(fuente, url)
        intentos.append(
            f"{url} → {prueba['status'] or prueba['error']}"
        )
        if prueba["ok"] and prueba["n_obs"] > 0:
            return registro_estandar(
                fuente=fuente, endpoint=url, ok=True,
                red="N/D", tipo_activo="N/D",
                desglose_red=False, desglose_tipo=False,
                granularidad=prueba["granularidad"],
                min_fecha=prueba["min_fecha"],
                max_fecha=prueba["max_fecha"],
                n_obs=prueba["n_obs"],
                motivo=(
                    "respondió con datos; revisar manualmente "
                    "el esquema para desgloses"
                ),
            )
        time.sleep(PAUSA)
    return registro_estandar(
        fuente=fuente, endpoint=RWA_XYZ_CANDIDATOS[0], ok=False,
        red="N/D", tipo_activo="N/D",
        motivo=(
            "sin API pública accesible sin key. Probados: "
            + " | ".join(intentos)
        ),
    )


# ── CoinGecko ─────────────────────────────────────────────────────


def verificar_coingecko_categorias() -> list[dict[str, Any]]:
    """
    Verifica el market cap por categoría RWA en CoinGecko.

    /coins/categories devuelve un SNAPSHOT (market cap actual),
    sin historia: se registra explícitamente como tal.

    Returns:
        Lista de registros estándar (uno por categoría buscada).
    """
    url = f"{COINGECKO_BASE}/coins/categories"
    if DEBUG:
        print("[DEBUG] CoinGecko /coins/categories ...")
    res = probar_endpoint(url)
    registros: list[dict[str, Any]] = []
    if not res["ok"] or not isinstance(res.get("data"), list):
        for categoria in CG_CATEGORIAS:
            registros.append(registro_estandar(
                fuente=f"CoinGecko/categoria/{categoria}",
                endpoint=url, ok=False, red="N/D",
                tipo_activo=categoria,
                motivo=f"endpoint error: {res['error']}",
            ))
        return registros
    por_id: dict[str, dict[str, Any]] = {
        str(c.get("id", "")): c
        for c in res["data"] if isinstance(c, dict)
    }
    for categoria in CG_CATEGORIAS:
        info = por_id.get(categoria)
        if info is None:
            registros.append(registro_estandar(
                fuente=f"CoinGecko/categoria/{categoria}",
                endpoint=url, ok=False, red="N/D",
                tipo_activo=categoria,
                motivo="categoría inexistente en la respuesta",
            ))
            continue
        mcap = info.get("market_cap")
        registros.append(registro_estandar(
            fuente=f"CoinGecko/categoria/{categoria}",
            endpoint=url, ok=True, red="N/D",
            tipo_activo=categoria,
            desglose_red=False, desglose_tipo=True,
            granularidad="puntual (sin serie)",
            n_obs=1,
            motivo=(
                f"SNAPSHOT: market_cap actual = {mcap}; "
                "sin historia por API"
            ),
        ))
    return registros


def _buscar_coingecko_id(
    simbolo: str,
    nombre_pista: str = "",
) -> str | None:
    """
    Busca el id canónico de CoinGecko para un símbolo.

    Args:
        simbolo: Símbolo exacto (ej. "OUSG").
        nombre_pista: Keyword del nombre para desambiguar.

    Returns:
        El id (str) o None si no se encontró.
    """
    url = f"{COINGECKO_BASE}/search"
    res = probar_endpoint(url, {"query": simbolo})
    if not res["ok"]:
        return None
    coins: list[dict] = (res.get("data") or {}).get("coins", [])
    s_up = simbolo.upper()
    candidatos = [
        c for c in coins
        if str(c.get("symbol", "")).upper() == s_up
    ]
    if not candidatos:
        return None
    if len(candidatos) == 1 or not nombre_pista:
        return str(candidatos[0].get("id", ""))
    pista = nombre_pista.lower()
    filtrados = [
        c for c in candidatos
        if pista in str(c.get("name", "")).lower()
    ]
    return str((filtrados or candidatos)[0].get("id", ""))


def verificar_coingecko_token_rwa(
    simbolo: str,
    nombre_pista: str,
    tipo_activo: str,
) -> dict[str, Any]:
    """
    Verifica la serie histórica de market cap de un token RWA en
    CoinGecko (proxy del valor tokenizado, agregado entre redes).

    Args:
        simbolo: Símbolo del token (ej. "PAXG").
        nombre_pista: Keyword del emisor para desambiguar.
        tipo_activo: Tipo de activo subyacente (ej. "oro").

    Returns:
        Registro estándar.
    """
    fuente = f"CoinGecko/token/{simbolo}"
    if DEBUG:
        print(f"[DEBUG] {fuente} ...")
    cg_id = _buscar_coingecko_id(simbolo, nombre_pista)
    if not cg_id:
        return registro_estandar(
            fuente=fuente, endpoint=f"{COINGECKO_BASE}/search",
            ok=False, red="N/D", tipo_activo=tipo_activo,
            motivo=f"no se encontró id para símbolo {simbolo}",
        )
    time.sleep(PAUSA)
    url = f"{COINGECKO_BASE}/coins/{cg_id}/market_chart"
    params: dict[str, Any] = {
        "vs_currency": "usd",
        "days": "365",
        "interval": "daily",
    }
    res = probar_endpoint(url, params)
    if not res["ok"] or not isinstance(res.get("data"), dict):
        return registro_estandar(
            fuente=fuente, endpoint=url, ok=False, red="N/D",
            tipo_activo=tipo_activo,
            motivo=f"market_chart error: {res['error']}",
        )
    puntos = res["data"].get("market_caps") or []
    fechas: list[str] = []
    for punto in puntos:
        if isinstance(punto, list) and punto:
            fecha = _a_fecha_iso(punto[0])
            if fecha:
                fechas.append(fecha)
    if not fechas:
        return registro_estandar(
            fuente=fuente, endpoint=url, ok=False, red="N/D",
            tipo_activo=tipo_activo,
            motivo="market_caps vacío",
        )
    fechas = sorted(fechas)
    min_f, max_f = _rango_fechas(fechas)
    return registro_estandar(
        fuente=fuente, endpoint=url, ok=True,
        red="N/D", tipo_activo=tipo_activo,
        desglose_red=False, desglose_tipo=True,
        granularidad=_inferir_granularidad(fechas),
        min_fecha=min_f, max_fecha=max_f, n_obs=len(fechas),
        motivo=(
            f"id={cg_id}; market_cap diario agregado entre redes "
            "(no desagrega por chain)"
        ),
    )


# ── CoinMetrics ───────────────────────────────────────────────────


def verificar_coinmetrics_rwa(
    asset: str,
    tipo_activo: str,
) -> dict[str, Any]:
    """
    Verifica si CoinMetrics Community expone supply / market cap
    diario de un token RWA.

    Args:
        asset: Símbolo del asset en CoinMetrics (ej. "paxg").
        tipo_activo: Tipo de activo subyacente.

    Returns:
        Registro estándar.
    """
    fuente = f"CoinMetrics/{asset.upper()}"
    params: dict[str, Any] = {
        "assets": asset,
        "metrics": ",".join(CM_METRICS_RWA),
        "frequency": "1d",
        "start_time": VENTANA_INICIO,
        "end_time": HOY,
        "page_size": 5000,
    }
    if DEBUG:
        print(f"[DEBUG] {fuente} ...")
    res = probar_endpoint(CM_BASE, params)
    if not res["ok"]:
        return registro_estandar(
            fuente=fuente, endpoint=CM_BASE, ok=False,
            red="N/D", tipo_activo=tipo_activo,
            motivo=f"endpoint error: {res['error']}",
        )
    observaciones: list[dict] = (
        (res.get("data") or {}).get("data", [])
    )
    if not observaciones:
        return registro_estandar(
            fuente=fuente, endpoint=CM_BASE, ok=False,
            red="N/D", tipo_activo=tipo_activo,
            motivo=(
                "sin datos para el asset/métricas en la ventana "
                "(asset posiblemente fuera de Community API)"
            ),
        )
    fechas = sorted(
        str(o.get("time", ""))[:10] for o in observaciones
    )
    min_f, max_f = _rango_fechas(fechas)
    metricas_con_datos = [
        m for m in CM_METRICS_RWA
        if any(o.get(m) not in (None, "") for o in observaciones)
    ]
    return registro_estandar(
        fuente=fuente, endpoint=CM_BASE, ok=True,
        red="N/D", tipo_activo=tipo_activo,
        desglose_red=False, desglose_tipo=True,
        granularidad="diaria",
        min_fecha=min_f, max_fecha=max_f,
        n_obs=len(observaciones),
        motivo=(
            "métricas con datos: "
            f"{','.join(metricas_con_datos) or 'ninguna'}"
        ),
    )


# ── Matriz de viabilidad ──────────────────────────────────────────


def construir_matriz_viabilidad(
    registros: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Construye el DataFrame de viabilidad con criterios V1..V5 y
    veredicto por fila (fuente × red × tipo de activo).

    Criterios:
        V1_tiene_api     : la API responde y devuelve datos.
        V2_desglose_red  : desagrega por red/chain.
        V3_desglose_tipo : desagrega por tipo de activo.
        V4_serie_temporal: hay historia con frecuencia regular
                           ≤ mensual (alineable a diaria).
        V5_cobertura     : min_fecha <= VENTANA_INICIO y
                           max_fecha >= MAX_FECHA_RECIENTE.

    Veredicto:
        ENTRA         : V1 ∧ V4 ∧ V5 ∧ (V2 ∨ V3)
        ENTRA_PARCIAL : V1 ∧ V4 ∧ V5 ∧ ¬V2 ∧ ¬V3
        NO ENTRA      : V1 pero falla V4 o V5
        NO DISPONIBLE : ¬V1

    Args:
        registros: Lista de registros estándar.

    Returns:
        DataFrame con una fila por registro.
    """
    frecuencias_ok: set[str] = {"diaria", "semanal", "mensual"}
    filas: list[dict[str, Any]] = []
    for reg in registros:
        v1: bool = bool(reg["ok"])
        gran: str = str(reg.get("granularidad", ""))
        v2: bool = bool(reg.get("desglose_red", False))
        v3: bool = bool(reg.get("desglose_tipo", False))
        v4: bool = gran in frecuencias_ok and int(
            reg.get("n_obs", 0)
        ) > 1
        min_ok = (
            reg.get("min_fecha") not in ("s/d", "", None)
            and str(reg["min_fecha"]) <= VENTANA_INICIO
        )
        max_ok = (
            reg.get("max_fecha") not in ("s/d", "", None)
            and str(reg["max_fecha"]) >= MAX_FECHA_RECIENTE
        )
        v5: bool = bool(min_ok and max_ok)

        razones: list[str] = []
        if not v1:
            veredicto = "NO DISPONIBLE"
        elif v4 and v5:
            veredicto = "ENTRA" if (v2 or v3) else "ENTRA_PARCIAL"
            if veredicto == "ENTRA_PARCIAL":
                razones.append("sin desglose por red ni por tipo")
        else:
            veredicto = "NO ENTRA"
            if not v4:
                razones.append(
                    f"granularidad '{gran}': no es serie "
                    "temporal alineable a diaria"
                )
            if not min_ok:
                razones.append(
                    f"min_fecha '{reg.get('min_fecha')}' "
                    f"> {VENTANA_INICIO}"
                )
            if not max_ok:
                razones.append(
                    f"max_fecha '{reg.get('max_fecha')}' "
                    f"< {MAX_FECHA_RECIENTE}"
                )
        motivo_final = " | ".join(
            filter(None, [str(reg.get("motivo", ""))]
                   + razones)
        )
        filas.append({
            "fuente": reg["fuente"],
            "red": reg.get("red", "N/D"),
            "tipo_activo": reg.get("tipo_activo", "N/D"),
            "V1_tiene_api": v1,
            "V2_desglose_red": v2,
            "V3_desglose_tipo": v3,
            "V4_serie_temporal": v4,
            "V5_cobertura": v5,
            "veredicto": veredicto,
            "endpoint": reg["endpoint"],
            "granularidad": gran,
            "min_fecha": reg.get("min_fecha", "s/d"),
            "max_fecha": reg.get("max_fecha", "s/d"),
            "n_obs": reg.get("n_obs", 0),
            "motivo": motivo_final,
            "fecha_verificacion": reg["fecha_verificacion"],
        })
    return pd.DataFrame(filas)


def registro_de_excepcion(
    fuente: str,
    endpoint: str,
    error: Exception,
) -> dict[str, Any]:
    """
    Envuelve una excepción inesperada como registro NO DISPONIBLE.

    Args:
        fuente: Nombre de la fuente que falló.
        endpoint: Endpoint asociado.
        error: Excepción capturada.

    Returns:
        Registro estándar con ok=False.
    """
    return registro_estandar(
        fuente=fuente, endpoint=endpoint, ok=False,
        red="N/D", tipo_activo="N/D",
        motivo=f"excepción inesperada: {error}",
    )


"""# 5. Proceso"""

resultados: list[dict[str, Any]] = []
protocolos_rwa: pd.DataFrame = pd.DataFrame()

# ── 5.a DeFiLlama: catálogo de protocolos RWA ─────────────────────

if PROBAR_DEFILLAMA:
    if DEBUG:
        print("\n[DEBUG] === 5.a Protocolos RWA en DeFiLlama ===")
    try:
        protocolos_rwa = listar_protocolos_rwa()
    except Exception as _e:
        protocolos_rwa = pd.DataFrame()
        resultados.append(registro_de_excepcion(
            "DeFiLlama/protocols", LLAMA_PROTOCOLS, _e
        ))
    if protocolos_rwa.empty:
        resultados.append(registro_estandar(
            fuente="DeFiLlama/protocols",
            endpoint=LLAMA_PROTOCOLS, ok=False,
            motivo=(
                "no se pudo listar protocolos de categoría "
                f"'{CATEGORIA_RWA}'"
            ),
        ))
    elif DEBUG:
        print(
            f"[DEBUG] protocolos RWA hallados: "
            f"{len(protocolos_rwa)}"
        )
        print(
            protocolos_rwa[
                ["name", "tags", "n_chains", "tvl"]
            ].head(15).to_string(index=False)
        )
        _redes = sorted({
            c for fila in protocolos_rwa["chains"]
            for c in str(fila).split(",") if c
        })
        print(f"[DEBUG] chains presentes en RWA: {len(_redes)}")
        print(f"[DEBUG] {', '.join(_redes)}")
        _tags = sorted({
            t for fila in protocolos_rwa["tags"]
            for t in str(fila).split(",") if t
        })
        print(f"[DEBUG] tags (tipos de activo): {', '.join(_tags)}")
        _interes = [
            r for r in REDES_INTERES if r in _redes
        ]
        print(f"[DEBUG] redes de interés cubiertas: {_interes}")

# ── 5.b Series históricas por chain de los mayores RWA ────────────

series_muestra: list[tuple[str, dict[str, dict[str, float]]]] = []
series_con_tipo: list[tuple[str, dict[str, dict[str, float]]]] = []

if PROBAR_DEFILLAMA and not protocolos_rwa.empty:
    if DEBUG:
        print(
            "\n[DEBUG] === 5.b Series históricas por chain "
            f"(muestra de {N_PROTOCOLOS_MUESTRA}) ==="
        )
    muestra = protocolos_rwa.head(N_PROTOCOLOS_MUESTRA)
    for _, _fila in muestra.iterrows():
        _slug = str(_fila["slug"])
        _nombre = str(_fila["name"])
        _tags = str(_fila["tags"])
        try:
            resultados.extend(
                verificar_protocolo_rwa(_slug, _nombre, _tags)
            )
            _detalle = serie_tvl_protocolo(_slug)
            if _detalle["ok"]:
                series_muestra.append((_nombre, _detalle["series"]))
                series_con_tipo.append((_tags, _detalle["series"]))
        except Exception as _e:
            resultados.append(registro_de_excepcion(
                f"DeFiLlama/protocolo/{_nombre}",
                f"{LLAMA_PROTOCOL}/{_slug}", _e,
            ))
        time.sleep(PAUSA)

# ── 5.c Agregados RWA por red y por tipo de activo ────────────────

if series_muestra:
    if DEBUG:
        print("\n[DEBUG] === 5.c Agregados RWA por red y tipo ===")
    try:
        _agg_red = agregar_series_por_chain(series_muestra)
        resultados.extend(registros_desde_agregado(
            fuente="DeFiLlama/RWA_agregado_por_red",
            endpoint=f"{LLAMA_PROTOCOL}/{{slug}} (chainTvls)",
            agregado=_agg_red,
            dimension="red",
            motivo=(
                "suma de TVL de los "
                f"{len(series_muestra)} mayores protocolos RWA "
                "por chain"
            ),
        ))
    except Exception as _e:
        resultados.append(registro_de_excepcion(
            "DeFiLlama/RWA_agregado_por_red", LLAMA_PROTOCOL, _e
        ))
    try:
        _agg_tipo = agregar_series_por_tipo(series_con_tipo)
        resultados.extend(registros_desde_agregado(
            fuente="DeFiLlama/RWA_agregado_por_tipo",
            endpoint=f"{LLAMA_PROTOCOL}/{{slug}} (tags+chainTvls)",
            agregado=_agg_tipo,
            dimension="tipo",
            motivo=(
                "suma de TVL por tag declarado (tipo de activo), "
                "agregando todas las chains"
            ),
        ))
    except Exception as _e:
        resultados.append(registro_de_excepcion(
            "DeFiLlama/RWA_agregado_por_tipo", LLAMA_PROTOCOL, _e
        ))

# ── 5.d Stablecoin RWA ya validada (USDY) por chain ───────────────

if PROBAR_DEFILLAMA:
    if DEBUG:
        print("\n[DEBUG] === 5.d USDY por chain (fuente base) ===")
    for _chain in CHAINS_STABLECOIN:
        try:
            resultados.append(verificar_stablecoin_rwa_por_chain(
                USDY_STABLECOIN_ID, _chain, "USDY"
            ))
        except Exception as _e:
            resultados.append(registro_de_excepcion(
                "DeFiLlama/stablecoin/USDY",
                f"{LLAMA_STABLE}/stablecoincharts/{_chain}", _e,
            ))
        time.sleep(PAUSA)

# ── 5.e TVL total por chain (contexto, no RWA) ────────────────────

if PROBAR_DEFILLAMA:
    if DEBUG:
        print("\n[DEBUG] === 5.e TVL total por chain (contexto) ===")
    for _chain in ["Stellar", "Ethereum"]:
        try:
            resultados.append(verificar_llama_chain_tvl(_chain))
        except Exception as _e:
            resultados.append(registro_de_excepcion(
                f"DeFiLlama/chainTVL/{_chain}",
                f"{LLAMA_CHAIN_TVL}/{_chain}", _e,
            ))
        time.sleep(PAUSA)

# ── 5.f RWA.xyz ───────────────────────────────────────────────────

if PROBAR_RWA_XYZ:
    if DEBUG:
        print("\n[DEBUG] === 5.f RWA.xyz ===")
    try:
        resultados.append(verificar_rwa_xyz())
    except Exception as _e:
        resultados.append(registro_de_excepcion(
            "RWA.xyz", RWA_XYZ_CANDIDATOS[0], _e
        ))

# ── 5.g CoinGecko: categorías y tokens RWA ────────────────────────

if PROBAR_COINGECKO:
    if DEBUG:
        print("\n[DEBUG] === 5.g CoinGecko ===")
    try:
        resultados.extend(verificar_coingecko_categorias())
    except Exception as _e:
        resultados.append(registro_de_excepcion(
            "CoinGecko/categorias",
            f"{COINGECKO_BASE}/coins/categories", _e,
        ))
    time.sleep(PAUSA)
    for _simbolo, (_pista, _tipo) in CG_TOKENS_RWA.items():
        try:
            resultados.append(verificar_coingecko_token_rwa(
                _simbolo, _pista, _tipo
            ))
        except Exception as _e:
            resultados.append(registro_de_excepcion(
                f"CoinGecko/token/{_simbolo}",
                f"{COINGECKO_BASE}/coins", _e,
            ))
        time.sleep(PAUSA)

# ── 5.h CoinMetrics: supply de tokens RWA ─────────────────────────

if PROBAR_COINMETRICS:
    if DEBUG:
        print("\n[DEBUG] === 5.h CoinMetrics ===")
    for _asset, _tipo in CM_ASSETS_RWA.items():
        try:
            resultados.append(
                verificar_coinmetrics_rwa(_asset, _tipo)
            )
        except Exception as _e:
            resultados.append(registro_de_excepcion(
                f"CoinMetrics/{_asset.upper()}", CM_BASE, _e
            ))
        time.sleep(PAUSA)

# ── 5.i Matriz de viabilidad ──────────────────────────────────────

if DEBUG:
    print("\n[DEBUG] === 5.i Matriz de viabilidad RWA ===")

matriz: pd.DataFrame = construir_matriz_viabilidad(resultados)

if DEBUG and not matriz.empty:
    pd.set_option("display.max_colwidth", 45)
    pd.set_option("display.width", 160)
    _cols = [
        "fuente", "red", "tipo_activo",
        "V1_tiene_api", "V2_desglose_red", "V3_desglose_tipo",
        "V4_serie_temporal", "V5_cobertura",
        "veredicto", "granularidad",
        "min_fecha", "max_fecha", "n_obs",
    ]
    print("\n[DEBUG] Matriz de viabilidad:")
    print(matriz[_cols].to_string(index=False))

# ── 5.j Resumen: qué series RWA son construibles ──────────────────

if not matriz.empty:
    entran: pd.DataFrame = matriz[
        matriz["veredicto"].isin(["ENTRA", "ENTRA_PARCIAL"])
    ]
    print("\n[RWA] === Series RWA construibles ===")
    print(f"[RWA] Fecha de verificación: {FECHA_VERIFICACION}")
    print(
        f"[RWA] Ventana exigida: {VENTANA_INICIO} → "
        f"{MAX_FECHA_RECIENTE} (mín.) / {HOY}"
    )
    for _veredicto, _grupo in matriz.groupby("veredicto"):
        print(f"\n[RWA] {_veredicto} ({len(_grupo)} filas):")
        for _, _fila in _grupo.iterrows():
            print(
                f"  - {_fila['fuente']} | red={_fila['red']} "
                f"| tipo={_fila['tipo_activo']} "
                f"| {_fila['granularidad']} "
                f"| {_fila['min_fecha']}→{_fila['max_fecha']} "
                f"| n={_fila['n_obs']}"
            )
            print(f"      endpoint: {_fila['endpoint']}")
            if _fila["motivo"]:
                print(f"      motivo:   {_fila['motivo']}")

    print("\n[RWA] === Alcance sugerido para la adquisición ===")
    _por_red = entran[entran["V2_desglose_red"]]
    _redes_ok = sorted(set(_por_red["red"]) - {"TODAS", "N/D"})
    print(
        f"[RWA] Redes con serie RWA construible "
        f"({len(_redes_ok)}): {', '.join(_redes_ok) or 'ninguna'}"
    )
    _por_tipo = entran[entran["V3_desglose_tipo"]]
    _tipos_ok = sorted(set(_por_tipo["tipo_activo"]) - {"TODOS",
                                                        "N/D"})
    print(
        f"[RWA] Tipos de activo con serie construible "
        f"({len(_tipos_ok)}): {', '.join(_tipos_ok) or 'ninguna'}"
    )
    _redes_interes_ok = [r for r in REDES_INTERES
                         if r in _redes_ok]
    print(
        f"[RWA] Redes de interés del proyecto cubiertas: "
        f"{', '.join(_redes_interes_ok) or 'ninguna'}"
    )
    print(f"[RWA] Total de filas verificadas: {len(matriz)}")
else:
    print("\n[RWA] Sin registros: ninguna fuente pudo verificarse.")

"""# 6. Save"""

# CAMBIO E2E-00-01: garantizar el esquema de ambos CSV de salida.
# Si las consultas no devuelven registros, escribir archivos con cabeceras
# y cero filas. Así no se conserva inadvertidamente el catálogo de una
# ejecución anterior ni se entrega un CSV sin columnas.
if SAVE_DATA:
    os.makedirs(DIR_REPORTES, exist_ok=True)

    if matriz.empty:
        columnas_matriz = [
            "fuente", "red", "tipo_activo", "V1_tiene_api",
            "V2_desglose_red", "V3_desglose_tipo", "V4_serie_temporal",
            "V5_cobertura", "veredicto", "endpoint", "granularidad",
            "min_fecha", "max_fecha", "n_obs", "motivo",
            "fecha_verificacion",
        ]
        pd.DataFrame(columns=columnas_matriz).to_csv(
            ARCHIVO_SALIDA, index=False
        )
        if DEBUG:
            print("[DEBUG][ALERTA] Matriz vacía: CSV creado con cabeceras.")
    else:
        matriz.to_csv(ARCHIVO_SALIDA, index=False)
    if DEBUG:
        print(f"\n[DEBUG] guardado: {ARCHIVO_SALIDA}")

    if protocolos_rwa.empty:
        columnas_protocolos = [
            "slug", "name", "category", "tags", "chains",
            "n_chains", "tvl",
        ]
        pd.DataFrame(columns=columnas_protocolos).to_csv(
            ARCHIVO_PROTOCOLOS, index=False
        )
        if DEBUG:
            print("[DEBUG][ALERTA] Catálogo RWA vacío: CSV con cabeceras.")
    else:
        protocolos_rwa.to_csv(ARCHIVO_PROTOCOLOS, index=False)
    if DEBUG:
        print(f"[DEBUG] guardado: {ARCHIVO_PROTOCOLOS}")
elif DEBUG:
    print("\n[DEBUG] SAVE_DATA=False — reporte solo en memoria")

