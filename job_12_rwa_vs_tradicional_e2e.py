"""# Job 12 - RWA vs. Activo Tradicional: ¿Seguidor, Simultáneo o Adelantado?"""

# ════════════════════════════════════════════════════════════
# 1. Configuración
# ════════════════════════════════════════════════════════════

DEBUG: bool = True
SAVE_DATA: bool = True

# ════════════════════════════════════════════════════════════
# 2. Imports
# ════════════════════════════════════════════════════════════

import contextlib
import io
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import grangercausalitytests

from google.colab import drive
drive.mount("/content/drive")

# ════════════════════════════════════════════════════════════
# 3. Paths y Constantes
# ════════════════════════════════════════════════════════════

# CAMBIO E2E-12-01: misma ruta predeterminada, configurable por entorno.
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
ARCHIVO_XCORR: str = os.path.join(
    DIR_REPORTES, "rwa_vs_tradicional.csv"
)
ARCHIVO_GRANGER: str = os.path.join(
    DIR_REPORTES, "granger_rwa_tradicional.csv"
)

# Pares a comparar: (tradicional, tokenizado, etiqueta)
PARES: list[tuple[str, str, str]] = [
    (
        "macro_tsy_1m",
        "rwa_tipo_treasurybills",
        "DGS1MO vs treasurybills",
    ),
    (
        "macro_tsy_3m",
        "rwa_tipo_treasurybills",
        "DGS3MO vs treasurybills",
    ),
    (
        "macro_tsy_1m",
        "rwa_usdy_supply",
        "DGS1MO vs USDY",
    ),
]

MAX_LAG: int = 14
ALFA_BH: float = 0.05

# CAMBIO E2E-12-01: esquemas explicitos para salidas sin pruebas evaluables.
COLUMNAS_XCORR: list[str] = [
    "par", "lag", "n", "correlacion", "pvalue", "interpretacion",
]
COLUMNAS_GRANGER: list[str] = [
    "par", "serie_x", "serie_y", "direccion", "n", "lag",
    "pval_f", "nota", "bh_q", "bh_rechaza", "ambito_bh",
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
# 4. Funciones
# ════════════════════════════════════════════════════════════


def cargar_series(ruta: str) -> pd.DataFrame:
    df = pd.read_csv(ruta, parse_dates=["fecha"])
    df = df.set_index("fecha").sort_index()
    if DEBUG:
        print(f"[DEBUG] {os.path.basename(ruta)}: {df.shape}")
    return df


# CAMBIO E2E-12-01: validacion estructural sin imputar ni modificar datos.
def validar_entrada_job12(df: pd.DataFrame, ruta: str) -> None:
    """Comprueba el contrato de series_completas.csv (productor: Job 4)."""
    if df.empty or len(df.columns) == 0:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 12: {ruta} sin observaciones o series "
            "(productor: Job 4)."
        )
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 12: fechas invalidas en {ruta} "
            "(productor: Job 4)."
        )
    if df.index.has_duplicates:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 12: fechas duplicadas en {ruta} "
            "(productor: Job 4)."
        )
    if not df.columns.is_unique:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 12: columnas duplicadas en {ruta} "
            "(productor: Job 4)."
        )
    # Las columnas ausentes siguen omitiendose por par, tal como en el original.
    # Solo verificamos las que existen y van a ser usadas en log-diff.
    requeridas = {nombre for trad, rwa, _ in PARES for nombre in (trad, rwa)}
    for columna in sorted(requeridas.intersection(df.columns)):
        if not pd.api.types.is_numeric_dtype(df[columna]):
            raise RuntimeError(
                f"[DEBUG][ERROR] Job 12: columna '{columna}' no numerica "
                f"en {ruta} (productor: Job 4)."
            )


def _log_diff(serie: pd.Series) -> pd.Series:
    return np.log(serie.where(serie > 0)).diff()


def correlacion_cruzada(
    s_trad: pd.Series,
    s_rwa: pd.Series,
    max_lag: int,
    etiqueta: str,
) -> pd.DataFrame:
    """
    Correlación cruzada entre tradicional y tokenizado.

    Lag negativo: el tradicional adelanta al tokenizado.
    Lag cero: simultáneos.
    Lag positivo: el tokenizado adelanta al tradicional.

    La convención es: corr(trad[t], rwa[t + lag]).
    """
    filas: list[dict] = []

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x = s_trad.iloc[:lag]
            y = s_rwa.iloc[-lag:]
        elif lag > 0:
            x = s_trad.iloc[lag:]
            y = s_rwa.iloc[:-lag]
        else:
            x = s_trad
            y = s_rwa

        idx_comun = x.index.intersection(y.index)
        if len(idx_comun) < 10:
            filas.append({
                "par": etiqueta,
                "lag": lag,
                "n": len(idx_comun),
                "correlacion": None,
                "pvalue": None,
                "interpretacion": "n insuficiente",
            })
            continue

        xv = x.reindex(idx_comun).dropna()
        yv = y.reindex(idx_comun).dropna()
        idx_ok = xv.index.intersection(yv.index)
        if len(idx_ok) < 10:
            filas.append({
                "par": etiqueta,
                "lag": lag,
                "n": len(idx_ok),
                "correlacion": None,
                "pvalue": None,
                "interpretacion": "n insuficiente",
            })
            continue

        r, p = stats.pearsonr(
            xv.reindex(idx_ok).values,
            yv.reindex(idx_ok).values,
        )

        if lag < 0:
            interp = "tradicional adelanta"
        elif lag > 0:
            interp = "tokenizado adelanta"
        else:
            interp = "simultáneo"

        filas.append({
            "par": etiqueta,
            "lag": lag,
            "n": len(idx_ok),
            "correlacion": round(float(r), 6),
            "pvalue": round(float(p), 6),
            "interpretacion": interp,
        })

    return pd.DataFrame(filas)


def _granger_raw(data: pd.DataFrame, max_lag: int) -> dict:
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


def granger_bidir(
    s_trad: pd.Series,
    s_rwa: pd.Series,
    max_lag: int,
    nombre_trad: str,
    nombre_rwa: str,
    etiqueta: str,
) -> list[dict]:
    """
    Granger bidireccional:
      D1: tradicional → tokenizado (trad rezagado predice rwa).
      D2: tokenizado → tradicional (rwa rezagado predice trad).
    """
    filas: list[dict] = []

    direcciones = [
        ("trad→token", s_rwa, s_trad, nombre_trad, nombre_rwa),
        ("token→trad", s_trad, s_rwa, nombre_rwa, nombre_trad),
    ]

    for dir_label, y, x, x_name, y_name in direcciones:
        data = pd.concat(
            [y.rename("y"), x.rename("x")], axis=1
        ).dropna()
        n_min = max_lag + 10
        fila_base = {
            "par": etiqueta,
            "serie_x": x_name,
            "serie_y": y_name,
            "direccion": dir_label,
            "n": len(data),
        }

        if len(data) < n_min:
            filas.extend([
                {
                    **fila_base, "lag": k,
                    "pval_f": None,
                    "nota": f"n={len(data)}<{n_min}",
                }
                for k in range(1, max_lag + 1)
            ])
            continue

        try:
            resultado = _granger_raw(data[["y", "x"]], max_lag)
        except Exception as exc:
            filas.extend([
                {
                    **fila_base, "lag": k,
                    "pval_f": None,
                    "nota": f"error: {str(exc)[:60]}",
                }
                for k in range(1, max_lag + 1)
            ])
            continue

        for k in range(1, max_lag + 1):
            try:
                pval_f = float(resultado[k][0]["ssr_ftest"][1])
            except Exception:
                filas.append({
                    **fila_base, "lag": k,
                    "pval_f": None,
                    "nota": "sin estadístico",
                })
                continue
            filas.append({
                **fila_base, "lag": k,
                "pval_f": round(pval_f, 6),
                "nota": "",
            })

    return filas


def aplicar_bh_global(
    df: pd.DataFrame,
    alfa: float = ALFA_BH,
) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    df["bh_q"] = np.nan
    df["bh_rechaza"] = False
    df["ambito_bh"] = (
        "global: 2 dir × todos los pares × lags"
    )
    pvals = pd.to_numeric(df["pval_f"], errors="coerce")
    validos = pvals.notna()
    if validos.sum() < 2:
        return df
    rechaza, adj, _, _ = multipletests(
        pvals[validos].values, alpha=alfa, method="fdr_bh"
    )
    df.loc[validos, "bh_q"] = np.round(adj, 6)
    df.loc[validos, "bh_rechaza"] = rechaza
    return df


def diagnosticar_escenario(
    df_xcorr_par: pd.DataFrame,
    df_granger_par: pd.DataFrame,
    etiqueta: str,
) -> dict:
    """
    Determina el escenario (A, B o C) para un par.

    A) Seguidor lento: tradicional Granger-causa tokenizado,
       correlación máxima en lag negativo.
    B) Simultáneo: máxima correlación en lag 0, sin Granger
       significativo en ninguna dirección.
    C) Adelantado: tokenizado Granger-causa tradicional,
       correlación máxima en lag positivo.
    """
    resultado: dict[str, Any] = {"par": etiqueta}

    # Correlación cruzada
    validos = df_xcorr_par[
        df_xcorr_par["correlacion"].notna()
    ]
    if validos.empty:
        resultado["escenario"] = "indeterminado"
        resultado["razon"] = "sin correlaciones válidas"
        return resultado

    idx_max = validos["correlacion"].abs().idxmax()
    lag_max = int(validos.loc[idx_max, "lag"])
    corr_max = float(validos.loc[idx_max, "correlacion"])
    resultado["lag_max_corr"] = lag_max
    resultado["corr_max"] = round(corr_max, 6)

    # Granger
    sig_trad_token = pd.DataFrame()
    sig_token_trad = pd.DataFrame()
    if not df_granger_par.empty:
        sig = df_granger_par[
            df_granger_par["bh_rechaza"] == True
        ]
        sig_trad_token = sig[
            sig["direccion"] == "trad→token"
        ]
        sig_token_trad = sig[
            sig["direccion"] == "token→trad"
        ]

    resultado["granger_trad_token"] = len(sig_trad_token) > 0
    resultado["granger_token_trad"] = len(sig_token_trad) > 0

    if not sig_trad_token.empty:
        best = sig_trad_token.loc[
            sig_trad_token["bh_q"].idxmin()
        ]
        resultado["granger_trad_token_lag"] = int(best["lag"])
        resultado["granger_trad_token_q"] = float(best["bh_q"])

    if not sig_token_trad.empty:
        best = sig_token_trad.loc[
            sig_token_trad["bh_q"].idxmin()
        ]
        resultado["granger_token_trad_lag"] = int(best["lag"])
        resultado["granger_token_trad_q"] = float(best["bh_q"])

    # Diagnóstico
    g_tt = resultado["granger_trad_token"]
    g_tk = resultado["granger_token_trad"]

    if g_tt and not g_tk:
        resultado["escenario"] = "A"
        resultado["descripcion"] = (
            "SEGUIDOR LENTO: el tradicional Granger-causa al "
            "tokenizado. El RWA sigue al Treasury con rezago."
        )
    elif g_tk and not g_tt:
        resultado["escenario"] = "C"
        resultado["descripcion"] = (
            "ADELANTADO: el tokenizado Granger-causa al "
            "tradicional. El RWA se mueve antes que el "
            "Treasury."
        )
    elif g_tt and g_tk:
        resultado["escenario"] = "bidireccional"
        resultado["descripcion"] = (
            "BIDIRECCIONAL: ambas direcciones son "
            "significativas. Retroalimentación mutua o "
            "factor común."
        )
    elif lag_max == 0:
        resultado["escenario"] = "B"
        resultado["descripcion"] = (
            "SIMULTÁNEO: sin Granger significativo, máxima "
            "correlación en lag 0. Mismo activo, distinto "
            "envase."
        )
    elif lag_max < 0:
        resultado["escenario"] = "A (débil)"
        resultado["descripcion"] = (
            "SEGUIDOR LENTO (débil): Granger no significativo "
            "pero máxima correlación con el tradicional "
            "adelantando. Sin validación formal."
        )
    elif lag_max > 0:
        resultado["escenario"] = "C (débil)"
        resultado["descripcion"] = (
            "ADELANTADO (débil): Granger no significativo "
            "pero máxima correlación con el tokenizado "
            "adelantando. Sin validación formal."
        )
    else:
        resultado["escenario"] = "B"
        resultado["descripcion"] = (
            "SIMULTÁNEO: sin evidencia clara de precedencia."
        )

    # Salvedades
    resultado["salvedades"] = (
        "Mercado RWA chico y de adopción incipiente "
        "(2025-2026). Treasury yield es un mercado profundo "
        "y líquido; RWA TVL refleja entradas/salidas netas "
        "de un mercado naciente."  # CAMBIO B3-3: muestra real en el veredicto
    )

    return resultado


# ════════════════════════════════════════════════════════════
# 5. Proceso principal
# ════════════════════════════════════════════════════════════


def main() -> None:
    # CAMBIO B1-4 (Bloque 1): hash de las entradas vs. manifest.
    verificar_entradas_snapshot([ARCHIVO_COMPLETAS], "Job 12")

    print(
        "\n[DEBUG] ═══════════════════════════════════════════"
        "═══════════════"
    )
    print(
        "[DEBUG] Job 12 — RWA vs. Activo Tradicional"
    )
    print(
        "[DEBUG] ═══════════════════════════════════════════"
        "═══════════════"
    )

    # CAMBIO E2E-12-01: verificar entrada antes del analisis.
    if not os.path.isfile(ARCHIVO_COMPLETAS):
        raise FileNotFoundError(
            f"[DEBUG][ERROR] Job 12: falta {ARCHIVO_COMPLETAS}. "
            "Debe producirlo el Job 4 (series_completas.csv)."
        )
    try:
        df = cargar_series(ARCHIVO_COMPLETAS)
    except (OSError, ValueError, KeyError, UnicodeError,
            pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 12: no se puede leer "
            f"{ARCHIVO_COMPLETAS}: {exc}"
        ) from exc
    validar_entrada_job12(df, ARCHIVO_COMPLETAS)

    all_xcorr: list[pd.DataFrame] = []
    all_granger: list[dict] = []
    diagnosticos: list[dict] = []

    for col_trad, col_rwa, etiqueta in PARES:
        print(f"\n[DEBUG] ── {etiqueta} ──")

        if col_trad not in df.columns:
            if DEBUG:
                print(f"[DEBUG] {col_trad} ausente, skip")
            continue
        if col_rwa not in df.columns:
            if DEBUG:
                print(f"[DEBUG] {col_rwa} ausente, skip")
            continue

        s_trad = _log_diff(df[col_trad]).dropna()
        s_rwa = _log_diff(df[col_rwa]).dropna()

        if DEBUG:
            print(
                f"[DEBUG] {col_trad}: {len(s_trad)} obs | "
                f"{col_rwa}: {len(s_rwa)} obs"
            )

        # 1. Correlación cruzada
        df_xcorr = correlacion_cruzada(
            s_trad, s_rwa, MAX_LAG, etiqueta,
        )
        all_xcorr.append(df_xcorr)

        validos = df_xcorr[df_xcorr["correlacion"].notna()]
        if not validos.empty:
            idx_max = validos["correlacion"].abs().idxmax()
            print(
                f"[DEBUG] Máx |corr|: lag="
                f"{validos.loc[idx_max, 'lag']}, "
                f"r={validos.loc[idx_max, 'correlacion']:.4f}, "
                f"p={validos.loc[idx_max, 'pvalue']:.4f} "
                f"({validos.loc[idx_max, 'interpretacion']})"
            )
            corr_0 = validos[validos["lag"] == 0]
            if not corr_0.empty:
                print(
                    f"[DEBUG] Corr lag=0: "
                    f"r={corr_0.iloc[0]['correlacion']:.4f}, "
                    f"p={corr_0.iloc[0]['pvalue']:.4f}"
                )

        # 2. Granger bidireccional
        filas_g = granger_bidir(
            s_trad, s_rwa, MAX_LAG,
            col_trad, col_rwa, etiqueta,
        )
        all_granger.extend(filas_g)

    # CAMBIO E2E-12-01: sin pares, tablas vacias con esquema explicito.
    # Con pares, se ejecutan exactamente las mismas operaciones originales.
    if not all_xcorr:
        print("[DEBUG][ALERTA] Job 12: ningun par tiene ambas columnas; "
              "el analisis no es evaluable.")
    df_granger = (
        pd.DataFrame(all_granger)
        if all_granger else pd.DataFrame(columns=COLUMNAS_GRANGER)
    )
    if not df_granger.empty:
        df_granger = aplicar_bh_global(df_granger)

    df_xcorr_all = (
        pd.concat(all_xcorr, ignore_index=True)
        if all_xcorr else pd.DataFrame(columns=COLUMNAS_XCORR)
    )

    # Resumen Granger
    _p = pd.to_numeric(
        df_granger["pval_f"], errors="coerce"
    ) if not df_granger.empty else pd.Series(dtype=float)
    n_tests = int(_p.notna().sum())
    n_crudo = int((_p < ALFA_BH).sum())
    n_bh = int(
        df_granger["bh_rechaza"].fillna(False).astype(bool).sum()
    ) if not df_granger.empty else 0

    print(
        f"\n[DEBUG] Granger bidireccional: {n_tests} tests | "
        f"p<{ALFA_BH} crudo: {n_crudo} | FDR GLOBAL: {n_bh}"
    )

    if n_bh > 0:
        _sig = df_granger[df_granger["bh_rechaza"]]
        _cols = [
            "par", "direccion", "lag",
            "pval_f", "bh_q", "n",
        ]
        _cols = [c for c in _cols if c in _sig.columns]
        print("[DEBUG] Pares significativos:")
        print(_sig[_cols].to_string(index=False))

    # Diagnóstico por par
    for _, _, etiqueta in PARES:
        # CAMBIO E2E-12-01: el DataFrame vacio conserva sus columnas.
        xcorr_par = df_xcorr_all[df_xcorr_all["par"] == etiqueta]
        granger_par = df_granger[df_granger["par"] == etiqueta]
        diag = diagnosticar_escenario(
            xcorr_par, granger_par, etiqueta,
        )
        diagnosticos.append(diag)

    # ── Reporte final ───────────────────────────────────────
    print("\n" + "═" * 64)
    print(
        "REPORTE FINAL — RWA vs. ACTIVO TRADICIONAL"
    )
    print("═" * 64)

    for diag in diagnosticos:
        print(f"\n{'─' * 64}")
        print(f"PAR: {diag['par']}")
        print(f"{'─' * 64}")
        print(
            f"  Escenario: {diag.get('escenario', '?')}"
        )
        print(
            f"  {diag.get('descripcion', '')}"
        )
        lag_mc = diag.get("lag_max_corr")
        corr_mc = diag.get("corr_max")
        if lag_mc is not None:
            print(
                f"  Correlación cruzada: máx |r| en lag="
                f"{lag_mc} (r={corr_mc})"
            )
        if diag.get("granger_trad_token"):
            print(
                f"  Granger trad→token: lag="
                f"{diag.get('granger_trad_token_lag')}, "
                f"q={diag.get('granger_trad_token_q')}"
            )
        if diag.get("granger_token_trad"):
            print(
                f"  Granger token→trad: lag="
                f"{diag.get('granger_token_trad_lag')}, "
                f"q={diag.get('granger_token_trad_q')}"
            )
        print(f"  {diag.get('salvedades', '')}")

    # Veredicto consolidado
    print(f"\n{'═' * 64}")
    print("VEREDICTO CONSOLIDADO")
    print("═" * 64)

    escenarios = {
        d["par"]: d.get("escenario", "?")
        for d in diagnosticos
    }
    for par, esc in escenarios.items():
        print(f"  {par}: Escenario {esc}")

    # CAMBIO B3-3 (Bloque 3): el veredicto sale del conteo de
    # escenarios. Antes "A" en "A (débil)" bastaba para declarar
    # "Escenario A dominante" aunque 2 de 3 pares fueran
    # bidireccionales, y la muestra "2021-2026" era texto fijo.
    _conteo = pd.Series(
        list(escenarios.values()), dtype=object
    ).value_counts()
    _validos = _conteo.drop("indeterminado", errors="ignore")
    _n_eval = int(_validos.sum())
    _cols_rwa = [c for _, c, _ in PARES if c in df.columns]
    _inicios = [
        df[c].first_valid_index() for c in _cols_rwa
        if df[c].first_valid_index() is not None
    ]
    _muestra = (
        f"{min(_inicios).date()} → {df.index.max().date()}"
        if _inicios else "sin datos RWA"
    )
    TEXTO_ESCENARIO: dict[str, str] = {
        "A": "el tradicional Granger-causa al tokenizado: el RWA "
             "sigue al tradicional con rezago.",
        "A (débil)": "máxima correlación con el tradicional "
                     "adelantando, sin Granger significativo.",
        "B": "movimiento simultáneo o sin precedencia detectable.",
        "C": "el tokenizado Granger-causa al tradicional.",
        "C (débil)": "máxima correlación con el tokenizado "
                     "adelantando, sin Granger significativo.",
        "bidireccional": "ambas direcciones de Granger son "
                         "significativas: retroalimentación o "
                         "factor común.",
    }

    # CAMBIO E2E-12-01: no declarar B si no se pudo evaluar ningun par.
    if not all_xcorr or _n_eval == 0:
        print("\n[DEBUG][ALERTA] Job 12: veredicto no evaluable "
              "(no se proceso ningun par).")
    else:
        print(
            f"\nEscenarios ({_n_eval} pares evaluables de "
            f"{len(escenarios)}): "
            + ", ".join(f"{k}={v}" for k, v in _validos.items())
            + f" | muestra RWA: {_muestra}"
        )
        _n_c = int(_validos.get("C", 0))
        if _n_c > 0:
            print(
                f"\nEVIDENCIA DE ESCENARIO C (validada por Granger) "
                f"en {_n_c} de {_n_eval} pares."
            )
            print(
                "SALVEDADES IMPORTANTES: mercado RWA chico y de "
                "adopción incipiente, Treasury es un mercado "
                f"profundo y líquido, muestra RWA {_muestra}, no "
                "se afirma causalidad — la precedencia temporal "
                "puede reflejar flujos de capital hacia/desde "
                "protocolos DeFi que responden a las mismas señales "
                "macro con distinto timing, no necesariamente que "
                "el token 'causa' movimientos en el Treasury."
            )
        _k_top = int(_validos.max())
        _tops = list(_validos[_validos == _k_top].index)
        if len(_tops) == 1 and _k_top > _n_eval / 2:
            print(
                f"\nEscenario {_tops[0]} dominante ({_k_top} de "
                f"{_n_eval} pares): "
                f"{TEXTO_ESCENARIO.get(_tops[0], '')}"
            )
        else:
            print(
                f"\nSin escenario dominante: ninguno reúne más de "
                f"la mitad de los {_n_eval} pares evaluables."
            )

    print("═" * 64)

    # ── Guardar ─────────────────────────────────────────────
    if SAVE_DATA:
        os.makedirs(DIR_REPORTES, exist_ok=True)
        guardados: list[str] = []

        # CAMBIO E2E-12-01: no dejar archivos antiguos cuando hay 0 pares.
        # Con registros, el formato CSV es identico al original.
        df_xcorr_all.to_csv(ARCHIVO_XCORR, index=False)
        guardados.append(ARCHIVO_XCORR)

        df_granger.to_csv(ARCHIVO_GRANGER, index=False)
        guardados.append(ARCHIVO_GRANGER)

        if DEBUG:
            print("\n[DEBUG] Archivos guardados:")
            for g in guardados:
                print(f"  {g}")
    elif DEBUG:
        print("\n[DEBUG] SAVE_DATA=False — nada guardado")


if __name__ == "__main__":
    main()

