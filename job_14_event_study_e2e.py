"""# Job 14 - Event Study: Señales de Tasa → TVL de Tokenized Treasuries"""

# ════════════════════════════════════════════════════════════
# 1. Configuración
# ════════════════════════════════════════════════════════════

DEBUG: bool = True
SAVE_DATA: bool = True

# ════════════════════════════════════════════════════════════
# 2. Imports
# ════════════════════════════════════════════════════════════

import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

# ════════════════════════════════════════════════════════════
# 3. Paths y Constantes
# ════════════════════════════════════════════════════════════

# CAMBIO E2E-14-01: ruta configurable, sin cambiar el valor predeterminado.
BASE_DIR: str = os.environ.get(
    "TESIS_BASE_DIR", "/content/drive/MyDrive/Tesis/expansion"
)
DIR_PROCESSED: str = os.path.join(BASE_DIR, "data", "processed")
DIR_REPORTES: str = os.path.join(
    BASE_DIR, "outputs", "reportes",
)
DIR_FIGURAS: str = os.path.join(
    DIR_REPORTES, "figuras_walkforward",
)

ARCHIVO_COMPLETAS: str = os.path.join(
    DIR_PROCESSED, "series_completas.csv",
)
ARCHIVO_CSV: str = os.path.join(
    DIR_REPORTES, "event_study.csv",
)
ARCHIVO_PNG: str = os.path.join(
    DIR_FIGURAS, "event_study_top3.png",
)
# CAMBIO B4-2 (Bloque 4): resumen de una fila (% condicional, %
# base) que lee la regla de decisión de Job 11.
ARCHIVO_RESUMEN: str = os.path.join(
    DIR_REPORTES, "event_study_resumen.csv",
)

COLS_SIGNAL: list[str] = [
    "macro_sofr",
    "macro_fed_funds",
    "macro_tsy_3m",
]
COL_TVL: str = "rwa_tipo_treasurybills"

VENTANA_DIAS: int = 49
MA_VENTANA: int = 30
UMBRAL_SIGMA: float = 1.0
COOLDOWN_EVENTOS: int = 14

# CAMBIO E2E-14-01: mismo esquema que el CSV original cuando hay eventos.
COLUMNAS_EVENT_STUDY: list[str] = [
    "variable_signal", "fecha_evento", "tasa_valor", "tvl_inicio",
    "tvl_fin", "delta_usd", "delta_pct",
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


# CAMBIO E2E-14-01: contrato de entrada del Job 4, sin cambiar datos.
def validar_entrada_job14(df: pd.DataFrame, ruta: str) -> None:
    """Comprueba fechas y columnas requeridas para el Event Study."""
    if df.empty or len(df.columns) == 0:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 14: {ruta} sin registros o series "
            "(productor: Job 4)."
        )
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 14: fechas invalidas en {ruta} "
            "(productor: Job 4)."
        )
    if df.index.has_duplicates:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 14: fechas duplicadas en {ruta} "
            "(productor: Job 4)."
        )
    if not df.columns.is_unique:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 14: columnas duplicadas en {ruta} "
            "(productor: Job 4)."
        )
    for columna in COLS_SIGNAL + [COL_TVL]:
        if columna not in df.columns:
            raise RuntimeError(
                f"[DEBUG][ERROR] Job 14: columna '{columna}' ausente "
                f"en {ruta} (productor: Job 4)."
            )
        if not pd.api.types.is_numeric_dtype(df[columna]):
            raise RuntimeError(
                f"[DEBUG][ERROR] Job 14: columna '{columna}' no "
                f"numerica en {ruta} (productor: Job 4)."
            )


# CAMBIO E2E-14-01: no reutilizar salidas de una corrida anterior.
def guardar_resumen_job14(
    pct_condicional: float | None,
    pct_base: float | None,
    n_eventos: int,
    n_ventanas_base: int,
    motivo: str = "",
    n_positivos: int = 0,
) -> None:
    """
    Guarda event_study_resumen.csv: % condicional (eventos con
    ΔTVL > 0), % base (todas las ventanas) y el SHA-256 de
    series_completas.csv, para que Job 11 verifique que el resumen
    corresponde a los mismos datos.

    CAMBIO B4-2 (Bloque 4).

    Args:
        pct_condicional: % de eventos con ΔTVL > 0 (o None).
        pct_base: % de ventanas con TVL creciente (o None).
        n_eventos: Cantidad de eventos.
        n_ventanas_base: Ventanas usadas en la tasa base.
        motivo: Por qué no es evaluable, si corresponde.
        n_positivos: Eventos con ΔTVL > 0 (para el test
            binomial de Job 11).
    """
    if not SAVE_DATA:
        return
    os.makedirs(DIR_REPORTES, exist_ok=True)
    pd.DataFrame([{
        "pct_condicional": pct_condicional,
        "pct_base": pct_base,
        "diferencia_pp": (
            round(pct_condicional - pct_base, 2)
            if pct_condicional is not None and pct_base is not None
            else None
        ),
        "n_eventos": n_eventos,
        "n_positivos": n_positivos,
        "n_ventanas_base": n_ventanas_base,
        "ventana_dias": VENTANA_DIAS,
        "sha256_series_completas": sha256_archivo(ARCHIVO_COMPLETAS),
        "motivo_no_evaluable": motivo,
    }]).to_csv(ARCHIVO_RESUMEN, index=False)
    print(f"[DEBUG] resumen para la regla de decisión: {ARCHIVO_RESUMEN}")


def guardar_sin_eventos_job14(motivo: str) -> None:
    """Publica artefactos vacios explicitos; no fabrica eventos."""
    print(f"[DEBUG][ALERTA] Job 14: NO EVALUABLE: {motivo}")
    if not SAVE_DATA:
        if DEBUG:
            print("[DEBUG] SAVE_DATA=False - no se guardan archivos")
        return

    os.makedirs(DIR_REPORTES, exist_ok=True)
    os.makedirs(DIR_FIGURAS, exist_ok=True)
    pd.DataFrame(columns=COLUMNAS_EVENT_STUDY).to_csv(
        ARCHIVO_CSV, index=False
    )
    guardar_resumen_job14(None, None, 0, 0, motivo)  # CAMBIO B4-2
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    ax.text(
        0.5, 0.58, "EVENT STUDY - NO EVALUABLE",
        ha="center", va="center", fontsize=16, weight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.44, motivo,
        ha="center", va="center", fontsize=11,
        transform=ax.transAxes,
    )
    fig.savefig(ARCHIVO_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if DEBUG:
        print(f"[DEBUG] Job 14: CSV vacio guardado: {ARCHIVO_CSV}")
        print(f"[DEBUG] Job 14: grafico NO EVALUABLE: {ARCHIVO_PNG}")


def detectar_eventos(
    serie: pd.Series,
    ma_ventana: int = MA_VENTANA,
    umbral_sigma: float = UMBRAL_SIGMA,
    cooldown: int = COOLDOWN_EVENTOS,
) -> list[int]:
    """
    Detecta eventos de SUBA de señal: días donde la serie sube
    más de `umbral_sigma` desvíos sobre su media móvil de
    `ma_ventana` días (`serie - ma > umbral_sigma * std`).

    CAMBIO I6-c: esto es explícito y a propósito — un evento
    exige una SUBA de la tasa (diff positivo), no una baja ni un
    movimiento en cualquier dirección. Si se quisiera estudiar
    bajas de tasa, haría falta un detector simétrico aparte (no
    lo agrega esta función).

    `cooldown` evita contar el mismo shock de corto plazo dos
    veces (un pico que dura varios días no genera N eventos
    pegados), pero NO garantiza que las ventanas de evaluación
    de 49 días de dos eventos consecutivos no se solapen — para
    eso ver eliminar_eventos_solapados() (CAMBIO I6-b).
    """
    ma = serie.rolling(ma_ventana).mean()
    std = serie.rolling(ma_ventana).std()

    diff = serie - ma
    umbral = umbral_sigma * std

    candidatos = diff > umbral
    candidatos = candidatos.fillna(False)

    eventos: list[int] = []
    ultimo: int = -cooldown - 1

    for i in range(ma_ventana, len(serie)):
        if candidatos.iloc[i] and (i - ultimo) > cooldown:
            eventos.append(i)
            ultimo = i

    return eventos


def eliminar_eventos_solapados(
    eventos: list[int],
    ventana: int = VENTANA_DIAS,
) -> tuple[list[int], int]:
    """
    Descarta eventos cuya ventana de evaluación de `ventana`
    días se solaparía con la del evento anterior YA CONSERVADO
    de la misma variable (voraz: conserva el primero
    cronológicamente de cada grupo solapado).

    CAMBIO I6-b: antes solo se aplicaba COOLDOWN_EVENTOS (14
    días) al detectar — menor que VENTANA_DIAS (49), así que dos
    eventos podían quedar separados por 15-48 días, con ventanas
    de evaluación que se pisan entre sí (no son observaciones
    independientes del mismo fenómeno).

    Args:
        eventos: Posiciones (índices enteros) de eventos de UNA
            variable, ya ordenadas ascendentemente.
        ventana: Días mínimos entre dos eventos conservados.

    Returns:
        (eventos_sin_solape, n_descartados).
    """
    if not eventos:
        return [], 0
    conservados = [eventos[0]]
    for pos in eventos[1:]:
        if pos - conservados[-1] >= ventana:
            conservados.append(pos)
    n_descartados = len(eventos) - len(conservados)
    return conservados, n_descartados


def tasa_base_tvl_creciente(
    df: pd.DataFrame,
    col_tvl: str = COL_TVL,
    ventana: int = VENTANA_DIAS,
) -> dict[str, Any]:
    """
    Tasa base (CAMBIO I6-a): porcentaje de TODAS las ventanas de
    `ventana` días del dataset (sin condicionar a ninguna señal)
    en las que el TVL es mayor al final que al principio.

    Sirve de contraste con la tasa condicional (tras un evento
    de señal, en resumen_por_variable): si ambas tasas son
    parecidas, el TVL crece la mayor parte del tiempo de
    cualquier forma (adopción estructural) y el evento de señal
    no aporta información adicional sobre la probabilidad de
    crecimiento.

    Args:
        df: Dataset diario.
        col_tvl: Columna de TVL a evaluar.
        ventana: Largo de la ventana, en días (mismo criterio
            que las ventanas condicionales).

    Returns:
        Dict con n_ventanas, n_crecientes, pct_creciente.
    """
    tvl = df[col_tvl]
    n = len(tvl)
    if n <= ventana:
        return {
            "n_ventanas": 0, "n_crecientes": 0,
            "pct_creciente": None,
        }
    n_total = 0
    n_crecientes = 0
    for pos in range(n - ventana):
        v0 = tvl.iloc[pos]
        v1 = tvl.iloc[pos + ventana]
        if pd.isna(v0) or pd.isna(v1) or v0 <= 0:
            continue
        n_total += 1
        if v1 > v0:
            n_crecientes += 1
    pct = (
        round(100.0 * n_crecientes / n_total, 1)
        if n_total > 0 else None
    )
    return {
        "n_ventanas": n_total,
        "n_crecientes": n_crecientes,
        "pct_creciente": pct,
    }


def evaluar_eventos(
    df: pd.DataFrame,
    col_signal: str,
    eventos: list[int],
    ventana: int = VENTANA_DIAS,
) -> pd.DataFrame:
    """
    Para cada evento, calcula la variación de TVL entre el día
    del evento y ventana días después.

    Descarta eventos sin historia posterior completa.
    """
    tvl = df[COL_TVL]
    signal = df[col_signal]
    idx = df.index

    filas: list[dict] = []

    for pos in eventos:
        pos_fin = pos + ventana
        if pos_fin >= len(df):
            continue

        tvl_inicio = tvl.iloc[pos]
        tvl_fin = tvl.iloc[pos_fin]
        tasa_valor = signal.iloc[pos]

        if (
            pd.isna(tvl_inicio)
            or pd.isna(tvl_fin)
            or pd.isna(tasa_valor)
            or tvl_inicio <= 0
        ):
            continue

        delta_usd = tvl_fin - tvl_inicio
        delta_pct = 100.0 * delta_usd / tvl_inicio

        filas.append({
            "variable_signal": col_signal,
            "fecha_evento": idx[pos],
            "tasa_valor": round(float(tasa_valor), 4),
            "tvl_inicio": round(float(tvl_inicio), 2),
            "tvl_fin": round(float(tvl_fin), 2),
            "delta_usd": round(float(delta_usd), 2),
            "delta_pct": round(float(delta_pct), 2),
            "pos_evento": pos,
        })

    return pd.DataFrame(filas)


def resumen_por_variable(
    df_eventos: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula media, desvío y % de eventos positivos por variable.
    """
    filas: list[dict] = []

    for var, grp in df_eventos.groupby("variable_signal"):
        n = len(grp)
        media = grp["delta_pct"].mean()
        std = grp["delta_pct"].std()
        pct_positivo = 100.0 * (grp["delta_pct"] > 0).sum() / n

        filas.append({
            "variable_signal": var,
            "n_eventos": n,
            "delta_pct_media": round(float(media), 2),
            "delta_pct_std": round(float(std), 2),
            "pct_eventos_positivos": round(float(pct_positivo), 1),
        })

    return pd.DataFrame(filas)


def top_bottom(
    df_eventos: pd.DataFrame,
    n: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Devuelve los n mejores y n peores ejemplos por delta_pct.
    """
    ordenado = df_eventos.sort_values(
        "delta_pct", ascending=False,
    )
    top = ordenado.head(n).copy()
    bottom = ordenado.tail(n).copy()
    return top, bottom


def graficar_top3(
    df: pd.DataFrame,
    top3: pd.DataFrame,
    ruta_png: str,
    ventana: int = VENTANA_DIAS,
) -> None:
    """
    Grafica la trayectoria del TVL de treasurybills en los
    49 días post-señal para los 3 mejores ejemplos.
    """
    tvl = df[COL_TVL]

    fig, ax = plt.subplots(figsize=(12, 6))
    colores = ["#2563eb", "#dc2626", "#16a34a"]

    for i, (_, row) in enumerate(top3.iterrows()):
        pos = int(row["pos_evento"])
        pos_fin = pos + ventana + 1
        if pos_fin > len(tvl):
            continue

        segmento = tvl.iloc[pos:pos_fin].values
        if len(segmento) == 0:
            continue

        normalizado = 100.0 * segmento / segmento[0]
        dias = list(range(len(normalizado)))

        fecha_str = str(row["fecha_evento"].date())
        label = (
            f"{fecha_str} "
            f"({row['variable_signal'].replace('macro_', '')}, "
            f"Δ={row['delta_pct']:+.1f}%)"
        )

        ax.plot(
            dias, normalizado,
            color=colores[i % len(colores)],
            linewidth=2,
            marker="o",
            markersize=3,
            label=label,
        )

    ax.axhline(
        100, color="gray", linestyle="--",
        linewidth=1, alpha=0.5,
    )
    ax.set_xlabel(
        "Días desde el evento de señal", fontsize=12,
    )
    ax.set_ylabel(
        "TVL treasurybills (base 100 = día del evento)",
        fontsize=12,
    )
    ax.set_title(
        "Event Study: TVL de Tokenized Treasuries\n"
        "post-señal de tasa (3 mejores ejemplos)",
        fontsize=14,
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    ax.annotate(
        "Señal: subida > 1σ sobre MA(30) en SOFR / Fed Funds / TSY_3M\n"
        "No es recomendación de inversión",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=8,
        color="gray",
        style="italic",
    )

    fig.tight_layout()
    fig.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if DEBUG:
        print(f"[DEBUG] Gráfico guardado: {ruta_png}")


# ════════════════════════════════════════════════════════════
# 5. Proceso principal
# ════════════════════════════════════════════════════════════


def main() -> None:
    # CAMBIO B1-4 (Bloque 1): hash de las entradas vs. manifest.
    verificar_entradas_snapshot([ARCHIVO_COMPLETAS], "Job 14")

    print(
        "\n[DEBUG] ═══════════════════════════════════════════"
        "═══════════════"
    )
    print(
        "[DEBUG] Job 14 — Event Study: Señales de Tasa → TVL "
        "Treasuries"
    )
    print(
        "[DEBUG] ═══════════════════════════════════════════"
        "═══════════════"
    )

    # CAMBIO E2E-14-01: validar productor y esquema antes del Event Study.
    if not os.path.isfile(ARCHIVO_COMPLETAS):
        raise FileNotFoundError(
            f"[DEBUG][ERROR] Job 14: falta {ARCHIVO_COMPLETAS}. "
            "Debe producirlo el Job 4 (series_completas.csv)."
        )
    try:
        df = cargar_series(ARCHIVO_COMPLETAS)
    except (OSError, ValueError, KeyError, TypeError, UnicodeError,
            pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 14: no se puede leer "
            f"{ARCHIVO_COMPLETAS}: {exc}"
        ) from exc
    validar_entrada_job14(df, ARCHIVO_COMPLETAS)

    # ── Detectar y evaluar eventos por variable ────────────
    todos_eventos: list[pd.DataFrame] = []

    for col_signal in COLS_SIGNAL:
        serie = df[col_signal].copy()
        n_validos = int(serie.notna().sum())
        if DEBUG:
            print(
                f"\n[DEBUG] {col_signal}: {n_validos} obs válidas"
            )

        eventos = detectar_eventos(serie)
        if DEBUG:
            print(
                f"[DEBUG]   Eventos detectados: {len(eventos)}"
            )

        # CAMBIO I6-b: se descartan eventos solapados (ventanas
        # de evaluación que se pisan entre sí).
        eventos, n_solapados = eliminar_eventos_solapados(
            eventos
        )
        if DEBUG:
            print(
                f"[DEBUG]   Eventos sin solape: {len(eventos)} "
                f"({n_solapados} descartados por solape "
                f"<{VENTANA_DIAS}d con el evento anterior)"
            )

        if not eventos:
            print(f"[DEBUG]   Sin eventos para {col_signal}")
            continue

        df_ev = evaluar_eventos(df, col_signal, eventos)
        if DEBUG:
            print(
                f"[DEBUG]   Eventos con 49 días completos: "
                f"{len(df_ev)}"
            )

        if not df_ev.empty:
            todos_eventos.append(df_ev)

    if not todos_eventos:
        # CAMBIO E2E-14-01: actualizar ambas salidas, no dejar las viejas.
        guardar_sin_eventos_job14(
            "No hubo eventos con ventana post-senal completa y TVL valido."
        )
        return

    df_all = pd.concat(todos_eventos, ignore_index=True)
    if DEBUG:
        print(f"\n[DEBUG] Total eventos evaluados: {len(df_all)}")

    # ── Resumen por variable ───────────────────────────────
    df_resumen = resumen_por_variable(df_all)

    print("\n" + "═" * 64)
    print("RESUMEN POR VARIABLE DE SEÑAL")
    print("═" * 64)
    for _, row in df_resumen.iterrows():
        print(
            f"\n  {row['variable_signal']}:"
        )
        print(
            f"    N eventos:       {row['n_eventos']}"
        )
        print(
            f"    ΔTVL media:      {row['delta_pct_media']:+.2f}% "
            f"(σ = {row['delta_pct_std']:.2f}%)"
        )
        print(
            f"    % con ΔTVL > 0:  {row['pct_eventos_positivos']:.1f}%"
        )

    # ── Resumen global ─────────────────────────────────────
    media_global = df_all["delta_pct"].mean()
    std_global = df_all["delta_pct"].std()
    pct_positivo_global = (
        100.0 * (df_all["delta_pct"] > 0).sum() / len(df_all)
    )

    print("\n" + "═" * 64)
    print("RESUMEN GLOBAL (todos los eventos)")
    print("═" * 64)
    print(
        f"  Total eventos:     {len(df_all)}"
    )
    print(
        f"  ΔTVL media:        {media_global:+.2f}% "
        f"(σ = {std_global:.2f}%)"
    )
    print(
        f"  % con ΔTVL > 0:   {pct_positivo_global:.1f}%"
    )

    # CAMBIO I6-a: tasa base, sin condicionar a ninguna señal,
    # para poder juzgar si el % condicional de arriba es o no
    # distinto del comportamiento "de fondo" del TVL.
    base = tasa_base_tvl_creciente(df)
    if base["pct_creciente"] is not None:
        print(
            f"  % base (sin señal, todas las ventanas de "
            f"{VENTANA_DIAS}d): {base['pct_creciente']:.1f}% "
            f"({base['n_crecientes']}/{base['n_ventanas']} "
            "ventanas con TVL creciente)"
        )
        _diff_base = pct_positivo_global - base["pct_creciente"]
        print(
            f"  Diferencia condicional − base: {_diff_base:+.1f} "
            "puntos porcentuales"
        )
    else:
        print(
            "  % base: no se pudo calcular (serie muy corta)."
        )

    print(
        f"\n  → {'Sí' if pct_positivo_global > 50 else 'No'}, "
        f"en la mayoría de los eventos "
        f"({'sí' if pct_positivo_global > 50 else 'no'} "
        f"se cumplió la señal): "
        f"{pct_positivo_global:.1f}% de los eventos "
        f"tuvieron ΔTVL positivo en 7 semanas."
    )

    # ── Top 3 y Bottom 3 ──────────────────────────────────
    top3, bottom3 = top_bottom(df_all, n=3)

    print("\n" + "═" * 64)
    print("TOP 3 MEJORES EJEMPLOS (mayor ΔTVL %)")
    print("═" * 64)
    for _, row in top3.iterrows():
        print(
            f"  {row['fecha_evento'].date()} "
            f"| {row['variable_signal']:20s} "
            f"| tasa={row['tasa_valor']:.4f} "
            f"| TVL: {row['tvl_inicio']:,.0f} → "
            f"{row['tvl_fin']:,.0f} "
            f"| Δ={row['delta_pct']:+.2f}%"
        )

    print("\n" + "═" * 64)
    print("BOTTOM 3 PEORES EJEMPLOS (menor ΔTVL %)")
    print("═" * 64)
    for _, row in bottom3.iterrows():
        print(
            f"  {row['fecha_evento'].date()} "
            f"| {row['variable_signal']:20s} "
            f"| tasa={row['tasa_valor']:.4f} "
            f"| TVL: {row['tvl_inicio']:,.0f} → "
            f"{row['tvl_fin']:,.0f} "
            f"| Δ={row['delta_pct']:+.2f}%"
        )

    # ── Guardar resultados ─────────────────────────────────
    if SAVE_DATA:
        os.makedirs(DIR_REPORTES, exist_ok=True)
        os.makedirs(DIR_FIGURAS, exist_ok=True)

        # CAMBIO E2E-14-01: mismo orden de columnas del original.
        df_all[COLUMNAS_EVENT_STUDY].to_csv(ARCHIVO_CSV, index=False)
        print(f"\n[DEBUG] CSV guardado: {ARCHIVO_CSV}")
        guardar_resumen_job14(  # CAMBIO B4-2
            round(float(pct_positivo_global), 2),
            base["pct_creciente"], int(len(df_all)),
            int(base["n_ventanas"]),
            "" if base["pct_creciente"] is not None
            else "tasa base no calculable (serie corta)",
            n_positivos=int((df_all["delta_pct"] > 0).sum()),
        )

        graficar_top3(df, top3, ARCHIVO_PNG)

    print("\n[DEBUG] Job 14 completado.\n")


if __name__ == "__main__":
    main()