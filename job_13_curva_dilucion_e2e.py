"""# Job 13 - Curva de Dilución: Rendimiento Efectivo de USDY vs. Timing de Entrada"""

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

# CAMBIO E2E-13-01: misma ruta predeterminada, configurable por entorno.
BASE_DIR: str = os.environ.get(
    "TESIS_BASE_DIR", "/content/drive/MyDrive/Tesis/expansion"
)
DIR_PROCESSED: str = os.path.join(BASE_DIR, "data", "processed")
DIR_REPORTES: str = os.path.join(
    BASE_DIR, "outputs", "reportes"
)
DIR_FIGURAS: str = os.path.join(
    DIR_REPORTES, "figuras_walkforward"
)

ARCHIVO_COMPLETAS: str = os.path.join(
    DIR_PROCESSED, "series_completas.csv"
)
ARCHIVO_DILUCION: str = os.path.join(
    DIR_REPORTES, "curva_dilucion.csv"
)
ARCHIVO_PNG: str = os.path.join(
    DIR_FIGURAS, "curva_dilucion.png"
)

COL_TSY: str = "macro_tsy_1m"
COL_TVL: str = "rwa_usdy_supply"

CAPITAL: float = 1_000_000.0
VENTANA_DIAS: int = 50         # 7 semanas ≈ 49 días + día 0
DIAS_ENTRADA: list[int] = [0, 7, 14, 21, 28, 35, 42, 49]
MA_VENTANA: int = 30           # Media móvil para detectar señal
UMBRAL_SIGMA: float = 1.0      # Desvíos sobre la MA
COOLDOWN_EVENTOS: int = 14     # Días entre eventos

# CAMBIO E2E-13-01: esquema de salida solo si no hay curva evaluable.
COLUMNAS_CURVA: list[str] = [
    "dia_entrada", "n_eventos", "ganancia_media_usd",
    "ganancia_std_usd", "ganancia_p25_usd", "ganancia_p75_usd",
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


# CAMBIO E2E-13-01: validar solo el contrato de entrada del Job 4.
def validar_entrada_job13(df: pd.DataFrame, ruta: str) -> None:
    """Valida fechas y tipos que usan los calculos, sin cambiar valores."""
    if df.empty or len(df.columns) == 0:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 13: {ruta} sin registros o columnas "
            "(productor: Job 4)."
        )
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 13: fechas invalidas en {ruta} "
            "(productor: Job 4)."
        )
    if df.index.has_duplicates:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 13: fechas duplicadas en {ruta} "
            "(productor: Job 4)."
        )
    if not df.columns.is_unique:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 13: columnas duplicadas en {ruta} "
            "(productor: Job 4)."
        )
    for columna in (COL_TSY, COL_TVL):
        if columna not in df.columns:
            raise RuntimeError(
                f"[DEBUG][ERROR] Job 13: columna '{columna}' ausente "
                f"en {ruta} (productor: Job 4)."
            )
        if not pd.api.types.is_numeric_dtype(df[columna]):
            raise RuntimeError(
                f"[DEBUG][ERROR] Job 13: columna '{columna}' no "
                f"numerica en {ruta} (productor: Job 4)."
            )


def detectar_eventos(
    tsy: pd.Series,
    ma_ventana: int = MA_VENTANA,
    umbral_sigma: float = UMBRAL_SIGMA,
    cooldown: int = COOLDOWN_EVENTOS,
) -> list[int]:
    """
    Detecta eventos de señal: días donde macro_tsy_1m sube
    más de umbral_sigma desvíos sobre su media móvil.

    Aplica cooldown para no contar el mismo shock dos veces.

    Returns:
        Lista de posiciones (iloc) de los eventos.
    """
    ma = tsy.rolling(ma_ventana).mean()
    std = tsy.rolling(ma_ventana).std()

    diff = tsy - ma
    umbral = umbral_sigma * std

    candidatos = diff > umbral
    candidatos = candidatos.fillna(False)

    eventos: list[int] = []
    ultimo: int = -cooldown - 1

    for i in range(ma_ventana, len(tsy)):
        if candidatos.iloc[i] and (i - ultimo) > cooldown:
            eventos.append(i)
            ultimo = i

    return eventos


def ganancia_acumulada_desde(
    tasa_anual: np.ndarray,
    tvl: np.ndarray,
    inicio: int,
    ventana: int,
    capital: float,
) -> tuple[float | None, str]:
    """
    Ganancia acumulada en USD de invertir capital desde
    posición inicio durante ventana días.

    Fórmula:
        ganancia_base_diaria(t) = capital × (tasa_anual(t) / 100 / 365)
        factor_dilucion(t)      = TVL(inicio) / TVL(t)
        ganancia = Σ ganancia_base_diaria(t) × factor_dilucion(t)

    CAMBIO I6-d: devuelve también el motivo del descarte ("" si
    hay ganancia), para poder contabilizar POR QUÉ se pierde un
    día de entrada: "ventana incompleta" (el evento+ventana se
    pasa del final de la serie) o "TVL día 0 inválido" (NaN o
    ≤0 justo en el día de entrada).

    Returns:
        (ganancia, motivo). Ganancia es None si motivo != "".
    """
    fin = inicio + ventana
    if fin > len(tasa_anual):
        return None, "ventana incompleta"

    tvl_dia0 = tvl[inicio]
    if np.isnan(tvl_dia0) or tvl_dia0 <= 0:
        return None, "TVL día 0 inválido"

    ganancia = 0.0
    for t in range(inicio, fin):
        ta = tasa_anual[t]
        tv = tvl[t]
        if np.isnan(ta) or np.isnan(tv) or tv <= 0:
            continue
        base_diaria = capital * (ta / 100.0 / 365.0)
        factor_dilucion = tvl_dia0 / tv
        ganancia += base_diaria * factor_dilucion

    return ganancia, ""


def simular_evento(
    tasa_anual: np.ndarray,
    tvl: np.ndarray,
    pos_evento: int,
    dias_entrada: list[int],
    ventana: int,
    capital: float,
) -> tuple[dict[int, float | None], dict[int, str]]:
    """
    Para un evento, calcula la ganancia acumulada por cada
    día de entrada relativo al evento.

    CAMBIO I6-d: devuelve también el motivo de descarte por día
    de entrada ("" si hubo ganancia), para poder reportar por
    qué se pierde cada (evento, día de entrada).

    Returns:
        (resultados, motivos).
    """
    resultados: dict[int, float | None] = {}
    motivos: dict[int, str] = {}
    for d in dias_entrada:
        pos_inicio = pos_evento + d
        if pos_inicio < 0 or pos_inicio >= len(tasa_anual):
            resultados[d] = None
            motivos[d] = "día de entrada fuera de rango"
            continue
        ganancia, motivo = ganancia_acumulada_desde(
            tasa_anual, tvl, pos_inicio, ventana, capital,
        )
        resultados[d] = ganancia
        motivos[d] = motivo
    return resultados, motivos


def curva_dilucion_promedio(
    eventos_ganancias: list[dict[int, float | None]],
    dias_entrada: list[int],
) -> pd.DataFrame:
    """
    Calcula la curva de dilución promedio across todos los
    eventos.
    """
    filas: list[dict] = []

    for d in dias_entrada:
        valores = [
            eg[d] for eg in eventos_ganancias
            if eg.get(d) is not None
        ]
        if not valores:
            filas.append({
                "dia_entrada": d,
                "n_eventos": 0,
                "ganancia_media_usd": None,
                "ganancia_std_usd": None,
                "ganancia_p25_usd": None,
                "ganancia_p75_usd": None,
            })
            continue

        arr = np.array(valores)
        filas.append({
            "dia_entrada": d,
            "n_eventos": len(arr),
            "ganancia_media_usd": round(float(np.mean(arr)), 2),
            "ganancia_std_usd": round(float(np.std(arr)), 2),
            "ganancia_p25_usd": round(
                float(np.percentile(arr, 25)), 2,
            ),
            "ganancia_p75_usd": round(
                float(np.percentile(arr, 75)), 2,
            ),
        })

    return pd.DataFrame(filas)


def ganancia_diferencial(
    df_curva: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calcula la ventaja media de entrar en día 0 vs día 49.
    """
    row_0 = df_curva[df_curva["dia_entrada"] == 0]
    row_49 = df_curva[df_curva["dia_entrada"] == 49]

    if row_0.empty or row_49.empty:
        return {"diferencial_media_usd": None}

    g0 = row_0.iloc[0]["ganancia_media_usd"]
    g49 = row_49.iloc[0]["ganancia_media_usd"]

    if g0 is None or g49 is None:
        return {"diferencial_media_usd": None}

    return {
        "ganancia_dia0_usd": g0,
        "ganancia_dia49_usd": g49,
        "diferencial_media_usd": round(g0 - g49, 2),
        "diferencial_pct": round(
            100.0 * (g0 - g49) / abs(g49), 2,
        ) if abs(g49) > 0.01 else None,
    }


def graficar_curva(
    df_curva: pd.DataFrame,
    ruta_png: str,
) -> None:
    """
    Grafica la curva de dilución con banda ±1 desvío.
    """
    validos = df_curva[
        df_curva["ganancia_media_usd"].notna()
    ].copy()
    if validos.empty:
        if DEBUG:
            print("[DEBUG] Sin datos para graficar")
        return

    x = validos["dia_entrada"].values
    y = validos["ganancia_media_usd"].values
    std = validos["ganancia_std_usd"].fillna(0).values

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        x, y, "o-", color="#2563eb", linewidth=2,
        markersize=8, label="Ganancia media",
    )
    ax.fill_between(
        x, y - std, y + std,
        alpha=0.2, color="#2563eb",
        label="±1 desvío",
    )
    ax.set_xlabel(
        "Día de entrada (relativo al evento de señal)",
        fontsize=12,
    )
    ax.set_ylabel(
        "Ganancia acumulada (USD por $1M invertido)",
        fontsize=12,
    )
    ax.set_title(
        "Curva de Dilución: Rendimiento Efectivo USDY\n"
        "según timing de entrada post-señal",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax.annotate(
        "Proxy: DGS1MO (tasa Treasury 1 mes)\n"
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


# CAMBIO E2E-13-01: salidas honestas para una corrida NO EVALUABLE.
def guardar_sin_curva_job13(motivo: str) -> None:
    """Sobrescribe las dos salidas con un estado no evaluable, sin datos."""
    if not SAVE_DATA:
        if DEBUG:
            print("[DEBUG] SAVE_DATA=False — nada guardado")
        return

    os.makedirs(DIR_REPORTES, exist_ok=True)
    os.makedirs(DIR_FIGURAS, exist_ok=True)
    pd.DataFrame(columns=COLUMNAS_CURVA).to_csv(
        ARCHIVO_DILUCION, index=False,
    )
    # Un PNG anterior podria aparentar ser la curva de esta corrida.
    # No se grafica ninguna observacion: el mensaje es explicito.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.text(
        0.5, 0.55, "CURVA DE DILUCION — NO EVALUABLE",
        ha="center", va="center", fontsize=16,
        transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.43, motivo,
        ha="center", va="center", fontsize=11,
        transform=ax.transAxes,
    )
    fig.savefig(ARCHIVO_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if DEBUG:
        print(f"[DEBUG][ALERTA] Job 13: {motivo}")
        print(f"[DEBUG] Salidas actualizadas (sin resultados): "
              f"{ARCHIVO_DILUCION}; {ARCHIVO_PNG}")


# ════════════════════════════════════════════════════════════
# 5. Proceso principal
# ════════════════════════════════════════════════════════════


def main() -> None:
    # CAMBIO B1-4 (Bloque 1): hash de las entradas vs. manifest.
    verificar_entradas_snapshot([ARCHIVO_COMPLETAS], "Job 13")

    print(
        "\n[DEBUG] ═══════════════════════════════════════════"
        "═══════════════"
    )
    print(
        "[DEBUG] Job 13 — Curva de Dilución: USDY vs. Timing"
    )
    print(
        "[DEBUG] ═══════════════════════════════════════════"
        "═══════════════"
    )

    # CAMBIO E2E-13-01: productor real Job 4; validacion anticipada.
    if not os.path.isfile(ARCHIVO_COMPLETAS):
        raise FileNotFoundError(
            f"[DEBUG][ERROR] Job 13: falta {ARCHIVO_COMPLETAS}. "
            "Debe producirlo el Job 4 (series_completas.csv)."
        )
    try:
        df = cargar_series(ARCHIVO_COMPLETAS)
    except (OSError, ValueError, KeyError, UnicodeError,
            pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise RuntimeError(
            f"[DEBUG][ERROR] Job 13: no se puede leer "
            f"{ARCHIVO_COMPLETAS}: {exc}"
        ) from exc
    validar_entrada_job13(df, ARCHIVO_COMPLETAS)

    # Alinear y limpiar
    tsy = df[COL_TSY].copy()
    tvl = df[COL_TVL].copy()

    mask_valid = tsy.notna() & tvl.notna() & (tvl > 0)
    n_valid = int(mask_valid.sum())
    if DEBUG:
        print(
            f"[DEBUG] {COL_TSY}: {tsy.notna().sum()} obs | "
            f"{COL_TVL}: {tvl.notna().sum()} obs | "
            f"ambos válidos: {n_valid}"
        )

    # DGS1MO viene en porcentaje anual — pasar tal cual;
    # la conversión a diaria se hace dentro de ganancia_acumulada_desde
    tasa_anual = tsy.values
    tvl_arr = tvl.values

    # Detectar eventos
    eventos = detectar_eventos(tsy)
    if DEBUG:
        print(f"[DEBUG] Eventos detectados: {len(eventos)}")
        for i, pos in enumerate(eventos):
            print(
                f"  evento {i+1}: {tsy.index[pos].date()} "
                f"(pos={pos}, tsy={tsy.iloc[pos]:.4f}%)"
            )

    if not eventos:
        print(
            "\n[DEBUG] Sin eventos de señal. "
            "No se puede calcular la curva."
        )
        # CAMBIO E2E-13-01: no dejar resultados antiguos vigentes.
        guardar_sin_curva_job13("No se detectaron eventos de senal.")
        return

    # Simular ganancia por evento y día de entrada
    eventos_ganancias: list[dict[int, float | None]] = []
    eventos_validos: int = 0
    # CAMBIO I6-d: conteo de descartes por motivo, para día 0
    # (la condición que decide si el evento entra a la curva) y
    # para el resto de los días de entrada (acumulan aparte,
    # cada uno puede fallar sin invalidar el evento entero).
    motivos_dia0: dict[str, int] = {}
    motivos_otros_dias: dict[str, int] = {}

    for pos in eventos:
        resultado, motivos = simular_evento(
            tasa_anual, tvl_arr, pos,
            DIAS_ENTRADA, VENTANA_DIAS, CAPITAL,
        )
        if resultado.get(0) is not None:
            eventos_ganancias.append(resultado)
            eventos_validos += 1
        else:
            motivo_0 = motivos.get(0, "motivo desconocido")
            motivos_dia0[motivo_0] = (
                motivos_dia0.get(motivo_0, 0) + 1
            )
            if DEBUG:
                print(
                    f"[DEBUG] Evento en pos={pos} descartado: "
                    f"{motivo_0}"
                )
        for d, motivo in motivos.items():
            if d != 0 and motivo:
                motivos_otros_dias[motivo] = (
                    motivos_otros_dias.get(motivo, 0) + 1
                )

    if DEBUG:
        print(
            f"[DEBUG] Eventos con datos completos: "
            f"{eventos_validos}/{len(eventos)}"
        )
        if motivos_dia0:
            print(
                "[DEBUG] Eventos descartados por motivo "
                f"(día 0): {motivos_dia0}"
            )
        if motivos_otros_dias:
            print(
                "[DEBUG] Días de entrada ≠0 descartados por "
                f"motivo (no invalidan el evento): "
                f"{motivos_otros_dias}"
            )

    if not eventos_ganancias:
        print("[DEBUG] Sin eventos simulables.")
        # CAMBIO E2E-13-01: curva sin datos, no reutilizar grafico viejo.
        guardar_sin_curva_job13(
            "No hay eventos con ventana y TVL inicial validos."
        )
        return

    # Curva de dilución promedio
    df_curva = curva_dilucion_promedio(
        eventos_ganancias, DIAS_ENTRADA,
    )

    # Ganancia diferencial
    dif = ganancia_diferencial(df_curva)

    # ── Reporte ─────────────────────────────────────────────
    print("\n" + "═" * 64)
    print(
        "CURVA DE DILUCIÓN — RENDIMIENTO EFECTIVO USDY"
    )
    print("═" * 64)

    print(f"\nCapital simulado: ${CAPITAL:,.0f}")
    print(f"Ventana post-señal: {VENTANA_DIAS} días (7 semanas)")
    print(f"Eventos de señal: {eventos_validos}")
    print(
        f"Proxy de rendimiento: {COL_TSY} (DGS1MO, "
        "tasa anual Treasury 1 mes)"
    )

    print(f"\n{'─' * 64}")
    print(
        "Ganancia acumulada por día de entrada "
        f"(USD por ${CAPITAL/1e6:.0f}M):"
    )
    print("─" * 64)
    print(df_curva.to_string(index=False))

    print(f"\n{'─' * 64}")
    print("GANANCIA DIFERENCIAL (día 0 vs día 49):")
    print("─" * 64)
    if dif.get("diferencial_media_usd") is not None:
        print(
            f"  Ganancia día 0:  ${dif['ganancia_dia0_usd']:,.2f}"
        )
        print(
            f"  Ganancia día 49: ${dif['ganancia_dia49_usd']:,.2f}"
        )
        print(
            f"  Ventaja de entrar en día 0: "
            f"${dif['diferencial_media_usd']:,.2f}"
        )
        if dif.get("diferencial_pct") is not None:
            print(
                f"  Ventaja relativa: "
                f"{dif['diferencial_pct']:.1f}%"
            )
    else:
        print("  Datos insuficientes para calcular.")

    print(f"\n{'═' * 64}")
    print("SALVEDADES:")
    print(
        "  - Rendimiento proxy: DGS1MO (tasa Treasury 1 mes). "
        "El APY real de USDY puede diferir por fees y "
        "estructura del protocolo."
    )
    print(
        "  - La fracción capital/TVL es una simplificación: "
        "asume que el rendimiento se distribuye pro-rata "
        "entre todo el TVL."
    )
    print(
        "  - Mercado USDY de adopción incipiente (2025-2026): "
        "pocos eventos, alta varianza."
    )
    print(
        "  - NO es recomendación de inversión."
    )
    print("═" * 64)

    # ── Guardar ─────────────────────────────────────────────
    if SAVE_DATA:
        os.makedirs(DIR_REPORTES, exist_ok=True)
        os.makedirs(DIR_FIGURAS, exist_ok=True)

        guardados: list[str] = []

        if not df_curva.empty:
            df_curva.to_csv(ARCHIVO_DILUCION, index=False)
            guardados.append(ARCHIVO_DILUCION)

        graficar_curva(df_curva, ARCHIVO_PNG)
        guardados.append(ARCHIVO_PNG)

        if DEBUG:
            print("\n[DEBUG] Archivos guardados:")
            for g in guardados:
                print(f"  {g}")
    elif DEBUG:
        print("\n[DEBUG] SAVE_DATA=False — nada guardado")


if __name__ == "__main__":
    main()

