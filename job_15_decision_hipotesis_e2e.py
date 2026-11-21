"""# Job 15 - Decisión de la Hipótesis"""

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

import pandas as pd

from google.colab import drive

drive.mount("/content/drive")

# ════════════════════════════════════════════════════════════
# 3. Paths y Constantes
# ════════════════════════════════════════════════════════════

# CAMBIO B6-1: la DECISIÓN vivía en Job 11, que corre antes que
# Job 14 y no encontraba event_study_resumen.csv ("no decidible").
# Este Job corre al final: lee convergencia_niveles.csv (Job 11) y
# event_study_resumen.csv (Job 14). Regla y lógica sin cambios.
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
# Salida de Job 11 (pares que convergen en los 4 niveles).
ARCHIVO_CONV: str = os.path.join(
    DIR_REPORTES, "convergencia_niveles.csv"
)

NO_EVALUABLE: str = "no evaluable"

# ── CAMBIO B4-1 (Bloque 4): REGLA DE DECISIÓN DE LA HIPÓTESIS ──
# CAMBIO B6-1: movida sin cambios desde Job 11.
# La completa el autor ANTES de la corrida final y NO se modifica
# después de ver el resultado. Mientras falte un valor, la
# conclusión es "no decidible" (no se elige un umbral a posteriori).
#   Precedencia: al menos REGLA_MIN_PARES_CONVERGEN par(es)
#                convergen en los 4 niveles (converge_4_niveles).
#   Utilidad:    el % condicional del event study (Job 14) supera
#                a la tasa base según un test binomial unilateral
#                (H0: p = p_base; H1: p > p_base) con
#                p-valor < REGLA_ALFA_BINOMIAL. Sin umbral fijo en
#                puntos porcentuales.
# Fijada por el autor el 2026-10-27.
REGLA_MIN_PARES_CONVERGEN: int | None = 1         # [N]
REGLA_ALFA_BINOMIAL: float | None = 0.05          # [X]: test binomial
REGLA_FECHA_FIJADA: str | None = "2026-10-27"

# Resumen que escribe Job 14 (correr job_11 y job_14 antes).
ARCHIVO_EVENT_RESUMEN: str = os.path.join(
    DIR_REPORTES, "event_study_resumen.csv"
)
ARCHIVO_DECISION: str = os.path.join(
    DIR_REPORTES, "decision_hipotesis.csv"
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

# ════════════════════════════════════════════════════════════
# 4. Funciones
# ════════════════════════════════════════════════════════════


def contar_pares_convergen() -> int:
    """
    Cuenta los pares con converge_4_niveles verdadero en
    convergencia_niveles.csv (Job 11). Si el archivo no existe,
    está vacío o no tiene la columna, devuelve 0 y lo registra
    con [DEBUG][ALERTA].

    CAMBIO B6-1.

    Returns:
        Pares que convergen en los 4 niveles.
    """
    if not os.path.isfile(ARCHIVO_CONV):
        print(
            f"[DEBUG][ALERTA] Job 15: falta {ARCHIVO_CONV} (correr "
            "job_11 y job_14 antes) → n_converge = 0"
        )
        return 0
    try:
        _conv = pd.read_csv(ARCHIVO_CONV)
    except pd.errors.EmptyDataError:
        _conv = pd.DataFrame()
    except (OSError, ValueError) as e:
        print(f"[DEBUG][ERROR] Job 15: {ARCHIVO_CONV} ilegible: {e}")
        _conv = pd.DataFrame()
    if _conv.empty or "converge_4_niveles" not in _conv.columns:
        print(
            f"[DEBUG][ALERTA] Job 15: {ARCHIVO_CONV} vacío o sin "
            "converge_4_niveles → n_converge = 0"
        )
        return 0
    # Tras el CSV puede llegar como bool o como texto "True"/"False".
    _n = int(
        _conv["converge_4_niveles"].astype(str).str.strip()
        .str.lower().eq("true").sum()
    )
    print(
        f"[DEBUG] Job 15: {_n} de {len(_conv)} par(es) con "
        "converge_4_niveles = True"
    )
    return _n


def leer_utilidad_job14() -> tuple[dict[str, Any] | None, str]:
    """
    Lee de event_study_resumen.csv (Job 14) los eventos totales,
    los eventos con ΔTVL > 0 y la tasa base. Devuelve (None,
    motivo) si falta, no es evaluable o corresponde a otro
    series_completas.

    CAMBIO B4-1 (Bloque 4). CAMBIO B6-1: movida desde Job 11.

    Returns:
        ({n_eventos, n_positivos, pct_condicional, pct_base} o
        None, detalle para el reporte).
    """
    if not os.path.isfile(ARCHIVO_EVENT_RESUMEN):
        return None, f"falta {ARCHIVO_EVENT_RESUMEN} (correr job_11 y job_14 antes)"
    try:
        _res = pd.read_csv(ARCHIVO_EVENT_RESUMEN).iloc[0]
    except (OSError, ValueError, IndexError) as e:
        print(f"[DEBUG][ERROR] Job 15: resumen de Job 14 ilegible: {e}")
        return None, "resumen de Job 14 ilegible"
    if _res.get("sha256_series_completas") != sha256_archivo(ARCHIVO_COMPLETAS):
        return None, (
            "el resumen de Job 14 es de otro series_completas.csv "
            "(re-correr Job 14)"
        )
    if (
        pd.isna(_res.get("pct_base"))
        or pd.isna(_res.get("n_positivos"))
        or not _res.get("n_eventos")
    ):
        return None, (
            "Job 14 no evaluable: "
            f"{_res.get('motivo_no_evaluable', '') or 'sin eventos/base'}"
        )
    _datos = {
        "n_eventos": int(_res["n_eventos"]),
        "n_positivos": int(_res["n_positivos"]),
        "pct_condicional": float(_res["pct_condicional"]),
        "pct_base": float(_res["pct_base"]),
    }
    return _datos, (
        f"{_datos['n_positivos']}/{_datos['n_eventos']} eventos con "
        f"ΔTVL > 0 = {_datos['pct_condicional']:.1f}% vs base "
        f"{_datos['pct_base']:.1f}%"
    )


def decidir_hipotesis(n_converge: int) -> dict[str, Any]:
    """
    Aplica la regla fijada de antemano e imprime el bloque
    DECISIÓN: cada condición con su valor y SÍ/NO, y la conclusión
    (sustentada / parcial / no sustentada). Si falta un umbral o
    un dato, la conclusión es "no decidible".

    Utilidad: test binomial unilateral (scipy.stats.binomtest,
    alternative="greater") de n_positivos sobre n_eventos contra
    p = pct_base / 100. Supuestos: eventos independientes y tasa
    base conocida (en realidad es estimada y las ventanas se
    solapan) — se declara como limitación.

    CAMBIO B4-1 (Bloque 4). CAMBIO B6-1: movida desde Job 11.

    Args:
        n_converge: Pares con converge_4_niveles == True.

    Returns:
        Dict con valores, cumplimiento y conclusión.
    """
    from scipy.stats import binomtest

    _datos, _detalle_util = leer_utilidad_job14()
    _pval: float | None = None
    if _datos is not None:
        _pval = float(binomtest(
            _datos["n_positivos"], _datos["n_eventos"],
            _datos["pct_base"] / 100.0, alternative="greater",
        ).pvalue)

    if REGLA_MIN_PARES_CONVERGEN is None:
        _prec: bool | str = NO_EVALUABLE
    else:
        _prec = n_converge >= REGLA_MIN_PARES_CONVERGEN
    if REGLA_ALFA_BINOMIAL is None or _pval is None:
        _util: bool | str = NO_EVALUABLE
    else:
        _util = _pval < REGLA_ALFA_BINOMIAL

    def _txt(v: bool | str) -> str:
        return "N/E" if isinstance(v, str) else ("SÍ" if v else "NO")

    _faltan = [
        _n for _n, _v in (
            ("N", REGLA_MIN_PARES_CONVERGEN),
            ("alfa binomial", REGLA_ALFA_BINOMIAL),
            ("fecha", REGLA_FECHA_FIJADA),
        ) if _v is None
    ]
    if isinstance(_prec, str) or isinstance(_util, str):
        _concl = "no decidible"
    elif _prec and _util:
        _concl = "sustentada"
    elif _prec or _util:
        _concl = "parcial"
    else:
        _concl = "no sustentada"

    print(f"\n{'═' * 64}")
    print("DECISIÓN — HIPÓTESIS (regla fijada antes de la corrida)")
    print("═" * 64)
    if _faltan:
        print(
            f"[DEBUG][ALERTA] Regla incompleta: falta {', '.join(_faltan)}"
            " en REGLA_* (Job 15). Se completa ANTES de correr."
        )
    else:
        print(f"Regla fijada el {REGLA_FECHA_FIJADA}.")
    print(
        f"  [{_txt(_prec)}] Precedencia: {n_converge} par(es) "
        "convergen en los 4 niveles | umbral: ≥ "
        f"{REGLA_MIN_PARES_CONVERGEN}"
    )
    print(
        f"  [{_txt(_util)}] Utilidad: {_detalle_util} | test binomial "
        "unilateral (p > p_base): p-valor = "
        + (f"{_pval:.4f}" if _pval is not None else "n/d")
        + f" | umbral: < {REGLA_ALFA_BINOMIAL}"
    )
    print(f"  CONCLUSIÓN: hipótesis {_concl.upper()}")
    print(
        "  Limitación: el test binomial supone eventos independientes "
        "y tasa base conocida; las ventanas se solapan y la base es "
        "estimada."
    )
    print("═" * 64)

    _decision = {
        "regla_fecha": REGLA_FECHA_FIJADA,
        "umbral_min_pares": REGLA_MIN_PARES_CONVERGEN,
        "alfa_binomial": REGLA_ALFA_BINOMIAL,
        "n_pares_convergen": n_converge,
        "precedencia_cumple": _txt(_prec),
        "utilidad_n_eventos": _datos["n_eventos"] if _datos else None,
        "utilidad_n_positivos": _datos["n_positivos"] if _datos else None,
        "utilidad_pct_base": _datos["pct_base"] if _datos else None,
        "utilidad_pvalor": _pval,
        "utilidad_detalle": _detalle_util,
        "utilidad_cumple": _txt(_util),
        "conclusion": _concl,
    }
    if SAVE_DATA:
        os.makedirs(DIR_REPORTES, exist_ok=True)
        pd.DataFrame([_decision]).to_csv(ARCHIVO_DECISION, index=False)
        print(f"[DEBUG] decisión guardada: {ARCHIVO_DECISION}")
    return _decision


# ════════════════════════════════════════════════════════════
# 5. Proceso
# ════════════════════════════════════════════════════════════


def main() -> None:
    """Cuenta pares convergentes y aplica la regla de decisión."""
    verificar_entradas_snapshot([ARCHIVO_COMPLETAS], "Job 15")
    n_converge = contar_pares_convergen()
    # Save: decidir_hipotesis guarda decision_hipotesis.csv.
    decidir_hipotesis(n_converge)


if __name__ == "__main__":
    main()
