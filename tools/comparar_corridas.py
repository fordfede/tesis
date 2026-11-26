"""# Bloque 5 - Reproducibilidad: comparación de corridas

Guarda las tablas de salida (CSV de outputs/reportes) de una corrida
y compara dos corridas por SHA-256, imprimiendo IGUAL/DIFIERE por
archivo. Si un archivo difiere, diagnostica la causa probable:
  · orden:        mismas filas en otro orden.
  · paralelismo:  mismos datos salvo diferencias numéricas mínimas
                  (suma en punto flotante en otro orden: n_jobs=-1
                  de Random Forest, hilos de XGBoost).
  · valores:      cambia el contenido (semilla, datos de entrada).

Uso en Colab (una celda por paso):

    import os, sys
    sys.path.insert(0, "/content/drive/MyDrive/Tesis/debug")  # carpeta de los jobs
    import comparar_corridas as cc

    # 1) Extracción que se defiende: genera el snapshot.
    cc.correr("descarga", cc.JOBS_TODOS)
    cc.guardar("descarga")

    # 2) Dos corridas en snapshot.
    cc.correr("snapshot", cc.JOBS_ANALISIS)
    cc.guardar("snapshot_1")
    cc.correr("snapshot", cc.JOBS_ANALISIS)
    cc.guardar("snapshot_2")

    # 3) Comparación.
    cc.comparar("snapshot_1", "snapshot_2")
"""

# 1. Configuración

DEBUG: bool = True

# 2. Imports

import glob
import hashlib
import json
import os
import shutil

import numpy as np
import pandas as pd

# 3. Paths y Constantes

BASE_DIR: str = os.environ.get(
    "TESIS_BASE_DIR", "/content/drive/MyDrive/Tesis/expansion"
)
DIR_REPORTES: str = os.path.join(BASE_DIR, "outputs", "reportes")
DIR_CORRIDAS: str = os.path.join(BASE_DIR, "outputs", "corridas")
DIR_JOBS: str = os.path.dirname(os.path.abspath(__file__))

# Job 14 antes que Job 11: la DECISIÓN de Job 11 lee el resumen
# del event study (CAMBIO B4-1).
JOBS_ANALISIS: list[str] = [
    "job_05", "job_06", "job_07", "job_08", "job_09",
    "job_10", "job_12", "job_13", "job_14", "job_11",
]
JOBS_TODOS: list[str] = [
    "job_01", "job_02", "job_03", "job_04",
] + JOBS_ANALISIS

# Diferencia numérica máxima que se atribuye a paralelismo.
TOL_FLOAT: float = 1e-9

# 4. Funciones


def sha256_archivo(ruta: str) -> str:
    """SHA-256 de un archivo, leído por bloques."""
    _h = hashlib.sha256()
    with open(ruta, "rb") as _f:
        for _bloque in iter(lambda: _f.read(1 << 20), b""):
            _h.update(_bloque)
    return _h.hexdigest()


def correr(modo: str, jobs: list[str]) -> None:
    """
    Corre los jobs en orden con TESIS_MODO_DATOS=modo y corta en el
    primer error. Usa runpy (no %run: IPython muestra el error pero
    no lo propaga, y la corrida seguiría con el job siguiente). El
    SystemExit(0) de los Jobs 1-3 en modo snapshot es un fin normal.

    Args:
        modo: "descarga" o "snapshot".
        jobs: Prefijos de archivo (ej. "job_05").
    """
    import runpy

    os.environ["TESIS_MODO_DATOS"] = modo
    for _job in jobs:
        _rutas = glob.glob(os.path.join(DIR_JOBS, f"{_job}_*.py"))
        if len(_rutas) != 1:
            raise FileNotFoundError(
                f"[DEBUG][ERROR] no se encuentra un único {_job}_*.py "
                f"en {DIR_JOBS}: {_rutas}"
            )
        print(f"\n[DEBUG] ═══ {os.path.basename(_rutas[0])} ({modo}) ═══")
        try:
            runpy.run_path(_rutas[0], run_name="__main__")
        except SystemExit as _fin:
            if _fin.code not in (0, None):
                raise


def guardar(nombre: str) -> str:
    """
    Copia los CSV de DIR_REPORTES (recursivo) a
    DIR_CORRIDAS/<nombre> y escribe hashes.json.

    Args:
        nombre: Nombre de la corrida (ej. "snapshot_1").

    Returns:
        Carpeta de la corrida.
    """
    _destino = os.path.join(DIR_CORRIDAS, nombre)
    if os.path.isdir(_destino):
        shutil.rmtree(_destino)
    _hashes: dict[str, str] = {}
    for _ruta in sorted(
        glob.glob(os.path.join(DIR_REPORTES, "**", "*.csv"), recursive=True)
    ):
        _rel = os.path.relpath(_ruta, DIR_REPORTES)
        os.makedirs(os.path.dirname(os.path.join(_destino, _rel)), exist_ok=True)
        shutil.copy2(_ruta, os.path.join(_destino, _rel))
        _hashes[_rel] = sha256_archivo(_ruta)
    with open(os.path.join(_destino, "hashes.json"), "w", encoding="utf-8") as _f:
        json.dump(_hashes, _f, indent=2, ensure_ascii=False)
    print(f"[DEBUG] corrida '{nombre}': {len(_hashes)} CSV en {_destino}")
    return _destino


def diagnosticar(ruta_a: str, ruta_b: str) -> str:
    """
    Causa probable de que dos CSV difieran: orden, paralelismo
    (diferencias de punto flotante <= TOL_FLOAT) o valores.

    Args:
        ruta_a: CSV de la corrida A.
        ruta_b: CSV de la corrida B.

    Returns:
        Texto breve con la causa probable.
    """
    try:
        _a = pd.read_csv(ruta_a)
        _b = pd.read_csv(ruta_b)
    except (OSError, ValueError) as e:
        return f"no se pudo leer como tabla: {e}"
    if list(_a.columns) != list(_b.columns):
        return "valores: columnas distintas"
    if _a.shape != _b.shape:
        return f"valores: filas distintas ({len(_a)} vs {len(_b)})"
    _num = _a.select_dtypes("number").columns.tolist()
    _txt = [c for c in _a.columns if c not in _num]

    def _iguales(x: pd.DataFrame, y: pd.DataFrame) -> tuple[bool, float]:
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)
        if not x[_txt].astype(str).equals(y[_txt].astype(str)):
            return False, float("inf")
        if not _num:
            return True, 0.0
        _xa = x[_num].to_numpy(dtype=float)
        _ya = y[_num].to_numpy(dtype=float)
        if not np.array_equal(np.isnan(_xa), np.isnan(_ya)):
            return False, float("inf")
        _d = np.nan_to_num(np.abs(_xa - _ya), nan=0.0)
        return True, float(_d.max()) if _d.size else 0.0

    _ok, _dmax = _iguales(_a, _b)
    if _ok and _dmax <= TOL_FLOAT:
        return f"paralelismo: mismos datos, diferencia máx {_dmax:.2e}"
    _orden = list(_a.columns)
    _ok_s, _dmax_s = _iguales(
        _a.sort_values(_orden, kind="mergesort"),
        _b.sort_values(_orden, kind="mergesort"),
    )
    if _ok_s and _dmax_s <= TOL_FLOAT:
        return "orden: mismas filas en otro orden"
    _dmax_txt = f"{_dmax:.3g}" if np.isfinite(_dmax) else "texto distinto"
    return f"valores: diferencia máx {_dmax_txt} (semilla o datos)"


def comparar(nombre_a: str, nombre_b: str) -> bool:
    """
    Compara por SHA-256 los CSV de dos corridas guardadas e
    imprime IGUAL/DIFIERE por archivo (y SOLO EN A/B).

    Args:
        nombre_a: Corrida A (carpeta en DIR_CORRIDAS).
        nombre_b: Corrida B.

    Returns:
        True si todo es IGUAL.
    """
    _dir_a = os.path.join(DIR_CORRIDAS, nombre_a)
    _dir_b = os.path.join(DIR_CORRIDAS, nombre_b)
    _h = []
    for _d in (_dir_a, _dir_b):
        with open(os.path.join(_d, "hashes.json"), encoding="utf-8") as _f:
            _h.append(json.load(_f))
    _ha, _hb = _h
    _n_dif = 0
    print(f"\n{'═' * 64}\nCOMPARACIÓN {nombre_a} vs {nombre_b}\n{'═' * 64}")
    for _rel in sorted(set(_ha) | set(_hb)):
        if _rel not in _hb:
            print(f"  SOLO EN {nombre_a:<10} {_rel}")
            _n_dif += 1
        elif _rel not in _ha:
            print(f"  SOLO EN {nombre_b:<10} {_rel}")
            _n_dif += 1
        elif _ha[_rel] == _hb[_rel]:
            print(f"  IGUAL              {_rel}")
        else:
            _causa = diagnosticar(
                os.path.join(_dir_a, _rel), os.path.join(_dir_b, _rel)
            )
            print(f"  DIFIERE            {_rel}  → {_causa}")
            _n_dif += 1
    _total = len(set(_ha) | set(_hb))
    print(
        f"\n{_total - _n_dif}/{_total} IGUAL"
        + ("" if _n_dif == 0 else f" | {_n_dif} con diferencias")
    )
    return _n_dif == 0


# 5. Proceso (línea de comandos)

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "guardar":
        guardar(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "comparar":
        sys.exit(0 if comparar(sys.argv[2], sys.argv[3]) else 1)
    else:
        print(
            "Uso: python comparar_corridas.py guardar <nombre>\n"
            "     python comparar_corridas.py comparar <A> <B>"
        )
        sys.exit(2)
