# Adopción de RWA tokenizados y condiciones monetarias de EE.UU.

**Análisis de adopción de RWA tokenizados en redes DLT: detección de señales anticipadas mediante modelos predictivos de aprendizaje supervisado, aplicado a la toma de decisiones sobre posicionamiento y gestión de riesgo (2021-2026).**

Autores: C. Federico Ford y Carla M. Veccio.

El trabajo estudia si las condiciones monetarias de EE.UU. (tasas de corto plazo, liquidez y condiciones financieras) anticipan la adopción de activos reales tokenizados (RWA), con qué ventana temporal y qué valor tiene esa señal para decidir sobre posicionamiento y riesgo. Usa correlación cruzada, causalidad de Granger con corrección por comparaciones múltiples, modelos lineales (Ridge y Lasso), no lineales (Random Forest y XGBoost) y validación walk-forward.

---

## 1. Contenido del repositorio

| Archivo | Qué es | ¿Necesario para el pipeline? |
|---|---|---|
| `e2e-model-v3.ipynb` | Pipeline completo (Jobs 0 a 17, más el 3m), con las salidas guardadas | Sí |
| `tools/append_maestro_tvl.ipynb` | Arma el maestro histórico de TVL por protocolo | Solo para los Jobs 3m y 17 |
| `tools/comparar_corridas.py` | Compara los CSV de dos corridas | No |
| `tools/comparar-datos.ipynb` | Comparación puntual entre dos fases de datos | No |
| `tools/Exportar-Resultados.ipynb` | Exporta a HTML las salidas de un notebook | No |
| `README.md` | Esta guía | No |

Los datos (`data/`) y los resultados (`outputs/`) los lee y escribe el notebook en Google Drive. No forman parte de esta lista de archivos.

> **Si solo querés ver los resultados:** abrí `e2e-model-v3.ipynb`. Tiene salidas guardadas y no hace falta ejecutar nada. Leé antes la [sección 7](#7-cosas-a-tener-en-cuenta).

---

## 2. Requisitos

- **Google Colab** y una cuenta de Google Drive. El notebook importa `google.colab` y monta Drive en `/content/drive`. Para correrlo fuera de Colab hay que modificar el código.
- **Python 3** con: `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `joblib` y `requests`. El notebook no instala paquetes ni fija versiones: usa las que tenga el entorno de Colab. Con otras versiones los resultados numéricos pueden variar.
- **API key de FRED**, solo para descargar datos nuevos.

### Configurar la API key de FRED

La key **no está incluida en el repositorio**. Cada persona usa la suya.

1. Creá una cuenta en [FRED](https://fred.stlouisfed.org/) y pedí una API key gratuita.
2. En Colab, abrí **Secrets** (ícono de la llave 🔑).
3. Agregá un secreto llamado `FRED_API_KEY` con tu key como valor.
4. Activá el acceso del notebook a ese secreto.

Si el código no puede leer el secreto con `google.colab.userdata`, busca la variable de entorno `FRED_API_KEY`.

No escribas la key en una celda ni la subas al repositorio. En modo `snapshot` no hace falta.

---

## 3. Cómo funciona

### Carpetas en Drive

La carpeta base por defecto es `/content/drive/MyDrive/Tesis/expansion`:

```text
MyDrive/Tesis/
├── expansion/                 ← TESIS_BASE_DIR
│   ├── data/
│   │   ├── raw/<fecha>/       respuestas crudas de las APIs
│   │   ├── processed/         series que usan los Jobs
│   │   └── snapshot/          copias congeladas + manifest.json
│   └── outputs/
│       ├── reportes/          CSV y figuras
│       └── modelos/           modelos .joblib
└── outputs_y_data/            ← TESIS_OUT_DIR (Jobs 3m y 17)
    ├── ejecucion_3/           copia de una corrida anterior
    ├── ejecucion_4/           copia de una corrida posterior
    ├── maestro/               salida de append_maestro_tvl.ipynb
    └── outputs/robustez/      salida del Job 17
```

### Modos de datos

| Modo | Qué hace | Para qué sirve |
|---|---|---|
| `descarga` (por defecto) | Consulta las APIs y, al guardar los datos, los congela en `data/snapshot/` | Actualizar el estudio |
| `snapshot` | Los Jobs 1, 2 y 3 restauran archivos congelados y terminan. El Job 4 restaura `series_completas.csv` sin consultar FRED. Los análisis verifican sus entradas | Reproducir con datos fijos |

El `manifest.json` guarda el hash SHA-256 de cada archivo congelado. Los Jobs 5 a 16 comparan sus entradas contra ese hash y se detienen si no coincide. En modo `descarga`, si una entrada no figura en el manifest, solo avisan.

> Una corrida en `descarga` **reemplaza el snapshot**. Antes de actualizar datos, hacé una copia de `expansion/` o usá otra carpeta base.

### Variables de entorno

| Variable | Valor por defecto |
|---|---|
| `TESIS_MODO_DATOS` | `descarga` |
| `TESIS_BASE_DIR` | `/content/drive/MyDrive/Tesis/expansion` |
| `TESIS_DIR_SNAPSHOT` | `<TESIS_BASE_DIR>/data/snapshot` |
| `TESIS_OUT_DIR` | `/content/drive/MyDrive/Tesis/outputs_y_data` |
| `TESIS_RUTA_EXT_ULTIMA` | `<TESIS_OUT_DIR>/ejecucion_4/data/processed/series_largas_ext.csv` |

Se definen con `os.environ[...]` antes de ejecutar los Jobs. Quedan activas hasta que se cambien o se reinicie el entorno.

---

## 4. Guía de uso

> ⚠️ **No uses "Ejecutar todo".** Debajo del título del Job 4 hay una celda de prueba que empieza con `!cp -r ... ejecucion_3 ... prueba_oct`. Cambia la carpeta base a `prueba_oct` y el modo a `snapshot` para todo lo que sigue. **Salteala en las dos opciones de abajo.** Además, en modo `snapshot` los Jobs 1 a 3 terminan con `SystemExit(0)`: es su final normal.

Si ya ejecutaste esa celda, reiniciá el entorno antes de seguir.

### Opción A: reproducir los análisis con datos congelados

No necesita API key. Necesitás estos archivos de la corrida original:

```text
expansion/data/snapshot/manifest.json
expansion/data/snapshot/series_completas.csv
expansion/data/processed/series_completas.csv
expansion/data/processed/series_largas_ext.csv
expansion/data/processed/series_diarias_ext.csv
```

Los CSV de `processed/` deben ser los mismos que registra el manifest. Si falta alguno, podés copiarlo desde `data/snapshot/`.

1. Copiá esos archivos a tu Drive respetando las rutas.
2. Abrí `e2e-model-v3.ipynb` en Colab.
3. En la primera celda (`# Modo`), dejá:
   ```python
   import os
   os.environ["TESIS_MODO_DATOS"] = "snapshot"
   ```
4. Ejecutá esa celda.
5. **Salteá los Jobs 0, 1, 2, 3 y 3m, y la celda `!cp`.**
6. Ejecutá en orden, de a una, las celdas de los **Jobs 4 a 16**.

Notas:

- El Job 4 restaura `series_completas.csv` y vuelve a correr sus pruebas y modelos. Necesita que `series_largas_ext.csv` ya esté en `processed/`.
- El Job 3 no se puede restaurar si el manifest atribuye sus archivos al Job 3m, que los reemplaza.
- El Job 3m no tiene un atajo de restauración: siempre recalcula desde el maestro y actualiza el snapshot.
- Los Jobs vuelven a escribir en `outputs/`. Trabajá sobre una copia si querés conservar los resultados originales.
- Para el Job 17, ver la [sección 6.1](#61-append_maestro_tvlipynb).

### Opción B: descargar datos nuevos

Los resultados pueden diferir de los guardados, porque cambian el período y las fuentes pueden revisar su historia.

1. Configurá `FRED_API_KEY`.
2. Definí una carpeta base nueva en la primera celda, para no pisar datos anteriores:
   ```python
   import os
   os.environ["TESIS_MODO_DATOS"] = "descarga"
   os.environ["TESIS_BASE_DIR"] = "/content/drive/MyDrive/Tesis/mi_corrida"
   ```
3. Ejecutá en orden los Jobs 0 (verificación de fuentes), 1, 2 y 3.
4. **Job 3m:** salteá esta celda si no tenés el maestro. En ese caso el pipeline sigue con las series del Job 3.
5. **Salteá la celda `!cp`.**
6. Ejecutá los Jobs 4 a 16.

La celda del Job 3m está justo después de la del Job 3, sin título propio en markdown. Si la ejecutás, reemplaza las series RWA del Job 3 por las del maestro. Las salidas guardadas de los Jobs 5 a 16 parten de datos generados con el 3m.

La fecha de corte de las series diarias del Job 1 está fija en el código (`FECHA_CORTE = "2026-08-08"`). Las series largas llegan hasta la fecha de ejecución.

---

## 5. El pipeline, Job por Job

### Adquisición y preparación

| Job | Qué hace | Salidas principales |
|---|---|---|
| **0** | Verifica qué fuentes RWA se pueden usar | `rwa_viabilidad.csv`, `rwa_protocolos_defillama.csv` |
| **1** | Adquisición base: actividad on-chain, USDY e índices de estrés | `series_diarias.csv`, `series_largas.csv` |
| **2** | Inventario y preselección de series RWA, con análisis de colinealidad | `inventario_rwa*.csv`, `preseleccion_rwa.csv` |
| **3** | Descarga series RWA por red y por tipo de activo | `series_rwa*.csv`, `series_diarias_ext.csv`, `series_largas_ext.csv` |
| **3m** | Reconstruye las series RWA desde el maestro, sin consultar DeFiLlama | Los mismos archivos de series que el Job 3 |
| **4** | Suma las variables macro de FRED, arma el dataset final, corre Granger y modelos en ambas direcciones | `series_completas.csv`, `granger_completo.csv`, `metricas_completo.csv` |

Fuentes usadas en la adquisición: DeFiLlama, FRED, ECB Data Portal, CoinGecko y CoinMetrics. El Job 0 prueba además el acceso a RWA.xyz.

### Análisis y modelado

| Job | Qué hace | Salidas principales |
|---|---|---|
| **5** | Análisis exploratorio: descriptivos, estacionariedad (ADF), saltos y correlaciones | `eda_*.csv`, figuras |
| **6** | Relaciones temporales entre RWA y estrés financiero | `relaciones_rwa_*.csv`, figuras CCF |
| **7** | Modelos de línea de base (on-chain + RWA) | `metricas_modelos_final.csv`, `predicciones_final.csv` |
| **8** | Validación y reporte de la línea de base | `granger_final*.csv`, `backtesting_final.csv`, `reporte_linea_base_final.csv` |
| **9** | Modelo bidireccional RWA ↔ estrés | `granger_bidireccional*.csv`, `metricas_bidireccional.csv` |
| **10** | Random Forest y XGBoost sobre pares validados por Granger | `metricas_nolineal.csv`, `feature_importance_rf.csv` |
| **11** | Validación walk-forward de pares convergentes | `walkforward_*.csv`, `convergencia_niveles.csv` |
| **12** | RWA frente a su activo tradicional: sigue, es simultáneo o se adelanta | `rwa_vs_tradicional.csv`, `granger_rwa_tradicional.csv` |
| **13** | Curva de dilución: rendimiento efectivo de USDY según el momento de entrada | `curva_dilucion.csv` y figura |
| **14** | Event study: señales de tasa y TVL de Treasuries tokenizados | `event_study.csv`, `event_study_resumen.csv` |
| **15** | Aplica la regla de decisión de la hipótesis | `decision_hipotesis.csv` |
| **16** | Análisis exploratorio: episodios por régimen de tasas, permutación y sensibilidad | `regimenes_*.csv`, `permutacion_episodios.csv`, `sensibilidad_detector.csv`, `resumen_job16.csv` |
| **17** | Robustez: Treasury Bills con una canasta fija de protocolos | `canasta_fija_*` en `outputs_y_data/outputs/robustez/` |

**Orden:** ejecutá los Jobs como aparecen en el notebook. Varios usan salidas de los anteriores.

### Dataset final

`series_completas.csv` tiene una columna `fecha` y variables de estos grupos:

- **`macro_*`:** variables macro de FRED (14), más `macro_eeuu_stlfsi` y `macro_euro_ciss`.
- **`rwa_*`:** supply de USDY y TVL por red y por tipo de activo, más el total global.

En el snapshot del 21/11/2026 son 24 variables, de 2021-04-22 a 2026-11-21, y **no figura `rwa_red_ripple`**. Las columnas disponibles dependen de los datos de cada corrida.

---

## 6. Herramientas (`tools/`)

### 6.1 `append_maestro_tvl.ipynb`

Combina corridas de distintas fechas en un historial por protocolo y red que **solo agrega fechas nuevas**. Así, un dato ya cargado no cambia si la fuente lo revisa o deja de publicar un protocolo.

**Entradas:** los JSON crudos de cada corrida en `outputs_y_data/<ejecucion>/data/raw/<fecha>/protocol_*.json`. Cada ejecución debe tener **una sola** carpeta de fecha. Una carpeta de ejecución es una copia de los datos de una corrida.

**Uso:**

1. Editá `EJECUCIONES` en orden cronológico (por defecto, `["ejecucion_3", "ejecucion_4"]`).
2. Ejecutá todas las celdas.

**Salidas** en `outputs_y_data/maestro/`:

| Archivo | Contenido |
|---|---|
| `maestro_tvl_largo.csv` | TVL por protocolo, red y fecha |
| `estado_protocolos.csv` | Estado de cada protocolo |
| `revisiones.csv` | Diferencias entre corridas en fechas comunes |
| `tb_agregado.csv` | Agregado de Treasury Bills de control |
| `manifest.json` | Hashes de entradas y salidas |

Si existe la serie de referencia de la primera ejecución, el notebook comprueba que el maestro la reproduzca.

**Para los Jobs 3m y 17** hacen falta el maestro y `ejecucion_4/data/processed/`:

- Job 3m: `series_largas_ext.csv` y `series_diarias_ext.csv`.
- Job 17: `series_largas_ext.csv` y `series_completas.csv`.

El Job 17 corre dos cortes (`2026-10-27` y todo el maestro). Si usás otras fechas, revisá `CORTES` y `FECHAS_INICIO`.

### 6.2 `comparar_corridas.py`

Guarda los CSV de `outputs/reportes/` y compara dos corridas por SHA-256. Indica `IGUAL` o `DIFIERE` y, si difieren, sugiere la causa probable: orden de filas, paralelismo o cambio de valores. No compara figuras ni modelos.

En Colab, indicando la carpeta donde está el script:

```python
import sys
sys.path.insert(0, "/content/ruta/al/repositorio/tools")
import comparar_corridas as cc

cc.guardar("corrida_1")     # después de la primera corrida
cc.guardar("corrida_2")     # después de la segunda
cc.comparar("corrida_1", "corrida_2")
```

Por consola, con `TESIS_BASE_DIR` apuntando a la carpeta de datos:

```bash
python tools/comparar_corridas.py guardar corrida_1
python tools/comparar_corridas.py comparar corrida_1 corrida_2
```

Las copias quedan en `outputs/corridas/<nombre>/`. `guardar` **borra y reemplaza** una corrida con el mismo nombre.

> La función `correr()` espera los Jobs como archivos separados `job_XX_*.py` en la misma carpeta que el script. Esos archivos no están en este repositorio, así que con el notebook usá solo `guardar` y `comparar`.

### 6.3 `comparar-datos.ipynb`

Comparación puntual entre la fase `fase_octubre` y `expansion`:

- resultados de Granger que habían sido significativos;
- diferencias en la serie de Treasury Bills;
- protocolos que perdieron o recuperaron su serie.

Requiere las dos carpetas de datos. Las rutas y la lista de protocolos revisados están fijas en el código: adaptalas antes de reutilizarlo.

### 6.4 `Exportar-Resultados.ipynb`

Genera y descarga `resultados.html` con las salidas de las celdas de código, sin el código ni las celdas markdown. No modifica el notebook original.

1. Guardá en Drive el notebook ya ejecutado.
2. Cambiá `notebook_path`: el valor actual apunta a `e2e-model-v1-2.ipynb`. Usá la ruta de tu `e2e-model-v3.ipynb`.
3. Ajustá `carpeta` si querés listar los notebooks disponibles.
4. Ejecutá las celdas.

---

## 7. Cosas a tener en cuenta

- **La salida guardada del Job 4 no es de la misma corrida que las de los Jobs 5 a 16.** La del Job 4 usa `prueba_oct` en modo `snapshot`. Las de los Jobs 5 a 16 usan `expansion` en modo `descarga`. No tomes las cifras del Job 4 guardado como la base de los análisis posteriores.
- **Las salidas guardadas no vienen de una única ejecución de principio a fin.** Las celdas se corrieron por separado.
- **Rutas fijas:** si tu estructura de Drive es distinta, usá las variables de entorno. Algunas herramientas tienen rutas escritas en el código.
- **Fuentes externas:** las APIs pueden cambiar, limitar pedidos o revisar datos. Para reproducir, usá datos congelados.
- **Reproducibilidad numérica:** versiones de librerías distintas o el paralelismo de los modelos pueden generar diferencias.

---

## 8. Problemas frecuentes

| Mensaje o síntoma | Qué revisar |
|---|---|
| `FRED_API_KEY no configurada` | El secreto, su nombre y el acceso del notebook. Para reproducir, usá `snapshot` |
| `SystemExit: 0` en Jobs 1 a 3 | Es el final normal en modo `snapshot` |
| `Job 3: ... no figura en manifest.json` | El manifest atribuye las series al 3m. Seguí la opción A y empezá en el Job 4 |
| `Job 4: falta ... series_largas_ext.csv` | Copialo a `data/processed/`, o ejecutá antes el Job 3 si estás descargando |
| `el hash ... no coincide` | La entrada no es la que registra el manifest. Restaurá el par correcto; no edites el hash a mano |
| Aparece `prueba_oct` en las rutas | Se ejecutó la celda `!cp`. Reiniciá el entorno y salteala |
| El Job 3m o 17 no encuentra archivos | Falta el maestro o `ejecucion_4` en `TESIS_OUT_DIR` |

---

## 9. Seguridad

No subas al repositorio API keys, archivos que las contengan (por ejemplo, `fred_apikey.txt` o `.env`) ni credenciales de Google. Antes de publicar cambios, revisá también las salidas guardadas de los notebooks.
