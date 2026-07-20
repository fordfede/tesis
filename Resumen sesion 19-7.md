# 1. Problemas encontrados

### **XRP**

Al intentar extraer XRP descubrimos que el RPC público no indexa por
fecha, solo por índice de ledger. Como XRPL cierra un ledger cada ~4
segundos, un año son ~7-8 millones de ledgers, y cada llamada devuelve
UN ledger. Barrer el año = millones de requests secuenciales contra un
nodo gratuito que rate-limitea antes de terminar. No es cuestión de
subir un parámetro: el modelo de acceso no lo permite.

>**Opciones que evaluamos para XRP:**
Nodo full-history propio (Clio/rippled): descargás la cadena entera y consultás tu base local. Viable pero requiere TB de disco + días de sync. Descartado por infra.
WebSocket en vivo: escuchás la red en tiempo real. Descartado porque requiere un proceso corriendo 24/7 sin cortes, incompatible con Google Colab (que desconecta por inactividad) y solo construye histórico hacia adelante, no recupera enero-hoy.
BigQuery (dataset público): lo investigamos a fondo (ver abajo).
Proveedores pagos (Bithomp, XRPScan): investigados.

>**Investigación de BigQuery:**
Verificamos el dataset público `xrpledgerdata.fullhistory.transactions`
(mantenido por Wietse Wind). Confirmamos vía query de esquema que tiene
las columnas necesarias: hash, Account, Destination, AmountXRP, Fee,
LedgerIndex, y el campo Memos (REPEATED RECORD, que requiere UNNEST).
Confirmamos que NO tiene columna de fecha (solo LedgerIndex), así que
el corte YTD se hace mapeando fecha→ledger. Anclamos el ledger inicial
de 2026 usando un ledger real de referencia (101.224.901 cerrado el
30/12/2025) → XRPL_LEDGER_INICIO ≈ 101.256.700.
PERO al correr la query real dio 0 filas. Diagnóstico: el MAX(LedgerIndex)
del dataset era 86.447.559 — el backfill del dataset está FRENADO antes
de 2026. El dataset no cubre nuestra ventana. BigQuery descartado.

>**Investigación de XRPScan:**
Verificamos su API. Confirmamos el endpoint account_tx
(/api/v1/account/{ACCOUNT}/transactions) que trae hash, Account,
Destination, Amount, Fee, Memos, ledger_index y date en ISO (resuelve
el hueco de timestamp de XRP). Cobertura desde génesis → incluye 2026.
Licencia Creative Commons NonCommercial → válida para tesis.
LIMITACIÓN: es por cuenta, no por fecha a nivel red. Necesita whitelist
de cuentas.


### **HBAR**
Al revisar los datos crudos de Hedera detectamos que la extracción trajo
solo 1852 transacciones. Analizándolas con código confirmamos que
abarcaban apenas 0,24 horas (15 minutos del día de la corrida), NO los
6 meses YTD. Además, el 87% (1605 txs) venía de UNA sola cuenta
(0.0.3229), que resultó ser el bot de monitoreo de la red
(hiero-mirror-monitor).

- **Diagnóstico del bug**: el techo MAX_PAGINAS × PAGINA_HEDERA (50×100=5000)
es ínfimo frente al volumen de la red; el barrido desc arranca de ahora
y choca el techo tras unos minutos, sin llegar a retroceder 6 meses.
- **Conclusion general**: barrer toda la red por fecha no escala en NINGUNA
red de alto volumen contra una API pública. El mismo muro de XRP.
#
---
#
# 2. GIRO METODOLÓGICO

Ante esto, cambiamos el enfoque de "barrer toda la red y después filtrar"
a "filtrar desde el origen por cuentas". Diseñamos un método de tres
listas:

- **BLANCA**: cuentas institucionales por identidad (directorio público). Filtro 1 (identidad) se aplica sin datos; filtros 2 (uso de memo estructurado) y 3 (actividad sostenida en 2026) se aplican después sobre las transacciones.
- **GRIS**: cuentas sin identidad etiquetada pero con comportamiento institucional (volumen + memo estructurado + diversidad de contrapartes). Se construye por semilla + expansión por grafo.
- **NEGRA**: exclusiones explícitas (bots, rewards, cuentas de sistema).

Ventaja: este método unifica las cuatro redes bajo la misma lógica
(extracción por cuenta, que SÍ escala) y de paso resuelve el muro de
XRP (que también es por cuenta).

### BÚSQUEDA DE FUENTES DE IDENTIDAD POR RED
Investigamos qué directorio de cuentas etiquetadas tiene cada red:

> **STELLAR**: encontramos Stellar.Expert Open Directory API (pública, gratuita). Expone tags que mapean 1:1 con identidad institucional: anchor, issuer, exchange, custodian, y tags de blacklist (malicious, unsafe, airdrop). Corrimos la función: 601 cuentas. Al ver el desglose por tags detectamos contaminación (cuentas con tag institucional PERO también unsafe/malicious). Aplicamos limpieza (excluir cualquier tag de blacklist) → ~450 cuentas limpias. También evaluamos tags dudosos (defi, sdf, wallet): defi trae 346 cuentas, pero son pools/contratos que NO usan memo con propósito ISO (codifican en la operación), así que los separamos como lista SECUNDARIA para análisis de liquidez, no en la whitelist principal.

> **XRP**: encontramos el endpoint well-known accounts de XRPScan (el mayor proveedor de identificación de cuentas XRPL, 550+ servicios verificados). Corrimos la función: 2772 cuentas. PERO solo tiene name/domain/verified, SIN campo de categoría (no distingue exchange de bot). Filtramos por verified=True → 1182 cuentas. Queda pendiente curar por name (hay casas de apuestas, PSPs mezclados).

>**ALGORAND**: NO tiene directorio de cuentas etiquetadas. Investigamos Allo.info y Pera: Pera verifica ASAs (tokens), no cuentas. La única fuente institucional fue la página de transparencia de la Algorand Foundation, que publica sus account addresses oficiales. Extrajimos: ~63 cuentas de Treasury + 2 de Market Operations (custodios BitGo y
Fireblocks) como semilla; Fee Sink y ~10 pools de Governance/Rewards a
blacklist. Corrimos la extracción por cuenta sobre la semilla en la ventana 2026: BitGo: 0 txs (dormida en 2026). Fireblocks: 8 txs. Total de las 65 cuentas: 50.253 txs, PERO el 99,5% (49.996) veníade UNA cuenta (MKZIWVBD...). Las otras 64 sumaban ~257 txs, la mayoría 0.
Inspeccionamos los memos de esa cuenta gigante: todos "ProposerPayout for Round XXXX" → es el distribuidor de pagos a proposers del consenso. Infraestructura de red, ruido puro.
Conclusión Algorand: la semilla de identidad no sirve; la actividad real no vive en cuentas identificables.

>**HEDERA**: NO tiene directorio útil. Los exchanges no publican sus account IDs (usan una cuenta + memo de destino por usuario, según guías de soporte). La única herramienta de identidad (HederaIntel) cobra por consulta y es de a una cuenta. Sin fuente de semillas por identidad.

### OTRAS REDES ISO 20022 EVALUADAS
Ante la caída de Algorand/Hedera, buscamos alternativas dentro del grupo ISO 20022 compliant (XRP, Stellar, Hedera, Algorand, XDC, IOTA, Quant, Cardano) más players del ecosistema:
- **METALLICUS**: confirmamos que ES ISO 20022 compliant
(lo declara su web) y que tiene explorer público (Blockscout en C-Chain
y L2). PERO al ver el explorer: volumen ínfimo (C-Chain ~25k txs
totales, X-Chain ~4,5k totales; es una red mayormente de staking).
Además arquitectura multi-chain heterogénea (P/X/C-Chain + L2) sin campo
memo uniforme. Descartada: sin volumen para estadística ni memo
clasificable.
- **XDC**: es EVM (fork de go-ethereum). NO tiene campo memo/note nativo; el
equivalente es el input/calldata (vacío en pagos, ABI-encoded en
contratos). El corte por fecha es indirecto (Blockscout pagina por
bloque). Usarla exigiría decodificar calldata con el ABI de cada
contrato → otro modelo de extracción. Descartada por incompatibilidad
con el enfoque memo.
#
---
#
# 3. CONCLUSIÓN POR RED 


### NUCLEO
>**STELLAR:**
Es la única con directorio de identidad limpio y consultable (Stellar.Expert). Tiene volumen, memo de texto clasificable e identidad institucional. Whitelist de ~450 cuentas ya construida y guardada (wl_stellar.csv) + secundaria DeFi ~346 (wl_stellar_defi.csv).

>**XRP:** el método por cuenta (XRPScan/account_tx) resuelve su muro de extracción, tiene identidad (well-known accounts) con date real y memos. Whitelist de 1182 cuentas verificadas guardada (wl_xrp.csv). 

*Pendiente: curar xrp por name (hay apuestas/PSP).*

### DESCARTADAS DEL NUCLEO
- **ALGORAND**: probamos empíricamente su única semilla de identidad (Foundation). 63 de 65 cuentas dormidas o casi en 2026; la única con volumen es infraestructura de consenso (ProposerPayout), ruido puro. La actividad institucional no vive en cuentas identificables. No hay directorio alternativo.

- **HEDERA**: sin fuente de identidad pública (exchanges no publican account IDs). La muestra cruda estaba dominada por bots de monitoreo. Sin semilla posible por identidad.

- **METALLICUS**: ISO 20022 sí, pero red diminuta (volumen insuficiente para estadística) y sin campo memo uniforme (multi-chain EVM heterogénea).

- **XDC**: EVM, sin memo de texto; el propósito vive en calldata, lo que exige un modelo de extracción/clasificación completamente distinto.
#
---
#
# 4. ALTERNATIVA DE EXPANSIÓN

### ONDO

Ondo es un actor de RWA (Real World Assets) que tokeniza bonos del Tesoro de EE.UU. (OUSG, USDY).

Por qué es conceptualmente valioso: los flujos de mint/redención de treasuries tokenizados son demanda institucional de activos de refugio — señal directa y anticipatoria de estrés de liquidez. Encaja con la hipótesis del proyecto mejor que forzar más blockchains de pago.

>**Reparos**
Es EVM (Ethereum/L2s): no hay memo de texto. El propósito vive en EVENTOS de smart contracts (mint, redeem, transfer). Requiere un módulo de extracción distinto al memo-based del pipeline actual. No usa ISO 20022, entra en tensión con el título de la tesis.Cambia el objeto de análisis de clasificar memos a interpretar eventos de contratos.