# Flujo ICA para limpieza de artefactos EEG

Este documento explica el flujo implementado en
[`ica_preprocessing.py`](./ica_preprocessing.py).

El objetivo del flujo es estimar una descomposicion ICA sobre los canales EEG,
identificar componentes asociados a artefactos oculares y musculares, registrar
las decisiones en un JSON editable y aplicar la reconstruccion de la señal sin
los componentes excluidos. La aplicacion posterior en scripts de analisis se
hace con [`ICAApplicator`](../../src/handwriting_analysis/ica/ica_apply.py).

Referencias usadas como base:

- Tutorial actual de MNE: [Repairing artifacts with ICA](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html).
- Ejemplo de MNE para musculo: [Removing muscle ICA components](https://mne.tools/stable/auto_examples/preprocessing/muscle_ica.html).
- Tutorial historico de MNE 0.14: [Artifact Correction with ICA](https://www.nmr.mgh.harvard.edu/mne/0.14/auto_tutorials/plot_artifacts_correction_ica.html).

## Idea general

ICA, o Independent Component Analysis, intenta separar las señales registradas en
fuentes latentes estadisticamente independientes. En EEG esto es util porque los
canales de cuero cabelludo mezclan actividad cerebral con actividad no cerebral:
parpadeos, movimientos oculares, tension muscular, ruido de red y otros
artefactos. El procedimiento de MNE ajusta una matriz de desmezcla, permite
marcar componentes como artefactos y luego reconstruye la señal excluyendo esos
componentes.

En este proyecto ICA se usa como herramienta de reparacion, no como metodo de
analisis final. La señal se transforma al espacio ICA, se anulan los componentes
marcados y se proyecta de vuelta al espacio de sensores. Por eso la decision mas
importante no es solo ajustar ICA, sino revisar cuidadosamente que componentes
se eliminan.

## Archivos del flujo

- [`ica_preprocessing.py`](./ica_preprocessing.py): script interactivo principal.
  Carga un registro HDF5, preprocesa una copia de la señal, ajusta ICA, detecta
  componentes candidatos, genera graficas, guarda el modelo `.fif` y escribe el
  JSON de resultados.
- [`ica_results_template.json`](../../analysis/ica_results_template.json):
  estructura base del JSON BIDS-like que registra parametros, canales malos,
  componentes detectados y componentes definitivos a excluir.
- [`ica_apply.py`](../../src/handwriting_analysis/ica/ica_apply.py): clase
  reutilizable `ICAApplicator`, usada por los analisis posteriores para aplicar
  una solucion ICA ya registrada.

## Parametros de entrada

Al inicio de `ica_preprocessing.py` se define que registro se va a procesar:

```python
sub  = "02"
ses  = "02"
task = "ejecutada"
run  = "06"
```

Con esos valores se construyen nombres BIDS-like:

- `sub-02_ses-02_task-ejecutada_run-06_eeg.hdf5`: señal EEG/EMG/EOG original.
- `sub-02_ses-02_task-ejecutada_run-06_ica.json`: resumen editable del analisis
  ICA.
- `sub-02_ses-02_task-ejecutada_run-06_ica.fif`: solucion ICA guardada por MNE.

Las rutas principales son:

- `path = D:\dataset\sub-{sub}\ses-{ses}`: carpeta de datos del sujeto/sesion.
- `montage_path = .\analysis\ghiamp_montage.sfp`: montaje con nombres y posiciones de
  los canales EEG.
- `template_path = .\analysis\ica_results_template.json`: plantilla del JSON.
- `output_path = path`: destino del JSON y del archivo `.fif`.

Hay que ajustar estas rutas si el proyecto se corre desde otro directorio o si
los datos no estan en `D:\dataset`.

## Carga y armado del objeto MNE

El script lee el HDF5 con `GHiampDataManager`:

```python
gmanager = GHiampDataManager(os.path.join(path, ghiamp_file), normalize_time=True)
raw_data = gmanager.raw_data.swapaxes(1, 0)
sfreq = gmanager.sample_rate
```

`GHiampDataManager.raw_data` queda transpuesto con `swapaxes(1, 0)` para obtener
la forma esperada por MNE: `canales x muestras`.

Luego se construye la informacion de canales:

- 64 canales EEG tomados del archivo `.sfp`.
- 1 canal EMG llamado `EMG1`.
- 2 canales EOG llamados `EOG1` y `EOG2`.

```python
ch_names = eeg_ch_names + ["EMG1"] + ["EOG1", "EOG2"]
ch_types = ["eeg"] * 64 + ["emg"] + ["eog"] * 2
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw_signal = mne.io.RawArray(raw_data, info)
```

El montaje se aplica despues con:

```python
montage = mne.channels.read_custom_montage(montage_path)
raw_signal.set_montage(montage, on_missing="ignore")
```

`on_missing="ignore"` es necesario porque el montaje contiene posiciones de EEG,
pero no necesariamente posiciones para `EMG1`, `EOG1` y `EOG2`.

## Preprocesamiento usado para ICA

El script no ajusta ICA directamente sobre `raw_signal`, sino sobre una copia:

```python
filt_raw = raw_signal.copy()
```

Esto separa la señal original cargada de la señal preparada para ICA. La copia se
filtra asi:

- EEG: pasa-alto `1 Hz`, sin corte superior (`h_freq=None`).
- EOG: pasa-alto `1 Hz`, sin corte superior (`h_freq=None`).
- EMG: pasa-alto `1 Hz`, sin corte superior (`h_freq=None`).
- Todos los canales: notch en `50 Hz`.

```python
filt_raw.filter(l_freq=1.0, h_freq=None, picks="eeg", fir_design="firwin")
filt_raw.filter(l_freq=1.0, h_freq=None, picks="eog", fir_design="firwin")
filt_raw.filter(l_freq=1.0, h_freq=None, picks="emg", fir_design="firwin")
filt_raw.notch_filter([50])
```

El corte inferior de `1 Hz` sigue la recomendacion habitual de MNE para ICA:
quitar derivas lentas mejora la estabilidad de la descomposicion. En este flujo
no se aplica corte superior antes de ICA; el notch en `50 Hz` queda como
tratamiento especifico de ruido de red. Esto no significa que todos los analisis
posteriores deban usar solo este filtro; por ejemplo, ERP o TFR pueden aplicar
filtros propios despues de la limpieza.

### Filtrado usado solo para graficas ICA

El entrenamiento y la deteccion automatica usan `filt_raw` con pasa-altos en
`1 Hz` y sin corte superior. Para que las visualizaciones sean mas legibles, el
script crea copias filtradas solo para graficar:

```python
plot_filter_l_freq = 1.0
plot_filter_h_freq = 40.0
```

La funcion `_make_ica_plot_raw(raw)` hace `raw.copy()` y filtra los canales
presentes (`eeg`, `eog`, `emg`) entre `1-40 Hz`. Esta copia se usa en
`plot_properties`, `plot_sources`, `plot_overlay`, PSD y senal scrollable. No se
usa para `ica.fit()`, `find_bads_eog()` ni `find_bads_muscle()`, por lo que no
cambia la solucion ICA ni los componentes detectados.

El overlay ademas puede acotar automaticamente el eje Y para evitar que outliers
grandes oculten la senal util:

```python
overlay_auto_ylim = True
overlay_ylim_percentiles = (1, 99)
overlay_ylim_margin = 0.10
```

Ese acotado solo modifica la escala visual de Matplotlib; no recorta ni altera
los datos.

Despues se marcan canales malos conocidos:

```python
bad_channels_known = ["F10"]
filt_raw.info["bads"] = bad_channels_known
```

Los canales en `info["bads"]` son ignorados por MNE cuando se seleccionan picks
EEG para ajustar ICA. Finalmente se configura referencia promedio para EEG:

```python
filt_raw.set_eeg_reference("average", projection=True)
filt_raw.apply_proj()
```

La referencia promedio cambia el espacio de los canales EEG antes del ajuste. La
proyeccion se aplica explicitamente para que ICA se ajuste sobre datos ya
referenciados.

## Ajuste ICA

La configuracion actual es:

```python
n_components = 30
ica_method = "fastica"
ica_random_state = 97
ica_max_iter = "auto"
```

Se crea el objeto MNE:

```python
ica = ICA(
    n_components=n_components,
    method=ica_method,
    max_iter=ica_max_iter,
    random_state=ica_random_state,
)
```

Y se ajusta solo con canales EEG:

```python
ica.fit(filt_raw, picks="eeg")
```

Esta es una decision importante: ICA aprende la mezcla sobre EEG, no sobre EMG ni
EOG. Los canales EOG se usan despues como referencia para detectar componentes
oculares, pero no forman parte del ajuste. `EMG1` tampoco forma parte del ajuste.

`n_components=30` significa que el algoritmo ICA recibe los 30 componentes PCA
de mayor varianza y estima 30 componentes independientes. Como hay 64 canales
EEG, esto es una reduccion dimensional deliberada: se busca capturar componentes
dominantes, especialmente artefactos grandes, sin modelar todos los grados de
libertad posibles. Además de esto, se podría usar un número entre `0` y `1` para indicar cuánta de la varianza acumulada se pretende tomar luego de que el algoritmo aplica PCA.

El `random_state=97` hace que la descomposicion sea reproducible para el mismo
registro, mismos parametros y misma version compatible de dependencias. Esto es
importante porque FastICA puede cambiar el orden o la forma de los componentes
si cambia la inicializacion aleatoria.

Luego el modelo se guarda:

```python
ica.save(ica_fif_path, overwrite=True)
```

El `.fif` conserva la solucion ICA de MNE y permite aplicarla despues sin
reajustar, por ejemplo desde `ICAApplicator.apply_to_raw`.

## Deteccion automatica de artefactos

La deteccion automatica genera candidatos. No debe tratarse como decision final
sin inspeccion visual.

### Componentes EOG

El script detecta componentes relacionados con movimientos oculares usando:

```python
eog_indices, eog_scores = ica.find_bads_eog(
    filt_raw, ch_name=["EOG1", "EOG2"], threshold=3.0
)
```

`find_bads_eog` compara las series temporales de los componentes ICA contra los
canales EOG filtrados. En MNE, la deteccion se basa en correlacion con la señal
EOG y umbralado por score. Si un componente ICA se parece temporalmente a
parpadeos o movimientos oculares, queda marcado como candidato.

Si `eog_ref_path` apunta a una grabacion de referencia, el script carga ese HDF5
con `_load_and_filter_ref()` y repite `find_bads_eog`. Esto permite usar una
tarea especifica de movimientos oculares intencionales para mejorar la
identificacion. Los indices detectados en el registro experimental y en la
referencia se combinan con union de conjuntos.

### Componentes musculares

La deteccion muscular se hace con:

```python
muscle_indices, muscle_scores = ica.find_bads_muscle(filt_raw)
```

`find_bads_muscle` no usa directamente el canal `EMG1`. MNE identifica
componentes musculares a partir de caracteristicas de los componentes ICA, como
contenido de alta frecuencia y patron espacial. Esto esta alineado con el ejemplo
de MNE para artefactos musculares, donde los componentes musculares suelen tener:

- actividad temporal con aspecto espiculado o de alta frecuencia;
- espectro con energia relativamente alta en frecuencias medias/altas;
- topografias focales, perifericas o poco suaves, a menudo cerca de zonas
  temporales o bordes del montaje.

Si `emg_ref_path` apunta a una grabacion de referencia, el script la carga y
vuelve a correr `find_bads_muscle`. Aunque se llame referencia EMG, la funcion
auxiliar descarta `EMG1` antes de devolver el `Raw` filtrado, por lo que la
deteccion sigue siendo sobre los componentes proyectados en EEG.

### Componentes conservados

La variable:

```python
components_to_keep = []
```

permite rescatar componentes que fueron detectados automaticamente pero que, al
revisarlos, parecen actividad cerebral. El script calcula:

```python
auto_detected = sorted(set(all_eog_indices) | set(all_muscle_indices))
auto_excluded = sorted(set(auto_detected) - set(components_to_keep))
ica.exclude = auto_excluded
```

Por defecto, todo candidato EOG o muscular queda en `auto_excluded`. Si se quiere
conservar alguno, se agrega su indice a `components_to_keep` antes de correr el
script.

## Inspeccion visual

Cuando `show_figs = True`, el script abre varias visualizaciones. La revision
manual deberia mirar todas, porque cada una responde una pregunta distinta.
Antes de llamar a las funciones graficas que dependen de la serie temporal, se
crea una copia de inspeccion:

```python
filt_raw_plot = _make_ica_plot_raw(filt_raw)
```

Esto mantiene separado el `Raw` usado para entrenar ICA (`filt_raw`) del `Raw`
usado para inspeccion visual (`filt_raw_plot`, filtrado `1-40 Hz`).

### Topomapas de componentes

```python
ica.plot_components(picks=range(n_components), title="Topomapas de componentes ICA")
```

Muestra la distribucion espacial de cada componente ICA sobre el cuero cabelludo.
En general:

- componentes oculares suelen verse frontales, simetricos o con patron
  horizontal/vertical segun el movimiento;
- componentes musculares suelen ser mas focales, perifericos o temporales;
- componentes cerebrales suelen tener topografias mas suaves y plausibles.

No alcanza con el topomapa para decidir, pero ayuda a detectar componentes
sospechosos.

### Scores EOG y musculares

```python
ica.plot_scores(eog_scores, exclude=eog_indices, ...)
ica.plot_scores(muscle_scores, exclude=muscle_indices, ...)
```

Estos graficos muestran que tan fuerte fue el match entre cada componente y el
criterio automatico. Sirven para distinguir componentes claramente extremos de
componentes apenas por encima del umbral.

### Propiedades detalladas

```python
ica.plot_properties(filt_raw_plot, picks=auto_excluded)
```

Para cada componente excluido muestra informacion combinada: topografia,
serie/epochs, espectro y otras vistas diagnosticas. Es una de las revisiones mas
importantes antes de confirmar la exclusion.

### Series temporales de fuentes ICA

```python
ica.plot_sources(filt_raw_plot, title="Series de tiempo de componentes ICA")
```

Permite inspeccionar la activacion temporal de cada componente. Es util para ver
si un componente coincide con parpadeos, movimientos oculares, tension muscular o
eventos aislados. En modo interactivo, MNE permite marcar o desmarcar
componentes desde esta vista.

### Overlay antes/despues

```python
fig_overlay_auto = ica.plot_overlay(filt_raw_plot, exclude=auto_excluded, ...)
_clip_overlay_ylim(fig_overlay_auto, ...)
```

Compara la señal original contra la reconstruida excluyendo los componentes
marcados. Debe mostrar reduccion del artefacto sin deformar de forma excesiva la
señal EEG. Si el overlay cambia demasiado zonas sin artefacto, conviene revisar
la lista de exclusiones. El helper `_clip_overlay_ylim()` solo cambia los
limites del eje Y para mejorar la lectura cuando hay valores extremos.

## JSON de resultados

El script carga la plantilla:

```python
with open(template_path, "r", encoding="utf-8") as f:
    result = json.load(f)
```

Luego completa campos y escribe `sub-..._ica.json`. Las secciones principales
son:

- `metadata`: sujeto, sesion, tarea, run, fecha y archivo fuente.
- `preprocessing`: filtros aplicados, notch, referencia y diseno FIR.
- `bad_channels`: canales malos automaticos/conocidos y canales agregados
  manualmente.
- `ica_settings`: parametros del ajuste ICA.
- `ref_paths`: rutas de grabaciones de referencia EOG/EMG, si se usaron.
- `auto_detected_components`: indices y scores EOG/musculares.
- `components_to_exclude`: componentes detectados, conservados, manuales y lista
  final.
- `ica_file`: nombre del `.fif` guardado.
- `ica_applied`: bandera que indica si la limpieza fue aplicada desde este flujo.
- `notes`: campo libre para documentar decisiones.

La seccion mas importante para la revision manual es:

```json
"components_to_exclude": {
  "auto_detected": [],
  "kept_from_auto": [],
  "auto": [],
  "manual": [],
  "final": []
}
```

Uso esperado:

- `auto_detected`: union de candidatos EOG y musculares.
- `kept_from_auto`: candidatos automaticos que se decidio conservar.
- `auto`: candidatos automaticos que quedan para excluir.
- `manual`: componentes agregados por inspeccion visual.
- `final`: lista definitiva que debe aplicarse.

En una revision manual, `final` deberia ser la lista de componentes que realmente
se van a remover. Si se agregan componentes en `manual`, tambien deben aparecer
en `final`. Si se decide rescatar un componente automatico, debe salir de
`final`.

## Aplicacion dentro de `ica_preprocessing.py`

El comentario inicial del script plantea dos pasadas:

1. `apply_ica=False`: ajustar ICA, detectar artefactos, revisar graficas y
   guardar JSON.
2. Editar el JSON manualmente.
3. `apply_ica=True`: aplicar la lista final y graficar antes/despues.

La seccion de aplicacion lee:

```python
final_components = ica_params["components_to_exclude"]["final"]
ica.exclude = final_components
raw_clean = filt_raw.copy()
ica.apply(raw_clean)
```

`ica.apply()` modifica el objeto `Raw` recibido. Por eso el script hace una copia
antes de aplicar, preservando `filt_raw` para comparaciones.

Antes de graficar, el script crea copias de inspeccion (`raw_before_plot` y
`raw_after_plot`) con `_make_ica_plot_raw()`. Ambas quedan filtradas entre
`1-40 Hz` para facilitar la lectura visual. Este filtro es solo para las
graficas y la PSD comparativa; no cambia la solucion ICA ya ajustada ni la lista
de componentes excluidos.

Despues genera:

- overlay MNE antes/despues;
- propiedades de los componentes finales;
- comparacion de PSD media EEG antes/despues;
- señal limpia scrollable.

Finalmente actualiza:

```json
"ica_applied": true
```

y agrega:

```json
"components_to_exclude": {
  "applied": [...]
}
```

### Advertencia sobre el estado actual del script

Actualmente `ica_preprocessing.py` siempre vuelve a ajustar ICA y siempre vuelve
a escribir el JSON antes de entrar al bloque `if apply_ica:`. Por lo tanto, si se
edita manualmente el JSON y luego se corre el mismo script con `apply_ica=True`,
existe riesgo de sobrescribir las ediciones manuales antes de leerlas.

Tambien se calcula:

```python
bad_ch_all = (
    ica_params["bad_channels"]["auto_detected"]
    + ica_params["bad_channels"]["manual"]
)
```

pero dentro de esa segunda seccion del script los canales manuales se imprimen y
no se reaplican al `filt_raw`, porque el preprocesamiento ya ocurrio antes con
`bad_channels_known`.

Por eso, para un flujo manual estricto, hay dos opciones mas seguras:

- editar `components_to_keep` y `bad_channels_known` en el script antes de volver
  a correrlo; o
- usar el JSON y el `.fif` desde `ICAApplicator.apply_to_raw`, evitando reajustar
  y sobrescribir la decision manual.

## Aplicacion posterior con `ICAApplicator`

`ICAApplicator` permite reutilizar la solucion ICA en otros scripts.

Hay dos modos principales:

### `load_and_fit()` + `apply()`

```python
cleaner = ICAApplicator(json_path)
cleaner.load_and_fit(hdf5_path)
raw_clean = cleaner.apply()
cleaner.plot_comparison()
```

Este camino carga el HDF5, preprocesa con los parametros guardados en el JSON,
reajusta ICA con el mismo `random_state` y aplica los componentes indicados por
`components_to_exclude.final`. Es reproducible en intencion, pero no es lo mismo
que cargar directamente el `.fif`: vuelve a estimar el modelo.

### `apply_to_raw()`

```python
cleaner = ICAApplicator(ica_json_path)
cleaner.apply_to_raw(raw_signal)
```

Este es el modo usado en `scripts/analysis/erp_analysis.py` y
`scripts/analysis/analisis_piloto.py`. Carga el archivo `.fif` indicado en
`ica_file`, toma la lista `components_to_exclude.final` del JSON y aplica ICA
in-place sobre un `Raw` ya cargado, anotado o recortado.

Antes de aplicar ICA, remueve del `Raw` los canales listados en
`bad_channels.auto_detected` y `bad_channels.manual` cuando esos canales no
forman parte del modelo ICA guardado. Este es el caso esperado para canales como
`F10` marcados en `bad_channels_known` durante `ica_preprocessing.py`: quedan
fuera de la senal antes de ICA, antes de los filtros de analisis y antes de crear
epocas.

Si un canal marcado manualmente si forma parte del modelo ICA, el aplicador emite
una advertencia y lo remueve inmediatamente despues de `ica.apply()` para
mantener la compatibilidad con MNE. Ese caso no es el flujo ideal: si se descubre
que un canal es malo y todavia estaba dentro del modelo ICA, lo mas consistente
es regenerar el ICA declarandolo en `bad_channels_known` antes del ajuste.

Este modo no reajusta ICA. Por eso es el mas consistente cuando se quiere aplicar
exactamente la solucion revisada y guardada durante el preprocesamiento.

## Criterios practicos para decidir exclusiones

Antes de agregar un componente a `final`, conviene revisar:

- Topomapa: patron frontal para EOG, periferico/focal para musculo o patron
  cerebral plausible.
- Serie temporal: parpadeos grandes, movimientos lentos, rafagas musculares o
  actividad ligada a eventos.
- Espectro: exceso de alta frecuencia para musculo; contenido lento fuerte para
  ocular.
- Scores: si el componente esta apenas sobre el umbral, requiere mas cautela.
- Overlay: la limpieza debe reducir artefactos sin distorsionar segmentos sanos.
- PSD: una caida excesiva de potencia en bandas de interes puede indicar que se
  removio actividad cerebral.

Regla practica: la deteccion automatica propone candidatos; la lista `final` es
una decision de analisis.

## Recomendaciones de uso

1. Ajustar `sub`, `ses`, `task`, `run`, rutas y `bad_channels_known`.
2. Correr el script con figuras activadas.
3. Revisar topomapas, scores, propiedades, fuentes y overlay.
4. Registrar decisiones en el JSON: `bad_channels.manual`,
   `components_to_exclude.manual`, `components_to_exclude.final` y `notes`.
5. Verificar que `final` contenga exactamente los componentes a remover.
6. Aplicar la solucion preferentemente con `ICAApplicator.apply_to_raw` en los
   scripts de analisis posteriores, o revisar la advertencia anterior si se usa
   `apply_ica=True` dentro del mismo script.

No conviene eliminar un componente solo porque fue detectado automaticamente. Si
su topografia, serie temporal o espectro son compatibles con actividad cerebral,
debe conservarse o al menos documentarse la duda en `notes`.
