"""
Análisis de señales para pruebas pilotos.

Se replica https://mne.tools/stable/auto_tutorials/time-freq/20_sensors_time_frequency.html
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyhwr.managers import GHiampDataManager, LSLDataManager
import mne

### Cargando datos
sub = "02"
ses = "01"
task = "imaginada"
run = "03"
subject_folder = "s2"
type_signal = "eeg"
path = f"D:\\dataset\\{subject_folder}"
show_figs = True
save_figs = False
baseline = (-1,0)

lsl_file = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_{type_signal}.xdf"
ghiamp_file = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_{type_signal}.hdf5"

gmanager = GHiampDataManager(os.path.join(path, ghiamp_file), normalize_time=True)
lsl_manager = LSLDataManager(os.path.join(path, lsl_file))

### Me quedo con los datos de la señal registrada
raw_data = gmanager.raw_data.swapaxes(1,0) #canales x muestras
raw_eeg = raw_data[:64,:] #desde 0 a 63
raw_emg = raw_data[[64],:]
raw_eog = raw_data[[65,66],:]

### ************** Obteniendo marcadores registrados por el g.HIAMP ***************
# print("Nombre de los marcadores:", gmanager.markers_info.keys())
# print("Tiempos de los marcadores:", gmanager.markers_info.values())
## cambio nombres de marcadores
gmanager.changeMarkersNames({1: "startRun", 2: "trialTablet", 3: "penDown", 4: "trialLaptop"})
t0_gtec = gmanager.markers_info["startRun"][0] #inicio de la ronda marcado por gRecorder usando el trigger de inicio de ronda

## Marcadores de trial de la tablet y de laptop. Se sugiere usar los marcadores de Tablet.
markers_info = gmanager.markers_info
trials_tablet = np.array(markers_info["trialTablet"])
trials_laptop = np.array(markers_info["trialLaptop"])

# pen_down solo existe en la tarea "ejecutada"; en "imaginada" no hay eventos de lápiz.
has_pen_down = "penDown" in markers_info and len(markers_info["penDown"]) > 0
pen_down = np.array(markers_info["penDown"]) if has_pen_down else np.array([])

### ************** Obteniendo etiquetas para cada trial ***************
##LSL
letras = [lsl_manager.trials_info["Tablet_Markers"][i]["letter"] for i in range(1,len(lsl_manager.trials_info["Tablet_Markers"])+1)]
start_time_tablet = lsl_manager.trials_info["Tablet_Markers"][1]["sessionStartTime"]/1000
##tiempos de inicio de restTime
rest_times = np.array(lsl_manager["Tablet_Markers","trialRestTime",:])/1000 - start_time_tablet
rest_times_relative_gtec = rest_times + t0_gtec #+ 3

##concateno trials_tablet, (pen_down si corresponde) y rest_times_relative_gtec y sorteo
arrays_markers = [trials_tablet, rest_times_relative_gtec]
if has_pen_down:
    arrays_markers.insert(1, pen_down)
times_markers = np.concatenate(arrays_markers)
times_markers.sort()

labels = []
for letra, rest in zip(letras, rest_times_relative_gtec):
    labels.append(letra)
    if has_pen_down:
        labels.append("pd")
    labels.append("rest")

# startRun es único y temporalmente anterior a todos los trials,
# por lo que se antepone directamente sin necesidad de re-sortear.
times_markers = np.concatenate(([t0_gtec], times_markers))
labels = ["startRun"] + labels

### ************** Info general ***************
sfreq = gmanager.sample_rate #frecuencia de muestreo del ampli
montage_df = pd.read_csv(".\\analysis\\ghiamp_montage.sfp", sep="\t", header=None)

# Los primeros 64 canales del SFP corresponden a EEG
# (se usa [:64] para descartar la fila vacía al final del archivo)
eeg_ch_names = list(montage_df[0])[:64]

# Nombres para los canales adicionales (índices 64, 65 y 66 del raw_data)
emg_ch_names = ["EMG1"]
eog_ch_names  = ["EOG1", "EOG2"]

# Lista completa de nombres y tipos para los 67 canales
ch_names  = eeg_ch_names + emg_ch_names + eog_ch_names
ch_types  = ["eeg"] * 64 + ["emg"] + ["eog"] * 2

# channels_to_remove = ["A1","A2"]
# ch_names = [ch for ch in ch_names if ch not in channels_to_remove]

### ************** Creando objeto MNE***************
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw_signal = mne.io.RawArray(raw_data, info)

# Leer el montage desde el archivo SFP y aplicarlo al objeto Raw.
# on_missing='ignore' evita errores para EMG1/EOG1/EOG2,
# que no tienen posición de electrodo en el montage.
montage = mne.channels.read_custom_montage(".\\analysis\\ghiamp_montage.sfp")
raw_signal.set_montage(montage, on_missing="ignore")
# raw_signal.plot_sensors(title="Montaje",show_names=True)

anotaciones = mne.Annotations(onset=times_markers,
                              duration=[0] * len(times_markers),  # fix: len() sobre el array
                              description=labels)

raw_signal.set_annotations(anotaciones)

# Crop: desde 4 s antes del primer trial (primera letra mostrada) hasta 2 s después
# del último marcador de la sesión (último "rest"). Descarta señal previa al inicio
# de la tarea y tiempo muerto al final, concentrando el análisis en la actividad útil.
tmin_crop = trials_tablet[0] - 4.0
tmax_crop = rest_times_relative_gtec[-1] + 2.0
raw_signal.crop(tmin=tmin_crop, tmax=tmax_crop)

print("Tipos de canal:", raw_signal.get_channel_types())

# Filtro pasa-banda 4-30 Hz SOLO sobre EEG.
# El EMG requiere un rango mucho mayor (ej. 20-500 Hz), así que lo excluimos
# para no destruir su contenido espectral.
raw_signal.filter(l_freq=4.0, h_freq=30.0, picks='eeg', fir_design='firwin')

# Filtro pasa-banda para EOG: los movimientos oculares son señales lentas,
# típicamente entre 0.5 y 15 Hz. El límite bajo elimina deriva DC;
# el límite alto recorta el ruido de alta frecuencia.
raw_signal.filter(l_freq=1.0, h_freq=15.0, picks='eog', fir_design='firwin')

# Filtro pasa-alto a 1 Hz para EMG: elimina deriva DC y artefactos de muy baja
# frecuencia (movimiento, respiración). Sin límite superior para preservar
# el contenido espectral muscular.
raw_signal.filter(l_freq=1.0, h_freq=None, picks='emg', fir_design='firwin')

# Filtro notch a 50 Hz (interferencia de red) para TODOS los canales
raw_signal.notch_filter([50])

# Escalas visuales diferenciadas por tipo de canal.
# EEG: ~30 uV; EOG: ~150 uV (potenciales oculares); EMG: ~300 uV (señal muscular)
scalings = {'eeg': 30, 'emg': 300, 'eog': 150}

# Colores diferenciados por tipo de canal
color = {'eeg': 'steelblue', 'emg': 'forestgreen', 'eog': 'darkorange'}

raw_signal.plot(scalings=scalings, color=color, duration=40)

# eeg_signal: copia de raw_signal con solo los 64 canales EEG.
# raw_signal se conserva intacto para cualquier analisis posterior de EMG/EOG.
# Las anotaciones se copian automaticamente junto con la señal.

### EEG SIGNAL
eeg_signal = raw_signal.copy().pick('eeg').drop_channels(['F7'])
# eeg_signal.pick(["FC3", "FC1", "FCz", "FC2", "FC4", "C3", "C1", "Cz", "C2", "C4", "CP3", "CP1", "CPz", "CP2", "CP4"])  # Descomentar para analizar solo canales centrales
# eeg_signal.plot(scalings=scalings, color=color, duration=50, start=242)
### ************** Creando epocas ***************

# Convertir anotaciones a eventos MNE (array [n_events x 3]: muestra | 0 | event_id)
events, event_id = mne.events_from_annotations(eeg_signal)

# Filtrar event_id para quedarse solo con los cues de letra.
# Se excluyen "startRun", "pd" y "rest", que no son el estimulo de interes.
marcadores_no_letra = {"startRun", "pd", "rest"}
event_id_letras = {k: v for k, v in event_id.items() if k not in marcadores_no_letra}

print("Eventos de letra encontrados:", event_id_letras)

# Crear epocas centradas en el cue de letra:
#   tmin = -1.0 s : 1 segundo previo al cue (pre-estimulo / baseline)
#   tmax = +4.0 s : 4 segundos posteriores al cue (actividad de escritura)
#
# baseline=(-1.0, 0.0): correccion de linea de base usando el segundo pre-cue,
#   elimina el offset DC y variaciones lentas no relacionadas con la tarea.
# preload=True: carga todas las epocas en memoria, necesario para analisis posteriores.
#
# reject: descarta epocas donde la amplitud pico a pico del EEG supera el umbral.

reject = {'eeg': 150}

epochs = mne.Epochs(
    eeg_signal,
    events,
    event_id=event_id_letras,
    tmin=-2.0,
    tmax=4.0,
    baseline=baseline,
    reject=reject,
    preload=True
)

print(epochs)
print(f"Epocas retenidas: {len(epochs)} / {len(epochs.drop_log)}")
# epochs.plot(scalings=scalings)
### ************** Analisis frecuencial (EEG, sin F7) ***************

# Cadena de identificacion reutilizada en todos los titulos de figuras.
run_info = f"Sub-{sub} | Ses-{ses} | Run-{run} | Tarea: {task}"

# Directorio de salida para las figuras del sujeto.
# Se crea solo si save_figs esta habilitado; exist_ok=True evita error si ya existe.
if save_figs:
    fig_dir = os.path.join("images", f"sub-{sub}", f"ses-{ses}", f"run-{run}", task)
    os.makedirs(fig_dir, exist_ok=True)

# Prefijo BIDS-like para los nombres de archivo de las figuras
fig_prefix = f"sub-{sub}_ses-{ses}_run-{run}_task-{task}"

# F7 ya fue excluido al construir eeg_signal; la PSD opera sobre los 63 canales restantes.
# Se usa .copy() para no alterar el objeto epochs original.
psd = epochs.copy().compute_psd(method='welch', fmin=4.0, fmax=30.0)

# psd.plot() no expone el parametro title; se captura la figura y se agrega suptitle.
fig_psd_ind = psd.plot(average=False, amplitude=True, show=False)
fig_psd_ind.set_size_inches(12, 6)
fig_psd_ind.suptitle(f"PSD por epoca — canales EEG (4-30 Hz)\n{run_info}")
if show_figs:
    fig_psd_ind.show()
if save_figs:
    fig_psd_ind.savefig(os.path.join(fig_dir, f"{fig_prefix}_psd_ind.png"), dpi=150, bbox_inches='tight')

fig_psd_avg = psd.plot(average=True, amplitude=True, show=False)
fig_psd_avg.set_size_inches(10, 5)
fig_psd_avg.suptitle(f"PSD promediada sobre epocas y canales EEG (4-30 Hz)\n{run_info}")
if show_figs:
    fig_psd_avg.show()
if save_figs:
    fig_psd_avg.savefig(os.path.join(fig_dir, f"{fig_prefix}_psd_avg.png"), dpi=150, bbox_inches='tight')

# plot_topomap requiere que cada banda este completamente contenida en el rango
# de la PSD computada (fmin=4, fmax=30).
# Se definen explicitamente solo las bandas disponibles:
#   Theta: 4-8 Hz   Alpha: 8-13 Hz   Beta: 13-30 Hz
bands = {
    'Theta (4-8 Hz)' : (4, 8),
    'Alpha (8-13 Hz)': (8, 13),
    'Beta (13-30 Hz)': (13, 30),
}

# size controla el tamaño de cada topomap individual en pulgadas (default=1).
fig_topo_psd = psd.plot_topomap(ch_type="eeg", normalize=False, contours=0, bands=bands, size=2,
                                show=False)

# Los ejes del colorbar se identifican por su ancho muy reducido (< 5% de la figura)
# y se les eliminan los ticks y sus etiquetas numericas.
for ax in fig_topo_psd.get_axes():
    if ax.get_position().width < 0.05:
        ax.yaxis.set_ticks([])

fig_topo_psd.set_size_inches(10, 5)
fig_topo_psd.suptitle(f"Distribución espacial de la PSD por banda (4-30 Hz)\n{run_info}")
if show_figs:
    fig_topo_psd.show()
if save_figs:
    fig_topo_psd.savefig(os.path.join(fig_dir, f"{fig_prefix}_psd_topo.png"), dpi=150, bbox_inches='tight')

### ************** Time-frequency analysis: power and inter-trial coherence ***************
# Replica de https://mne.tools/stable/auto_tutorials/time-freq/20_sensors_time_frequency.html#time-frequency-analysis-power-and-inter-trial-coherence
#
# Se calculan representaciones tiempo-frecuencia (TFR) usando wavelets de Morlet.
# Se obtienen simultaneamente la potencia promedio y la coherencia inter-trial (ITC).
#
# Parametros clave:
#   freqs   : frecuencias de interes en escala logaritmica (5-30 Hz, 20 bins)
#             rango acotado al filtro pasa-banda aplicado al EEG.
#   n_cycles: numero de ciclos por frecuencia = freqs / 2.0
#             valor bajo → mejor resolucion temporal, peor espectral (y viceversa).
#   decim   : decimacion temporal de la TFR para reducir uso de memoria.
#             si sfreq=512 Hz y decim=4 → resolucion temporal resultante ~128 Hz.
#   average : True → promedia la potencia sobre todas las epocas (TFR media).
#   return_itc: True → devuelve tambien el objeto ITC en el mismo calculo.

freqs   = np.logspace(*np.log10([5, 30]), num=20)  # 20 frecuencias log-espaciadas entre 5 y 30 Hz
n_cycles = freqs / 2.0                              # ciclos proporcionales a la frecuencia

power, itc = epochs.compute_tfr(
    method="morlet",
    freqs=freqs,
    n_cycles=n_cycles,
    average=True,
    return_itc=True,
    decim=4,
)

# ---- Inspeccion de POTENCIA ----

# Topomap interactivo: potencia media sobre todos los canales y tiempos.
# baseline=(-1.0, 0): correccion respecto al periodo pre-cue.
# mode="logratio": log(potencia_activa / potencia_baseline) — resalta cambios relativos.
ch_idx = power.ch_names.index("Cz")

fig_power_topo = power.plot_topo(
    baseline=baseline, mode="logratio",
    title=f"Potencia media (TFR) | {run_info}",
    show=False,
)
fig_power_topo.set_size_inches(10, 8)
if show_figs:
    fig_power_topo.show()
if save_figs:
    fig_power_topo.savefig(os.path.join(fig_dir, f"{fig_prefix}_tfr_potencia_topo.png"), dpi=150, bbox_inches='tight')

# TFR de un canal de interes: Cz (canal central, relevante para escritura).
# plot() devuelve una lista de figuras (una por canal); se toma la primera.
fig_power_ch = power.plot(
    picks=[ch_idx],
    baseline=baseline,
    mode="logratio",
    title=f"TFR potencia — {power.ch_names[ch_idx]} | {run_info}",
    show=False,
)[0]
fig_power_ch.set_size_inches(10, 5)
if show_figs:
    fig_power_ch.show()
if save_figs:
    fig_power_ch.savefig(os.path.join(fig_dir, f"{fig_prefix}_tfr_potencia_ch-{power.ch_names[ch_idx]}.png"), dpi=150, bbox_inches='tight')

# Topomapas por banda de frecuencia (solo bandas dentro del rango filtrado).
fig_power_bands, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
topomap_kw = dict(
    ch_type="eeg",
    tmin=0.0,
    tmax=4.0,
    baseline=baseline,
    mode="logratio",
    show=False,
)
plot_dict = dict(
    Alpha=dict(fmin=8, fmax=13),
    Beta =dict(fmin=13, fmax=30),
)
for ax, (title, fmin_fmax) in zip(axes, plot_dict.items()):
    power.plot_topomap(**fmin_fmax, axes=ax, **topomap_kw)
    ax.set_title(title)
fig_power_bands.suptitle(f"Topomapas de potencia por banda\n{run_info}")
if show_figs:
    fig_power_bands.show()
if save_figs:
    fig_power_bands.savefig(os.path.join(fig_dir, f"{fig_prefix}_tfr_potencia_bandas.png"), dpi=150, bbox_inches='tight')

# Joint plot: TFR + topomapas en instantes/frecuencias de interes.
# timefreqs: lista de (tiempo_s, frecuencia_Hz) a destacar con topomap.
fig_power_joint = power.plot_joint(
    baseline=baseline,
    mode="mean",
    tmin=-1.0,
    tmax=4.0,
    timefreqs=[(0.5, 10), (2.0, 20)],
    title=f"TFR potencia (Morlet) + topomapas\n{run_info}",
    show=False,
)
fig_power_joint.set_size_inches(12, 6)
if show_figs:
    fig_power_joint.show()
if save_figs:
    fig_power_joint.savefig(os.path.join(fig_dir, f"{fig_prefix}_tfr_potencia_joint.png"), dpi=150, bbox_inches='tight')

# ---- Inspeccion de ITC (coherencia inter-trial) ----
# ITC mide la consistencia de la fase entre epocas (0=aleatorio, 1=perfectamente coherente).
# No lleva correccion de baseline porque ya es una medida relativa (adimensional).

fig_itc_topo = itc.plot_topo(
    title=f"ITC media (coherencia inter-trial) | {run_info}",
    show=False,
)
fig_itc_topo.set_size_inches(10, 8)
if show_figs:
    fig_itc_topo.show()
if save_figs:
    fig_itc_topo.savefig(os.path.join(fig_dir, f"{fig_prefix}_tfr_itc_topo.png"), dpi=150, bbox_inches='tight')

# plot() devuelve lista de figuras; se toma la primera.
fig_itc_ch = itc.plot(
    picks=[ch_idx],
    title=f"ITC — {itc.ch_names[ch_idx]} | {run_info}",
    show=False,
)[0]
fig_itc_ch.set_size_inches(10, 5)
if show_figs:
    fig_itc_ch.show()
if save_figs:
    fig_itc_ch.savefig(os.path.join(fig_dir, f"{fig_prefix}_tfr_itc_ch-{itc.ch_names[ch_idx]}.png"), dpi=150, bbox_inches='tight')

fig_itc_joint = itc.plot_joint(
    tmin=-1.0,
    tmax=4.0,
    timefreqs=[(0.5, 10), (2.0, 20)],
    title=f"ITC + topomapas\n{run_info}",
    show=False,
)
fig_itc_joint.set_size_inches(12, 6)
if show_figs:
    fig_itc_joint.show()
if save_figs:
    fig_itc_joint.savefig(os.path.join(fig_dir, f"{fig_prefix}_tfr_itc_joint.png"), dpi=150, bbox_inches='tight')