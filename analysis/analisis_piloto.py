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
sub = "01"
ses = "01"
task = "ejecutada" 
run = "01"
subject_folder = "s1"
type_signal = "eeg"
path = f"D:\\dataset\\{subject_folder}"

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
print("Nombre de los marcadores:", gmanager.markers_info.keys())
print("Tiempos de los marcadores:", gmanager.markers_info.values())
## cambio nombres de marcadores
gmanager.changeMarkersNames({1: "startRun", 2: "trialTablet", 3: "penDown", 4: "trialLaptop"})
t0_gtec = gmanager.markers_info["startRun"][0] #inicio de la ronda marcado por gRecorder usando el trigger de inicio de ronda

## Marcadores de trial de la tablet y de laptop. Se sugiere usar los marcadores de Tablet.
markers_info = gmanager.markers_info
trials_tablet = np.array(markers_info["trialTablet"])
trials_laptop = np.array(markers_info["trialLaptop"])
pen_down = np.array(markers_info["penDown"])

### ************** Obteniendo etiquetas para cada trial ***************
##LSL
letras = [lsl_manager.trials_info["Tablet_Markers"][i]["letter"] for i in range(1,len(lsl_manager.trials_info["Tablet_Markers"])+1)]
start_time_tablet = lsl_manager.trials_info["Tablet_Markers"][1]["sessionStartTime"]/1000
##tiempos de inicio de restTime
rest_times = np.array(lsl_manager["Tablet_Markers","trialRestTime",:])/1000 - start_time_tablet
rest_times_relative_gtec = rest_times + t0_gtec #+ 3
times_trialtablet = gmanager.markers_info["trialTablet"]

##concateno trials_tablet, pen_down y rest_times_relative_gtec y sorteo
times_markers = np.concatenate((trials_tablet, pen_down, rest_times_relative_gtec))
times_markers.sort()

labels = []
for letra, pen, rest in zip(letras, pen_down, rest_times_relative_gtec):
    labels.append(letra)
    labels.append("pd")
    labels.append("rest")

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
ch_types  = ["eeg"] * 64 + ["emg"] * 1 + ["eog"] * 2

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

print("Tipos de canal:", raw_signal.get_channel_types())

# Filtro pasa-banda 5-30 Hz SOLO sobre EEG.
raw_signal.filter(l_freq=5.0, h_freq=30.0, picks='eeg', fir_design='firwin')

#Filtro EOG
raw_signal.filter(l_freq=1, h_freq=15.0, picks='eog', fir_design='firwin')

# Filtro pasa-alto a 5 Hz para EMG: elimina deriva DC y artefactos de muy baja
# frecuencia (movimiento, respiración)
raw_signal.filter(l_freq=5.0, h_freq=None, picks='emg', fir_design='firwin')

# Filtro notch a 50 Hz (interferencia de red) para TODOS los canales
raw_signal.notch_filter([50])

# Escalas visuales diferenciadas por tipo de canal.
# EEG: ~30 µV; EOG: ~150 µV (potenciales oculares); EMG: ~300 µV (señal muscular)
scalings = {'eeg': 30, 'emg': 300, 'eog': 150}

# Colores diferenciados por tipo de canal
color = {'eeg': 'steelblue', 'emg': 'forestgreen', 'eog': 'darkorange'}

raw_signal.plot(scalings=scalings, color=color, duration=50, start=242)