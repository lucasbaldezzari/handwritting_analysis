"""
Análisis de épocas de escritura y reposo por trial.

Genera dos tipos de épocas de igual duración:
  - trials_epochs : ancladas al primer penDown de cada trial (inicio de escritura)
  - rest_epochs   : ancladas al marcador rest de cada trial (período de reposo)

La duración compartida (tmax_epoch) es el mínimo de (último penUp - primer penDown)
de todos los trials, con un piso en tmax_umbral.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyhwr.managers import GHiampDataManager, LSLDataManager
from analysis.ica_apply import ICAApplicator
import mne

# ─── Parámetros configurables ─────────────────────────────────────────────────

sub  = "01"
ses  = "02"
task = "ejecutada"
run  = "06"

type_signal = "eeg"
path = f"D:\\dataset\\sub-{sub}\\ses-{ses}"

show_figs  = True
save_figs  = True
block_figs = True

use_ica       = True
ica_json_path = f"{path}\\sub-{sub}_ses-{ses}_task-{task}_run-{run}_ica.json"

drop_occipital_channels = False
occipital_channels = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]

# Duración mínima aceptable de época (segundos).
# Si min(last_penUp − first_penDown) < tmax_umbral se usa tmax_umbral.
tmax_umbral  = 1.0
tmin_epocs   = -1.25   # segundos previos al evento de anclaje (penDown o rest)

# Umbral de rechazo de época por EEG (µV pico a pico → V para MNE).
# MNE usa la unidad SI del canal: para EEG es Voltios.
# Ejemplo: 5000 µV = 5000e-6 V = 5 mV (umbral generoso, ajustar según la señal).
reject_threshold  =  150  # uV

# Canales centromotores a mostrar en el image plot (vacío → todos los EEG)
channels_image_plot = ["FC3", "FC1", "FCz", "FC2", "FC4",
                       "C3",  "C1",  "Cz",  "C2",  "C4",
                       "CP3", "CP1", "CPz", "CP2", "CP4"]

scalings = {'eeg': 20, 'emg': 100, 'eog': 100}
color    = {'eeg': 'steelblue', 'emg': 'forestgreen', 'eog': 'darkorange'}

# ─── Parámetros TFR ───────────────────────────────────────────────────────────
tfr_freqs          = np.logspace(*np.log10([5, 30]), num=20)
tfr_ncycles        = tfr_freqs / 2.0
tfr_time_bandwidth = 5.0    # resolución espectral multitaper (nº tapers = 2*tbw-1 = 7)
tfr_decim          = 5

# Bandas de potencia para topomaps
tfr_bands = {"Theta (4-8 Hz)": (4, 8), "Alpha (8-13 Hz)": (8, 13), "Beta (13-30 Hz)": (13, 30)}

# Puntos (tiempo_s, frecuencia_Hz) para los topomaps del plot_joint.
# None → MNE detecta automáticamente los picos más prominentes.
# Ejemplo manual: [(0.3, 10), (0.8, 20)]
tfr_timefreqs = [(0.,10.5),(0.1,10.5),(0.3,10.5),(0.5,10.5)]#None

# Baseline para corrección de TFR/topomaps.
# None → sin corrección.  Ejemplo: (tmin_epocs, 0.0) usa el período pre-evento.
baseline_topomaps = (-1.2, -1.)#None
tfr_mode          = "logratio"   # modo de corrección: logratio, ratio, mean, percent, zscore

# ─── Archivos de entrada ──────────────────────────────────────────────────────

lsl_file    = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_{type_signal}.xdf"
ghiamp_file = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_{type_signal}.hdf5"

# ─── Carga de datos ───────────────────────────────────────────────────────────

gmanager    = GHiampDataManager(os.path.join(path, ghiamp_file), normalize_time=True)
lsl_manager = LSLDataManager(os.path.join(path, lsl_file))

raw_data = gmanager.raw_data.swapaxes(1, 0)  # canales × muestras
raw_eeg  = raw_data[:64, :]
raw_emg  = raw_data[[64], :]
raw_eog  = raw_data[[65, 66], :]

# ─── Marcadores del g.HIAMP ───────────────────────────────────────────────────

gmanager.changeMarkersNames({1: "startRun", 2: "trialTablet", 3: "penDown", 4: "trialLaptop"})
t0_gtec = gmanager.markers_info["startRun"][0]

markers_info   = gmanager.markers_info
trials_tablet  = np.array(markers_info["trialTablet"])
trials_laptop  = np.array(markers_info["trialLaptop"])

raw_has_pen_down = "penDown" in markers_info and len(markers_info["penDown"]) > 0
pen_down         = np.array(markers_info["penDown"]) if raw_has_pen_down else np.array([])
has_pen_down     = task == "ejecutada" and raw_has_pen_down

if raw_has_pen_down and task != "ejecutada":
    print(f"Advertencia: se ignoraron {len(pen_down)} marcadores penDown porque task='{task}'.")

# ─── Etiquetas y tiempos de rest desde LSL ───────────────────────────────────

letras = [
    lsl_manager.trials_info["Tablet_Markers"][i]["letter"]
    for i in range(1, len(lsl_manager.trials_info["Tablet_Markers"]) + 1)
]
start_time_tablet       = lsl_manager.trials_info["Tablet_Markers"][1]["sessionStartTime"] / 1000
rest_times              = np.array(lsl_manager["Tablet_Markers", "trialRestTime", :]) / 1000 - start_time_tablet
rest_times_relative_gtec = rest_times + t0_gtec

n_trials = min(len(letras), len(trials_tablet), len(rest_times_relative_gtec))
if n_trials < max(len(letras), len(trials_tablet), len(rest_times_relative_gtec)):
    print(
        f"Advertencia: se recortaron eventos para alinear "
        f"letras={len(letras)}, trialTablet={len(trials_tablet)}, "
        f"rest={len(rest_times_relative_gtec)}."
    )

# ─── Tiempos de penDown / penUp por trial desde LSL ──────────────────────────
#
# penDownMarkers y penUpMarkers vienen en ms absolutos desde el tablet.
# Se convierten a segundos en el frame del g.HIAMP usando la misma fórmula
# que rest_times: time_gtec = (ms / 1000) - start_time_tablet + t0_gtec

first_pendown_gtec = []
writing_durations  = []

for i in range(1, n_trials + 1):
    trial  = lsl_manager.trials_info["Tablet_Markers"][i]
    pd_ms  = np.array(trial.get("penDownMarkers", []))
    pu_ms  = np.array(trial.get("penUpMarkers",   []))

    if len(pd_ms) == 0 or len(pu_ms) == 0:
        # Fallback: usar marcador GHiamp o cue de trial si no hay datos tablet
        fallback = pen_down[i - 1] if (has_pen_down and i - 1 < len(pen_down)) else trials_tablet[i - 1]
        first_pendown_gtec.append(float(fallback))
        writing_durations.append(tmax_umbral)
        print(f"  Trial {i}: sin penDownMarkers/penUpMarkers en LSL, usando fallback.")
        continue

    first_pd_gtec = (pd_ms[0]  / 1000) - start_time_tablet + t0_gtec
    last_pu_gtec  = (pu_ms[-1] / 1000) - start_time_tablet + t0_gtec
    duration      = last_pu_gtec - first_pd_gtec

    first_pendown_gtec.append(float(first_pd_gtec))
    writing_durations.append(float(duration))

# tmax compartido: mínimo de la duración de escritura, con piso en tmax_umbral
computed_tmax = min(writing_durations)
if computed_tmax < tmax_umbral:
    print(
        f"tmax calculado ({computed_tmax:.3f} s) es menor que tmax_umbral "
        f"({tmax_umbral} s). Se usa tmax_umbral."
    )
    tmax_epoch = tmax_umbral
else:
    tmax_epoch = computed_tmax

print(f"tmax_epoch = {tmax_epoch:.3f} s (duración de cada época)")

# ─── Construcción del objeto MNE Raw ──────────────────────────────────────────

sfreq      = gmanager.sample_rate
montage_df = pd.read_csv(".\\analysis\\ghiamp_montage.sfp", sep="\t", header=None)

eeg_ch_names = list(montage_df[0])[:64]
emg_ch_names = ["EMG1"]
eog_ch_names = ["EOG1", "EOG2"]
ch_names = eeg_ch_names + emg_ch_names + eog_ch_names
ch_types = ["eeg"] * 64 + ["emg"] + ["eog"] * 2

info       = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw_signal = mne.io.RawArray(raw_data, info)

montage = mne.channels.read_custom_montage(".\\analysis\\ghiamp_montage.sfp")
raw_signal.set_montage(montage, on_missing="ignore")

# Anotaciones: misma lógica que analisis_piloto.py (no se usan en epoch creation,
# pero sirven para visualización de la señal cruda)
events_labeled = [(t0_gtec, "startRun")]
for i in range(n_trials):
    events_labeled.append((trials_tablet[i], letras[i]))
    if has_pen_down and i < len(pen_down):
        events_labeled.append((pen_down[i], "pd"))
    events_labeled.append((rest_times_relative_gtec[i], "rest"))

events_labeled.sort(key=lambda item: item[0])
times_markers = np.array([t for t, _ in events_labeled])
labels        = [lbl for _, lbl in events_labeled]

anotaciones = mne.Annotations(
    onset=times_markers,
    duration=[0] * len(times_markers),
    description=labels,
)
raw_signal.set_annotations(anotaciones)

# Crop al intervalo útil de la sesión
tmin_crop = trials_tablet[0] - 4.0
tmax_crop = rest_times_relative_gtec[-1] + 2.0
raw_signal.crop(tmin=tmin_crop, tmax=tmax_crop)

# ─── ICA (opcional) ───────────────────────────────────────────────────────────

if use_ica and ica_json_path:
    _cleaner = ICAApplicator(ica_json_path)
    _cleaner.apply_to_raw(raw_signal)

# ─── Filtros ──────────────────────────────────────────────────────────────────

raw_signal.filter(l_freq=4.0, h_freq=30.0, picks="eeg", fir_design="firwin")
raw_signal.filter(l_freq=1.0, h_freq=15.0, picks="eog", fir_design="firwin")
raw_signal.filter(l_freq=5.0, h_freq=40.0, picks="emg", fir_design="firwin")
raw_signal.notch_filter([50])

# ─── Remoción de canales occipitales ─────────────────────────────────────────

if drop_occipital_channels:
    channels_to_drop = [ch for ch in occipital_channels if ch in raw_signal.ch_names]
    raw_signal.drop_channels(channels_to_drop)
    print(f"Canales occipitales removidos: {channels_to_drop}")

# ─── Señal EEG ───────────────────────────────────────────────────────────────

eeg_signal = raw_signal.copy().pick("eeg")

# ─── Construcción de eventos MNE ──────────────────────────────────────────────
#
# Se construyen directamente desde los tiempos calculados (no desde anotaciones)
# para anclar cada época al primer penDown / marcador rest del trial.

_sfreq        = eeg_signal.info["sfreq"]
_first_sample = eeg_signal.first_samp
_last_sample  = eeg_signal.last_samp

unique_letters = sorted(set(letras[:n_trials]))
letter_event_id = {letter: idx + 1 for idx, letter in enumerate(unique_letters)}

trial_events_list = []
rest_events_list  = []

for i in range(n_trials):
    ev_id = letter_event_id[letras[i]]

    # Índice absoluto del primer penDown (MNE espera índices absolutos, con first_samp)
    pd_sample = int(round(first_pendown_gtec[i] * _sfreq))
    if _first_sample <= pd_sample <= _last_sample:
        trial_events_list.append([pd_sample, 0, ev_id])
    else:
        print(f"  Trial {i+1}: penDown fuera de rango ({pd_sample}), descartado.")

    # Índice absoluto del marcador rest
    rest_sample = int(round(rest_times_relative_gtec[i] * _sfreq))
    if _first_sample <= rest_sample <= _last_sample:
        rest_events_list.append([rest_sample, 0, ev_id])
    else:
        print(f"  Trial {i+1}: rest fuera de rango ({rest_sample}), descartado.")

trial_events = np.array(trial_events_list, dtype=int)
rest_events  = np.array(rest_events_list,  dtype=int)

# ─── Creación de épocas ───────────────────────────────────────────────────────

common_epoch_kwargs = dict(
    tmin=tmin_epocs,
    tmax=tmax_epoch,
    baseline=None,
    reject={"eeg": reject_threshold},
    preload=True,
)

trials_epochs = mne.Epochs(
    eeg_signal, trial_events,
    event_id=letter_event_id,
    **common_epoch_kwargs,
)

rest_epochs = mne.Epochs(
    eeg_signal, rest_events,
    event_id=letter_event_id,
    **common_epoch_kwargs,
)

print(f"\ntrial_epochs  retenidas: {len(trials_epochs)} / {len(trials_epochs.drop_log)}")
print(f"rest_epochs   retenidas: {len(rest_epochs)}  / {len(rest_epochs.drop_log)}")

# ─── Preparación para guardar figuras ─────────────────────────────────────────

run_info   = f"Sub-{sub} | Ses-{ses} | Run-{run} | Tarea: {task}"
ica_suffix = f"ica{use_ica}"
fig_prefix = f"sub-{sub}_ses-{ses}_run-{run}_task-{task}_{ica_suffix}"

if save_figs:
    fig_dir = os.path.join("images", f"sub-{sub}", f"ses-{ses}", f"run-{run}", task)
    os.makedirs(fig_dir, exist_ok=True)


def _save(fig, fname):
    if save_figs:
        fig.savefig(os.path.join(fig_dir, fname), dpi=300, bbox_inches="tight")
    if show_figs:
        fig.show()
    else:
        plt.close(fig)


# ─── Señal completa con marcadores ───────────────────────────────────────────

fig_raw = raw_signal.plot(scalings=scalings, color=color, duration=40, show=show_figs)
if save_figs:
    fig_raw.savefig(
        os.path.join(fig_dir, f"{fig_prefix}_senal_filtrada.png"),
        dpi=300, bbox_inches="tight",
    )

# ─── Figuras temporales ───────────────────────────────────────────────────────

for epoch_type, epochs_obj, label in [
    ("trials", trials_epochs, "Escritura (penDown)"),
    ("rest",   rest_epochs,   "Reposo (rest)"),
]:
    # ── Browser de épocas ────────────────────────────────────────────────────
    fig_browser = epochs_obj.plot(
        scalings={'eeg': scalings['eeg']},
        title=f"{epoch_type}_epochs | {run_info}",
        show=show_figs,
    )
    if save_figs:
        fig_browser.savefig(
            os.path.join(fig_dir, f"{fig_prefix}_{epoch_type}_epochs_browser.png"),
            dpi=300, bbox_inches="tight",
        )

    evoked = epochs_obj.average()

    # ── Butterfly ERP ────────────────────────────────────────────────────────
    fig_erp = evoked.plot(show=False, time_unit="s")
    fig_erp.set_size_inches(12, 5)
    fig_erp.suptitle(
        f"ERP promedio — {label}\n"
        f"tmax={tmax_epoch:.3f} s | {run_info}"
    )
    _save(fig_erp, f"{fig_prefix}_{epoch_type}_erp.png")

    # ── Joint plot con topomaps ───────────────────────────────────────────────
    fig_joint = evoked.plot_joint(show=False, title=f"ERP Joint — {label} | {run_info}")
    _save(fig_joint, f"{fig_prefix}_{epoch_type}_joint.png")

    # ── Image plot (canales centromotores o todos si la lista está vacía) ─────
    picks_image = (
        [ch for ch in channels_image_plot if ch in epochs_obj.ch_names]
        if channels_image_plot else None
    )
    figs_image = epochs_obj.plot_image(
        picks=picks_image,
        combine="mean",   # promedia los canales seleccionados en una sola imagen
        title=f"Épocas imagen — {label} | {run_info}",
        show=False,
    )
    # plot_image puede devolver una lista de figuras
    if not isinstance(figs_image, list):
        figs_image = [figs_image]
    for j, fig_img in enumerate(figs_image):
        suffix = f"_{j}" if len(figs_image) > 1 else ""
        _save(fig_img, f"{fig_prefix}_{epoch_type}_image{suffix}.png")

    # ── TFR multitaper ───────────────────────────────────────────────────────
    power = epochs_obj.compute_tfr(
        method="multitaper",
        freqs=tfr_freqs,
        n_cycles=tfr_ncycles,
        time_bandwidth=tfr_time_bandwidth,
        average=True,
        return_itc=False,
        decim=tfr_decim,
    )

    # Topomap general: potencia media sobre todos los tiempos y canales
    fig_tfr_topo = power.plot_topo(
        baseline=baseline_topomaps,
        mode=tfr_mode,
        title=f"TFR potencia — {label} | {run_info}",
        show=False,
    )
    _save(fig_tfr_topo, f"{fig_prefix}_{epoch_type}_tfr_topo.png")

    # Joint plot: TFR de Cz + topomaps en los puntos (t, f) de interés.
    # tfr_timefreqs=None → MNE elige automáticamente los picos más prominentes.
    fig_tfr_joint = power.plot_joint(
        baseline=baseline_topomaps,
        mode=tfr_mode,
        timefreqs=tfr_timefreqs,
        title=f"TFR potencia (Multitaper) — {label} | {run_info}",
        show=False,
    )
    _save(fig_tfr_joint, f"{fig_prefix}_{epoch_type}_tfr_joint.png")

    # Topomaps de potencia por banda de frecuencia
    n_bands = len(tfr_bands)
    fig_bands, axes_bands = plt.subplots(1, n_bands, figsize=(4 * n_bands, 4),
                                         layout="constrained")
    if n_bands == 1:
        axes_bands = [axes_bands]
    topomap_kw = dict(ch_type="eeg", baseline=baseline_topomaps, mode=tfr_mode, show=False)
    for ax, (band_name, (fmin, fmax)) in zip(axes_bands, tfr_bands.items()):
        power.plot_topomap(fmin=fmin, fmax=fmax, axes=ax, **topomap_kw)
        ax.set_title(band_name)
    fig_bands.suptitle(f"Topomapas potencia por banda — {label} | {run_info}")
    _save(fig_bands, f"{fig_prefix}_{epoch_type}_tfr_bandas.png")

if block_figs and show_figs:
    plt.show(block=True)
