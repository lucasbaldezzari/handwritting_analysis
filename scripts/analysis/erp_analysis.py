"""
Análisis de ERPs (potenciales evocados relacionados a eventos).
Referencia: https://mne.tools/stable/auto_tutorials/evoked/30_eeg_erp.html
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pyhwr.managers import GHiampDataManager, LSLDataManager
import mne

# ─── Parámetros ───────────────────────────────────────────────────────────────
sub = "02"
ses = "02"
task = "ejecutada"   # "ejecutada" incluye penDown; "imaginada" no
run = "05"
subject_folder = f"ses-{ses}"
type_signal = "eeg"
path = f"D:\\dataset\\sub-{sub}\\ses-{ses}"
show_figs = True
save_figs = False

use_ica       = True
ica_json_path = f"{path}\\sub-{sub}_ses-{ses}_task-{task}_run-{run}_ica.json"

baseline_erp = (-0.3, 0.)
tmin_epocs = -1.5
tmax_epocs = 4

# ─── Carga de datos ────────────────────────────────────────────────────────────
lsl_file    = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_{type_signal}.xdf"
ghiamp_file = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_{type_signal}.hdf5"

gmanager = GHiampDataManager(os.path.join(path, ghiamp_file), normalize_time=True)
lsl_manager = LSLDataManager(os.path.join(path, lsl_file))

raw_data = gmanager.raw_data.swapaxes(1, 0)

gmanager.changeMarkersNames({1: "startRun", 2: "trialTablet", 3: "penDown", 4: "trialLaptop"})
t0_gtec = gmanager.markers_info["startRun"][0]
markers_info = gmanager.markers_info
trials_tablet = np.array(markers_info["trialTablet"])
has_pen_down = "penDown" in markers_info and len(markers_info["penDown"]) > 0
pen_down = np.array(markers_info["penDown"]) if has_pen_down else np.array([])

letras = [
    lsl_manager.trials_info["Tablet_Markers"][i]["letter"]
    for i in range(1, len(lsl_manager.trials_info["Tablet_Markers"]) + 1)
]
start_time_tablet        = lsl_manager.trials_info["Tablet_Markers"][1]["sessionStartTime"] / 1000
rest_times               = np.array(lsl_manager["Tablet_Markers", "trialRestTime", :]) / 1000 - start_time_tablet
rest_times_relative_gtec = rest_times + t0_gtec

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

times_markers = np.concatenate(([t0_gtec], times_markers))
labels = ["startRun"] + labels

# ─── Objeto MNE Raw ────────────────────────────────────────────────────────────
sfreq        = gmanager.sample_rate
montage_df   = pd.read_csv(".\\ghiamp_montage.sfp", sep="\t", header=None)
eeg_ch_names = list(montage_df[0])[:64]
ch_names     = eeg_ch_names + ["EMG1"] + ["EOG1", "EOG2"]
ch_types     = ["eeg"] * 64 + ["emg"] + ["eog"] * 2

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw_signal = mne.io.RawArray(raw_data, info)

montage = mne.channels.read_custom_montage(".\\ghiamp_montage.sfp")
raw_signal.set_montage(montage, on_missing="ignore")

anotaciones = mne.Annotations(
    onset=times_markers, duration=[0] * len(times_markers), description=labels
)
raw_signal.set_annotations(anotaciones)

tmin_crop = trials_tablet[0] - 4.0
tmax_crop = rest_times_relative_gtec[-1] + 2.0
raw_signal.crop(tmin=tmin_crop, tmax=tmax_crop)

# ─── Limpieza ICA (opcional) ──────────────────────────────────────────────────
if use_ica and ica_json_path:
    from analysis.ica_apply import ICAApplicator
    _cleaner = ICAApplicator(ica_json_path)
    # Remueve los canales malos del JSON antes de aplicar ICA y filtrar.
    _cleaner.apply_to_raw(raw_signal)

# ─── Filtros adaptados a ERP ──────────────────────────────────────────────────
# EEG: 0.1 Hz como cota inferior para conservar componentes lentos (P300, N200, etc.).
# El análisis TFR/PSD usa 4 Hz, lo que eliminaría estas componentes.
raw_signal.filter(l_freq=4.0, h_freq=30.0, picks='eeg', fir_design='firwin')
raw_signal.filter(l_freq=1.0, h_freq=15.0, picks='eog', fir_design='firwin')
raw_signal.filter(l_freq=1.0, h_freq=None, picks='emg', fir_design='firwin')
raw_signal.notch_filter([50])

scalings = {'eeg': 30, 'emg': 300, 'eog': 150}

# Colores diferenciados por tipo de canal
color = {'eeg': 'steelblue', 'emg': 'forestgreen', 'eog': 'darkorange'}

# raw_signal.plot(scalings=scalings, color=color, duration=40)

# ─── Pendown delay ────────────────────────────────────────────────────────────
# Tiempo medio desde el cue de letra hasta el contacto del lápiz.
# Solo disponible en tarea "ejecutada"; en "imaginada" se omite la segunda línea.
if has_pen_down:
    n_pairs  = min(len(trials_tablet), len(pen_down))
    delays   = pen_down[:n_pairs] - trials_tablet[:n_pairs]
    mean_pd  = float(np.mean(delays[delays > 0]))
    vlines   = [0.0, mean_pd]
    print(f"Pendown delay medio: {mean_pd * 1000:.1f} ms")
else:
    mean_pd = None
    vlines  = [0.0]

# ─── Épocas ───────────────────────────────────────────────────────────────────
eeg_signal = raw_signal.copy().pick('eeg')
events, event_id = mne.events_from_annotations(eeg_signal)

marcadores_no_letra = {"startRun", "pd", "rest"}
event_id_letras     = {k: v for k, v in event_id.items() if k not in marcadores_no_letra}
print("Eventos de letra:", event_id_letras)


reject       = {'eeg': 300}   # µV pico a pico — más estricto que TFR

epochs = mne.Epochs(
    eeg_signal,
    events,
    event_id=event_id_letras,
    tmin=tmin_epocs,
    tmax=tmax_epocs,
    baseline=baseline_erp,
    reject=reject,
    preload=True,
)
print(epochs)
print(f"Épocas retenidas: {len(epochs)} / {len(epochs.drop_log)}")

# ─── Configuración de figuras ─────────────────────────────────────────────────
run_info   = f"Sub-{sub} | Ses-{ses} | Run-{run} | Tarea: {task}"
fig_prefix = f"sub-{sub}_ses-{ses}_run-{run}_task-{task}"

if save_figs:
    fig_dir = os.path.join("images", f"sub-{sub}", f"ses-{ses}", f"run-{run}", task)
    os.makedirs(fig_dir, exist_ok=True)

# Canales de interés para ERP: Pz (clásico P300), Cz (central), FCz (fronto-central)
erp_channels = ["Pz", "Cz", "FCz", "C1", "C2"]

# Handles de leyenda para las líneas verticales (reutilizado en varias figuras)
vline_handles = [Line2D([0], [0], color='gray', ls='--', lw=1, label='Cue onset (t=0)')]
if has_pen_down:
    vline_handles.append(
        Line2D([0], [0], color='tomato', ls='--', lw=1.5,
               label=f'Pendown medio ({mean_pd * 1000:.0f} ms)'))

# ─── Figura 1a: Promedio evocado (canales ERP clave) ─────────────────────────
evoked_all = epochs.average()

fig_evoked = evoked_all.plot(picks=erp_channels, show=False, gfp=True)
if has_pen_down:
    for ax in fig_evoked.get_axes():
        if ax.get_xlabel():   # ejes de datos (tienen etiqueta en x)
            ax.axvline(mean_pd, color='tomato', ls='--', lw=1.5)
            ax.legend(handles=vline_handles, fontsize=8, loc='upper right')
fig_evoked.suptitle(f"Promedio evocado — {', '.join(erp_channels)}\n{run_info}")
if show_figs:
    fig_evoked.show()
if save_figs:
    fig_evoked.savefig(
        os.path.join(fig_dir, f"{fig_prefix}_erp_evoked.png"), dpi=150, bbox_inches='tight'
    )

# ─── Figura 1b: Joint plot — ERP + topomapas en picos principales ─────────────
fig_joint = evoked_all.plot_joint(picks='eeg', times='peaks', show=False)
fig_joint.suptitle(f"Evocado + distribución topográfica\n{run_info}")
if show_figs:
    fig_joint.show()
if save_figs:
    fig_joint.savefig(
        os.path.join(fig_dir, f"{fig_prefix}_erp_joint.png"), dpi=150, bbox_inches='tight')

# ─── Figura 2: Comparación de evocados por letra ──────────────────────────────
# Superpone el promedio de cada letra en Pz para evaluar si distintas letras
# producen patrones ERP diferenciados o si hay un potencial consistente entre ellas.
evokeds_letras = {
    letra: epochs[letra].average()
    for letra in sorted(event_id_letras.keys())
    if len(epochs[letra]) > 0
}

if len(evokeds_letras) > 1:
    figs_compare = mne.viz.plot_compare_evokeds(
        evokeds_letras,
        picks=["Cz"],
        vlines=vlines,
        show=False,
        title=f"Evocados por letra — Pz\n{run_info}",
    )
    fig_compare = figs_compare[0] if isinstance(figs_compare, list) else figs_compare
else:
    fig_compare = evoked_all.plot(picks=["Pz"], vlines=[0.0], show=False)
    fig_compare.suptitle(f"Evocado promedio — Pz\n{run_info}")

if show_figs:
    fig_compare.show()
if save_figs:
    fig_compare.savefig(
        os.path.join(fig_dir, f"{fig_prefix}_erp_compare_letras.png"),
        dpi=150, bbox_inches='tight',
    )

# ─── Figura 3: Heatmap de épocas (epoch image) ───────────────────────────────
# Eje X: tiempo; eje Y: cada trial; color: voltaje.
# Los trials se ordenan por letra para agrupar condiciones similares verticalmente,
# facilitando detectar visualmente si hay ERPs consistentes dentro de cada letra.
id_to_letter  = {v: k for k, v in epochs.event_id.items()}
letter_labels = np.array([id_to_letter[e] for e in epochs.events[:, 2]])
order = np.argsort(letter_labels, kind='stable')

for ch in erp_channels:
    figs_img = epochs.plot_image(
        picks=[ch],
        # order=order,
        show=False,
    )
    fig_img = figs_img[0]
    fig_img.suptitle(f"Epoch image — {ch} (ordenado por letra)\n{run_info}")
    if show_figs:
        fig_img.show()
    if save_figs:
        fig_img.savefig(
            os.path.join(fig_dir, f"{fig_prefix}_erp_image_{ch}.png"),
            dpi=150, bbox_inches='tight',
        )

plt.show()
