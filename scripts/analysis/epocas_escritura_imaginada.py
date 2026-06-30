"""
Análisis de épocas de escritura IMAGINADA y reposo por trial.

Genera dos tipos de épocas de igual duración:
  - trials_epochs : ancladas al PRIMER penDown ESTIMADO de cada trial.
  - rest_epochs   : ancladas al marcador rest de cada trial (período de reposo).

En la tarea imaginada el sujeto IMAGINA escribir y no hay penDown ni penUp, por lo
que el tiempo de anclaje de la escritura debe estimarse. El tiempo de penDown es un
offset que transcurre desde trialTablet (el go-cue del experimento). La estimación se
obtiene, en orden de preferencia, de:

  1. Mínimo por letra del offset (first_penDown − trialTablet) medido en uno o más
     registros EJECUTADOS de referencia (mismo sub/ses, task="ejecutada").
  2. Mínimo global del offset sobre todos los registros ejecutados.
  3. Diccionario manual de offsets por letra definido por el usuario.
  4. Un valor escalar por defecto.

Las rest_epochs se obtienen igual que en epocas_escritura_ejecutada.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyhwr.managers import GHiampDataManager, LSLDataManager
from analysis.ica_apply import ICAApplicator
from analysis.pdf_report import PdfReport
import mne

# ─── Parámetros configurables ─────────────────────────────────────────────────

sub  = "01"
ses  = "02"
task = "imaginada"
run  = "14"

type_signal = "eeg"
path = f"D:\\dataset\\sub-{sub}\\ses-{ses}"

show_figs  = False
save_figs  = False
save_pdf   = True   # agrupa todas las figuras y el DataFrame en un único PDF
block_figs = True

use_ica       = True
ica_json_path = f"{path}\\sub-{sub}_ses-{ses}_task-{task}_run-{run}_ica.json"

drop_occipital_channels = False
occipital_channels = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]

# Lista adicional de canales a remover (independiente de los occipitales).
drop_additional_channels = True
additional_channels = ["C5","FC5"]   # ej. ["Fp1", "Fp2", "AF7"]

# ─── Estimación del penDown imaginado ─────────────────────────────────────────
#
# Registros EJECUTADOS de referencia (mismo sub/ses, task="ejecutada") de los que
# se estima el offset penDown por letra. Vacío → se usa el fallback manual.
executed_runs = ["05","06","07"]

# Fallback manual: offset (s desde trialTablet hasta el primer penDown) por letra.
# Se usa cuando una letra no tiene dato en los registros ejecutados (y tampoco hay
# mínimo global) o cuando executed_runs está vacío.
manual_pendown_offsets = {}          # ej. {"a": 0.8, "e": 0.75}
default_pendown_offset = 0.8         # s, usado para letras sin dato

# Si False, se ignora todo offset y la escritura se ancla exactamente en trialTablet
# (offset 0 para todas las letras). La tabla de offsets se sigue calculando y mostrando.
use_pendown_offset = False

# ─── Duración de las épocas ──────────────────────────────────────────────────
# Duración mínima aceptable de época (segundos).
tmax_umbral  = 2.0
# Duración mínima (s) para considerar válida la escritura de un trial.
# Trials con (último penUp − primer penDown) por debajo se descartan como artefacto
# (p. ej. toques espurios) al computar tmax. Los trials genuinos duran ≥ ~1.25 s.
min_valid_duration = 2.0
# Duración de cada época. None → se estima de los registros ejecutados
# (min de las duraciones de escritura, con piso tmax_umbral); si no hay
# registros ejecutados → tmax_umbral.
tmax_imaginada = None

tmin_epocs   = -1.25   # segundos previos al evento de anclaje (penDown o rest)

# Umbral de rechazo de época por EEG (µV pico a pico → V para MNE).
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
tfr_timefreqs = [(0.1,9.6),(0.5,9.6),(1.,9.6),
                 (0.1,18.),(0.5,18.),(1.,18.)]#None

# Baseline para corrección de TFR/topomaps.
baseline_topomaps = (-1.2, -1.0)#None
tfr_mode          = "logratio"   # modo de corrección: logratio, ratio, mean, percent, zscore

# Ventana temporal para gráficos de comparación (baseline aplicado al objeto, luego crop).
tmin_plot = 0.0   # segundos; inicio del recorte (por defecto: evento penDown/rest)
tmax_plot = None  # segundos; None → se asigna tmax_epoch una vez calculado


# ─── Estimación de offsets penDown desde registros ejecutados ────────────────

def collect_executed_pendown_offsets(sub, ses, runs, type_signal, path,
                                     min_valid_duration=0.5):
    """
    Para cada run EJECUTADO calcula, por trial:
        offset_i = first_penDown_gtec − trialTablet_gtec   (s desde el go-cue)
    y lo agrupa por letra. También acumula la duración de escritura por trial
    (last_penUp − first_penDown) para estimar tmax.

    Trials sin penDown/penUp se omiten por completo. Trials con duración menor a
    min_valid_duration se consideran artefacto (p. ej. toques espurios) y se excluyen
    SOLO del cómputo de duración (su offset por letra sigue siendo válido).

    Devuelve
    -------
    per_letter_offsets : dict[str, float]        offset MÍNIMO por letra (para anclaje)
    global_offset      : float | None            offset mínimo global (None si no hubo datos)
    writing_durations  : list[float]             duraciones de escritura válidas (s)
    offsets_by_letter  : dict[str, list[float]]  offsets crudos por letra (para la tabla)
    """
    offsets_by_letter = {}
    writing_durations = []

    for r in runs:
        lsl_file    = f"sub-{sub}_ses-{ses}_task-ejecutada_run-{r}_{type_signal}.xdf"
        ghiamp_file = f"sub-{sub}_ses-{ses}_task-ejecutada_run-{r}_{type_signal}.hdf5"

        try:
            gmanager    = GHiampDataManager(os.path.join(path, ghiamp_file), normalize_time=True)
            lsl_manager = LSLDataManager(os.path.join(path, lsl_file))
        except Exception as exc:
            print(f"  [ejecutado run-{r}] no se pudo cargar ({exc}). Se omite.")
            continue

        gmanager.changeMarkersNames(
            {1: "startRun", 2: "trialTablet", 3: "penDown", 4: "trialLaptop"}
        )
        t0_gtec       = gmanager.markers_info["startRun"][0]
        trials_tablet = np.array(gmanager.markers_info["trialTablet"])

        letras = [
            lsl_manager.trials_info["Tablet_Markers"][i]["letter"]
            for i in range(1, len(lsl_manager.trials_info["Tablet_Markers"]) + 1)
        ]
        start_time_tablet = lsl_manager.trials_info["Tablet_Markers"][1]["sessionStartTime"] / 1000

        n = min(len(letras), len(trials_tablet))
        n_trials_with_pd = 0
        n_short = 0

        for i in range(1, n + 1):
            trial = lsl_manager.trials_info["Tablet_Markers"][i]
            pd_ms = np.array(trial.get("penDownMarkers", []))
            pu_ms = np.array(trial.get("penUpMarkers",   []))
            if len(pd_ms) == 0 or len(pu_ms) == 0:
                continue

            # min/max para ser robustos al orden de los markers dentro del trial.
            first_pd_gtec = (pd_ms.min() / 1000) - start_time_tablet + t0_gtec
            last_pu_gtec  = (pu_ms.max() / 1000) - start_time_tablet + t0_gtec

            offset   = first_pd_gtec - trials_tablet[i - 1]
            duration = last_pu_gtec - first_pd_gtec

            # El offset por letra siempre es válido (primer penDown real).
            letra = letras[i - 1]
            offsets_by_letter.setdefault(letra, []).append(float(offset))
            n_trials_with_pd += 1

            # La duración sólo se usa si supera el piso (descarta toques espurios).
            if duration >= min_valid_duration:
                writing_durations.append(float(duration))
            else:
                n_short += 1
                print(f"    run-{r} t{i} ({letra}): duración {duration:.3f}s < "
                      f"{min_valid_duration}s, descartada del cómputo de tmax (artefacto).")

        print(f"  [ejecutado run-{r}] {n_trials_with_pd}/{n} trials con penDown utilizados; "
              f"{n_short} con duración corta excluidos de tmax.")

    # Para anclar la escritura imaginada se usa el MÍNIMO offset por letra
    # (el primer penDown más temprano observado), no el promedio.
    per_letter_offsets = {
        letra: float(np.min(vals)) for letra, vals in offsets_by_letter.items()
    }
    all_offsets   = [o for vals in offsets_by_letter.values() for o in vals]
    global_offset = float(np.min(all_offsets)) if all_offsets else None

    return per_letter_offsets, global_offset, writing_durations, offsets_by_letter


def build_offset_dataframe(offsets_by_letter):
    """Construye un DataFrame con estadísticos del offset penDown por letra.

    Columnas: n_trials, offset_min_s, offset_mean_s, offset_max_s, offset_std_s.
    El offset MÍNIMO (offset_min_s) es el usado para anclar la escritura imaginada.
    La última fila ("TODAS") resume sobre todos los offsets.
    Si no hay datos (sin registros ejecutados), devuelve un DataFrame vacío con
    las mismas columnas.
    """
    columns = ["n_trials", "offset_min_s", "offset_mean_s", "offset_max_s", "offset_std_s"]
    if not offsets_by_letter:
        return pd.DataFrame(columns=columns)

    def _stats(arr):
        return {
            "n_trials":      int(arr.size),
            "offset_min_s":  float(arr.min()),
            "offset_mean_s": float(arr.mean()),
            "offset_max_s":  float(arr.max()),
            # desvío estándar muestral (ddof=1); NaN si hay un solo trial.
            "offset_std_s":  float(arr.std(ddof=1)) if arr.size > 1 else np.nan,
        }

    rows = {
        letra: _stats(np.asarray(vals, dtype=float))
        for letra, vals in sorted(offsets_by_letter.items())
    }
    all_offsets = np.asarray(
        [o for vals in offsets_by_letter.values() for o in vals], dtype=float
    )
    rows["TODAS"] = _stats(all_offsets)

    df = pd.DataFrame.from_dict(rows, orient="index")[columns]
    df.index.name = "letra"
    return df


print("─── Estimando offsets penDown desde registros ejecutados ───")
if executed_runs:
    per_letter_offsets, global_offset, exec_writing_durations, offsets_by_letter = (
        collect_executed_pendown_offsets(
            sub, ses, executed_runs, type_signal, path,
            min_valid_duration=min_valid_duration,
        )
    )
else:
    print("  No se pasaron executed_runs: se usará el fallback manual de offsets.")
    per_letter_offsets, global_offset, exec_writing_durations, offsets_by_letter = {}, None, [], {}

# DataFrame de offsets penDown por letra (min, media, max, desvío y nº de trials).
# Vacío si no hay/no se pasan registros ejecutados.
offset_df = build_offset_dataframe(offsets_by_letter)
if not offset_df.empty:
    print("\n  Offset penDown por letra (s desde trialTablet):")
    print(offset_df.to_string(float_format=lambda v: f"{v:.3f}"))
else:
    print("  Sin registros ejecutados: DataFrame de offsets vacío.")

if global_offset is not None:
    print(f"  Offset global mínimo: {global_offset:.3f} s")


def resolve_offset(letra):
    """Resuelve el offset penDown para una letra según la precedencia acordada.

    Devuelve (offset_s, fuente)."""
    if letra in per_letter_offsets:
        return per_letter_offsets[letra], "por_letra_ejecutado"
    if global_offset is not None:
        return global_offset, "global_ejecutado"
    if letra in manual_pendown_offsets:
        return manual_pendown_offsets[letra], "manual_por_letra"
    return default_pendown_offset, "default_escalar"


# ─── Archivos de entrada (run imaginado) ──────────────────────────────────────

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

# En imaginada no debería haber penDown; si lo hubiera, se ignora (no se usa para anclar).
raw_has_pen_down = "penDown" in markers_info and len(markers_info["penDown"]) > 0
if raw_has_pen_down:
    print(
        f"Advertencia: se ignoraron {len(markers_info['penDown'])} marcadores penDown "
        f"porque task='{task}' (escritura imaginada)."
    )

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

# ─── Estimación del primer penDown por trial (escritura imaginada) ────────────
#
# No existen penDownMarkers en la tarea imaginada: el tiempo de anclaje se estima
# como trialTablet + offset, donde offset se resuelve por letra (ver resolve_offset).

print("\n─── Estimando primer penDown por trial imaginado ───")
first_pendown_gtec = []
fuentes_usadas     = {}

if not use_pendown_offset:
    print("  use_pendown_offset=False → offset 0; anclaje en trialTablet.")

for i in range(n_trials):
    letra = letras[i]
    if use_pendown_offset:
        offset, fuente = resolve_offset(letra)
    else:
        offset, fuente = 0.0, "sin_offset"
    first_pendown_gtec.append(float(trials_tablet[i] + offset))
    fuentes_usadas.setdefault(letra, (offset, fuente))

print("  Offset aplicado por letra (s desde trialTablet) y fuente:")
for letra, (offset, fuente) in sorted(fuentes_usadas.items()):
    print(f"    {letra}: {offset:.3f}  [{fuente}]")

# ─── Duración de las épocas ───────────────────────────────────────────────────

if tmax_imaginada is not None:
    tmax_epoch = tmax_imaginada
    print(f"\ntmax_epoch = {tmax_epoch:.3f} s (valor fijo tmax_imaginada).")
elif exec_writing_durations:
    computed_tmax = min(exec_writing_durations)
    tmax_epoch = max(computed_tmax, tmax_umbral)
    if computed_tmax < tmax_umbral:
        print(
            f"\ntmax estimado de ejecutados ({computed_tmax:.3f} s) < tmax_umbral "
            f"({tmax_umbral} s). Se usa tmax_umbral."
        )
    print(f"tmax_epoch = {tmax_epoch:.3f} s (estimado de registros ejecutados).")
else:
    tmax_epoch = tmax_umbral
    print(f"\ntmax_epoch = {tmax_epoch:.3f} s (tmax_umbral; sin registros ejecutados).")

if tmax_plot is None:
    tmax_plot = tmax_epoch

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

# Anotaciones (sólo para visualización de la señal cruda): trialTablet (letra),
# penDown estimado (pd_est) y rest.
events_labeled = [(t0_gtec, "startRun")]
for i in range(n_trials):
    events_labeled.append((trials_tablet[i], letras[i]))
    events_labeled.append((first_pendown_gtec[i], "pd_est"))
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

# ─── Remoción de canales adicionales ─────────────────────────────────────────

if drop_additional_channels:
    extra_to_drop = [ch for ch in additional_channels if ch in raw_signal.ch_names]
    raw_signal.drop_channels(extra_to_drop)
    print(f"Canales adicionales removidos: {extra_to_drop}")

# ─── Señal EEG ───────────────────────────────────────────────────────────────

eeg_signal = raw_signal.copy().pick("eeg")

# ─── Construcción de eventos MNE ──────────────────────────────────────────────
#
# Se construyen desde los tiempos calculados para anclar cada época al primer
# penDown ESTIMADO / marcador rest del trial.

_sfreq        = eeg_signal.info["sfreq"]
_first_sample = eeg_signal.first_samp
_last_sample  = eeg_signal.last_samp

unique_letters = sorted(set(letras[:n_trials]))
letter_event_id = {letter: idx + 1 for idx, letter in enumerate(unique_letters)}

trial_events_list = []
rest_events_list  = []

for i in range(n_trials):
    ev_id = letter_event_id[letras[i]]

    # Índice absoluto del primer penDown estimado (MNE espera índices absolutos, con first_samp)
    pd_sample = int(round(first_pendown_gtec[i] * _sfreq))
    if _first_sample <= pd_sample <= _last_sample:
        trial_events_list.append([pd_sample, 0, ev_id])
    else:
        print(f"  Trial {i+1}: penDown estimado fuera de rango ({pd_sample}), descartado.")

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

if save_figs or save_pdf:
    fig_dir = os.path.join("images", f"sub-{sub}", f"ses-{ses}", f"run-{run}", task)
    os.makedirs(fig_dir, exist_ok=True)

# Guardar el DataFrame de offsets penDown por letra como CSV (si hay datos).
if save_figs and not offset_df.empty:
    offset_df.to_csv(os.path.join(fig_dir, f"{fig_prefix}_offsets_pendown.csv"))

# Reporte PDF que agrupa el DataFrame y todas las figuras.
report = None
if save_pdf:
    report = PdfReport(os.path.join(fig_dir, f"{fig_prefix}_reporte.pdf"))
    if not offset_df.empty:
        report.add_dataframe(offset_df, title=f"Offsets penDown por letra (s) | {run_info}")


def _save(fig, fname):
    if save_figs:
        fig.savefig(os.path.join(fig_dir, fname), dpi=300, bbox_inches="tight")
    if report is not None:
        report.add_figure(fig)
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
if report is not None:
    report.add_figure(fig_raw)

# ─── Figuras temporales ───────────────────────────────────────────────────────

for epoch_type, epochs_obj, label in [
    ("trials", trials_epochs, "Escritura imaginada (penDown estimado)"),
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
    if report is not None:
        report.add_figure(fig_browser)

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

    # ── TFR recortado para comparación (baseline aplicado al objeto) ────────
    power_crop = power.copy()
    power_crop.apply_baseline(baseline=baseline_topomaps, mode=tfr_mode)
    power_crop.crop(tmin=tmin_plot, tmax=tmax_plot)
    _plot_time = f"{tmin_plot:.1f}s–{tmax_plot:.1f}s"

    fig_tfr_topo_crop = power_crop.plot_topo(
        title=f"TFR potencia [{_plot_time}] — {label} | {run_info}",
        show=False,
    )
    _save(fig_tfr_topo_crop, f"{fig_prefix}_{epoch_type}_tfr_topo_crop.png")

    fig_tfr_joint_crop = power_crop.plot_joint(
        timefreqs=tfr_timefreqs,
        title=f"TFR potencia (Multitaper) [{_plot_time}] — {label} | {run_info}",
        show=False,
    )
    _save(fig_tfr_joint_crop, f"{fig_prefix}_{epoch_type}_tfr_joint_crop.png")

    fig_bands_crop, axes_bands_crop = plt.subplots(
        1, n_bands, figsize=(4 * n_bands, 4), layout="constrained"
    )
    if n_bands == 1:
        axes_bands_crop = [axes_bands_crop]
    for ax, (band_name, (fmin, fmax)) in zip(axes_bands_crop, tfr_bands.items()):
        power_crop.plot_topomap(fmin=fmin, fmax=fmax, axes=ax, ch_type="eeg", show=False)
        ax.set_title(band_name)
    fig_bands_crop.suptitle(
        f"Topomapas potencia por banda [{_plot_time}] — {label} | {run_info}"
    )
    _save(fig_bands_crop, f"{fig_prefix}_{epoch_type}_tfr_bandas_crop.png")

if report is not None:
    report.close()
    print(f"\nReporte PDF guardado ({report.n_pages} páginas): {report.filepath}")

if block_figs and show_figs:
    plt.show(block=True)
