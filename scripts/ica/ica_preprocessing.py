"""
Preprocesamiento ICA para limpieza de artefactos en señales EEG.

Flujo de trabajo en dos pasadas:
  Pasada 1 (apply_ica=False): Ajusta ICA, detecta artefactos automáticamente,
    muestra gráficas interactivas y guarda un JSON BIDS-like con los resultados.
  Edición manual: el usuario edita el JSON para ajustar canales/componentes malos.
  Pasada 2 (apply_ica=True): Lee el JSON con ediciones manuales, aplica ICA
    y muestra gráficas de comparación antes/después.
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyhwr.managers import GHiampDataManager
import mne
from mne.preprocessing import ICA

# ─── Parámetros de configuración ─────────────────────────────────────────────
sub  = "01"
ses  = "02"
task = "imaginada"
run  = "14"

type_signal   = "eeg"
path          = f"D:\\dataset\\sub-{sub}\\ses-{ses}"
montage_path  = ".\\analysis\\ghiamp_montage.sfp"
template_path = ".\\analysis\\ica_results_template.json"
output_path   = path   # dónde se guarda el JSON (misma carpeta que los datos)

ica_method         = "fastica"
ica_random_state   = 97
ica_max_iter       = "auto"
bad_channels_known = []   # canales malos conocidos a priori
n_components       = 50 - len(bad_channels_known) #se puede usar un número entre 0 y 1 (varianza acumulada)

# Componentes auto-detectados que en realidad parecen actividad cerebral.
# Estos se conservan: se excluyen del conjunto final a remover.
components_to_keep = []#[9,28,25,10,7,4]

apply_ica = True   # True → 2da pasada: aplica ICA y grafica antes/después
show_figs = False
show_final_overlay = False
show_final_properties = False
overlay_auto_ylim = True
overlay_ylim_percentiles = (1, 90)
overlay_ylim_margin = 0.10
overlay_start = 1.5
overlay_stop = None
plot_filter_l_freq = 1.0
plot_filter_h_freq = 40.0

# Grabaciones de referencia opcionales (mismo sujeto/sesión)
# Asignar la ruta al HDF5 correspondiente para mejorar la detección automática.
eog_ref_path = None   # task-eog: movimientos oculares intencionales
emg_ref_path = None   # task-emg: contracciones musculares intencionales

scalings = {'eeg': 20, 'emg': 150, 'eog': 150}
color    = {'eeg': 'steelblue', 'emg': 'forestgreen', 'eog': 'darkorange'}

# ─── Nombres de archivo ───────────────────────────────────────────────────────
ghiamp_file = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_{type_signal}.hdf5"
json_fname  = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_ica.json"
json_path   = os.path.join(output_path, json_fname)
ica_fif_fname = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_ica.fif"
ica_fif_path  = os.path.join(output_path, ica_fif_fname)
run_info    = f"Sub-{sub} | Ses-{ses} | Run-{run} | Tarea: {task}"

# ─── Carga de datos ────────────────────────────────────────────────────────────
print(f"\n[1/7] Cargando datos: {ghiamp_file}")
gmanager = GHiampDataManager(os.path.join(path, ghiamp_file), normalize_time=True)

raw_data = gmanager.raw_data.swapaxes(1, 0)   # → canales × muestras
sfreq    = gmanager.sample_rate

montage_df   = pd.read_csv(montage_path, sep="\t", header=None)
eeg_ch_names = list(montage_df[0])[:64]
ch_names     = eeg_ch_names + ["EMG1"] + ["EOG1", "EOG2"]
ch_types     = ["eeg"] * 64 + ["emg"] + ["eog"] * 2

info       = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw_signal = mne.io.RawArray(raw_data, info)

montage = mne.channels.read_custom_montage(montage_path)
raw_signal.set_montage(montage, on_missing="ignore")

print(f"    Cargados: {raw_signal.n_times} muestras | {len(raw_signal.ch_names)} canales | sfreq={sfreq} Hz")


def _drop_present_channels(raw, channels, context):
    present = [ch for ch in channels if ch in raw.ch_names]
    missing = [ch for ch in channels if ch not in raw.ch_names]
    if present:
        raw.drop_channels(present)
        print(f"    Canales removidos {context}: {present}")
    if missing:
        print(f"    Canales no presentes {context}: {missing}")
    return present


def _load_and_filter_ref(hdf5_path, keep_emg=True):
    """Carga un registro de referencia (task-eog o task-emg).
    Si keep_emg=True conserva EMG1 y lo filtra; útil para detección por correlación EMG."""
    gm  = GHiampDataManager(hdf5_path, normalize_time=True)
    rd  = gm.raw_data.swapaxes(1, 0)
    inf = mne.create_info(ch_names=ch_names, sfreq=gm.sample_rate, ch_types=ch_types)
    raw_ref = mne.io.RawArray(rd, inf)
    raw_ref.set_montage(montage, on_missing="ignore")
    if not keep_emg:
        raw_ref.drop_channels(['EMG1'])
    _drop_present_channels(raw_ref, bad_channels_known, "en referencia")
    raw_ref.filter(l_freq=1.0, h_freq=None, picks='eeg', fir_design='firwin')
    raw_ref.filter(l_freq=1.0, h_freq=None, picks='eog', fir_design='firwin')
    if keep_emg:
        raw_ref.filter(l_freq=1.0, h_freq=None, picks='emg', fir_design='firwin')
    raw_ref.notch_filter([50])
    raw_ref.set_eeg_reference('average', projection=True)
    raw_ref.apply_proj()
    return raw_ref


def _clip_overlay_ylim(fig, percentiles=(1, 99), margin=0.10):
    """Acota los ejes Y del overlay usando percentiles de las lineas visibles."""
    lower, upper = percentiles
    for ax in fig.axes:
        y_segments = []
        for line in ax.get_lines():
            if not line.get_visible():
                continue
            y_data = np.asarray(line.get_ydata(), dtype=float)
            y_data = y_data[np.isfinite(y_data)]
            if y_data.size:
                y_segments.append(y_data)

        if not y_segments:
            continue

        y_all = np.concatenate(y_segments)
        y_min, y_max = np.percentile(y_all, [lower, upper])
        if not (np.isfinite(y_min) and np.isfinite(y_max)):
            continue

        if y_min == y_max:
            pad = abs(y_min) * margin if y_min != 0 else margin
        else:
            pad = (y_max - y_min) * margin
        ax.set_ylim(y_min - pad, y_max + pad)

    fig.canvas.draw_idle()
    return fig


def _make_ica_plot_raw(raw):
    """Devuelve una copia filtrada solo para graficas ICA."""
    raw_plot = raw.copy()
    present_types = raw_plot.get_channel_types(unique=True)
    for ch_type in ("eeg", "eog", "emg"):
        if ch_type in present_types:
            raw_plot.filter(
                l_freq=plot_filter_l_freq,
                h_freq=plot_filter_h_freq,
                picks=ch_type,
                fir_design='firwin',
            )
    return raw_plot


# ─── Preprocesamiento (copia para ICA) ────────────────────────────────────────
# Se usa 1 Hz como corte inferior para todos los canales antes de ICA.
print("\n[2/7] Preprocesando señal para ICA...")
filt_raw = raw_signal.copy()
_drop_present_channels(filt_raw, bad_channels_known, "antes de ICA")

filt_raw.filter(l_freq=1.0, h_freq=None, picks='eeg', fir_design='firwin')
filt_raw.filter(l_freq=1.0, h_freq=None, picks='eog', fir_design='firwin')
filt_raw.filter(l_freq=1.0, h_freq=None, picks='emg', fir_design='firwin')
filt_raw.notch_filter([50])

filt_raw.set_eeg_reference('average', projection=True)
filt_raw.apply_proj()

print(f"    Canales disponibles para ICA: {len(filt_raw.ch_names)}")

# ─── Ajuste ICA ───────────────────────────────────────────────────────────────
print(f"\n[3/7] Ajustando ICA ({n_components} componentes, método={ica_method})...")
ica = ICA(
    n_components=n_components,
    method=ica_method,
    max_iter=ica_max_iter,
    random_state=ica_random_state,
)
ica.fit(filt_raw, picks='eeg')
print(ica)

ica.save(ica_fif_path, overwrite=True)
print(f"    ICA guardado en: {ica_fif_path}")

# ─── Detección automática de artefactos ───────────────────────────────────────
print("\n[4/7] Detectando artefactos automáticamente...")

# ── EOG: registro experimental
eog_indices, eog_scores = ica.find_bads_eog(
    filt_raw, ch_name=['EOG1', 'EOG2'], threshold=3.0
)
print(f"    Componentes EOG (registro experimental): {eog_indices}")

# ── EOG: registro de referencia (si está disponible)
eog_ref_indices, eog_ref_scores = [], np.array([])
eog_ref_used = False
if eog_ref_path:
    print(f"    Cargando grabación de referencia EOG: {eog_ref_path}")
    filt_eog_ref = _load_and_filter_ref(eog_ref_path)
    eog_ref_indices, eog_ref_scores = ica.find_bads_eog(
        filt_eog_ref, ch_name=['EOG1', 'EOG2'], threshold=3.0
    )
    eog_ref_used = True
    print(f"    Componentes EOG (grabación de referencia): {eog_ref_indices}")

all_eog_indices = sorted(set(eog_indices) | set(eog_ref_indices))
print(f"    Componentes EOG combinados: {all_eog_indices}")

# ── Músculo: find_bads_muscle detecta potencia de alta frecuencia (espectral)
muscle_indices, muscle_scores = ica.find_bads_muscle(filt_raw)
print(f"    Componentes musculares espectral (experimental): {muscle_indices}")

# ── Músculo: correlación con canal EMG1 (registro experimental)
emg_corr_indices, emg_corr_scores = ica.find_bads_eog(
    filt_raw, ch_name='EMG1', threshold=3.0
)
print(f"    Componentes EMG por correlación (experimental): {emg_corr_indices}")

# ── Músculo: registro de referencia (si está disponible)
muscle_ref_indices, muscle_ref_scores   = [], np.array([])
emg_ref_corr_indices, emg_ref_corr_scores = [], np.array([])
muscle_ref_used = False
if emg_ref_path:
    print(f"    Cargando grabación de referencia EMG: {emg_ref_path}")
    filt_emg_ref = _load_and_filter_ref(emg_ref_path, keep_emg=True)
    muscle_ref_indices, muscle_ref_scores = ica.find_bads_muscle(filt_emg_ref)
    emg_ref_corr_indices, emg_ref_corr_scores = ica.find_bads_eog(
        filt_emg_ref, ch_name='EMG1', threshold=3.0
    )
    muscle_ref_indices = sorted(set(muscle_ref_indices) | set(emg_ref_corr_indices))
    muscle_ref_used = True
    print(f"    Componentes musculares (grabación de referencia): {muscle_ref_indices}")

all_muscle_indices = sorted(set(muscle_indices) | set(emg_corr_indices) | set(muscle_ref_indices))
print(f"    Componentes musculares combinados: {all_muscle_indices}")

auto_detected  = sorted(set(all_eog_indices) | set(all_muscle_indices))
auto_excluded  = sorted(set(auto_detected) - set(components_to_keep))
ica.exclude    = auto_excluded
if components_to_keep:
    print(f"    Componentes auto-detectados: {auto_detected}")
    print(f"    Conservados (actividad cerebral): {components_to_keep}")
print(f"    Componentes auto-excluidos: {auto_excluded}")

# ─── Visualizaciones de inspección ────────────────────────────────────────────
if show_figs:
    print("\n[5/7] Generando gráficas de inspección...")
    filt_raw_plot = _make_ica_plot_raw(filt_raw)

    # 1. Topomapas de todos los componentes, divididos en dos figuras
    n_components_found = int(ica.n_components_)
    split_idx = int(np.ceil(n_components_found / 2))
    ica.plot_components(
        picks=range(0, split_idx),
        title="Topomapas de componentes ICA — primera mitad",
    )
    ica.plot_components(
        picks=range(split_idx, n_components_found),
        title="Topomapas de componentes ICA — segunda mitad",
    )

    # 2. Scores EOG (registro experimental)
    ica.plot_scores(eog_scores, exclude=eog_indices, title="Scores EOG — registro experimental")

    # 3. Scores EOG desde grabación de referencia (si fue usada)
    if eog_ref_used and len(eog_ref_scores) > 0:
        ica.plot_scores(
            eog_ref_scores, exclude=eog_ref_indices,
            title="Scores EOG — grabación de referencia"
        )

    # 4. Scores musculares espectral (registro experimental)
    if len(muscle_scores) > 0:
        ica.plot_scores(
            muscle_scores, exclude=muscle_indices,
            title="Scores musculares espectral — registro experimental"
        )

    # 5. Scores EMG por correlación con EMG1 (registro experimental)
    if len(emg_corr_scores) > 0:
        ica.plot_scores(
            emg_corr_scores, exclude=emg_corr_indices,
            title="Scores EMG (correlación con EMG1) — registro experimental"
        )

    # 6. Scores musculares desde grabación de referencia (si fue usada)
    if muscle_ref_used and len(muscle_ref_scores) > 0:
        ica.plot_scores(
            muscle_ref_scores, exclude=muscle_ref_indices,
            title="Scores musculares — grabación de referencia"
        )

    # 7. Scores EMG correlación desde grabación de referencia (si fue usada)
    if muscle_ref_used and len(emg_ref_corr_scores) > 0:
        ica.plot_scores(
            emg_ref_corr_scores, exclude=emg_ref_corr_indices,
            title="Scores EMG (correlación con EMG1) — grabación de referencia"
        )

    # 6. Propiedades detalladas de los componentes auto-excluidos
    if auto_excluded:
        ica.plot_properties(filt_raw_plot, picks=auto_excluded)

    # 7. Series de tiempo de los componentes (interactivo: click para incluir/excluir)
    ica.plot_sources(filt_raw_plot, title="Series de tiempo de componentes ICA")

    # 8. Overlay señal original vs. reconstruida (solo con componentes auto-detectados)
    if auto_excluded:
        fig_overlay_auto = ica.plot_overlay(
            filt_raw_plot, exclude=auto_excluded,
            start=overlay_start, stop=overlay_stop,
            title=f"Overlay auto — antes vs. después de ICA\n{run_info}"
        )
        if overlay_auto_ylim:
            _clip_overlay_ylim(
                fig_overlay_auto,
                percentiles=overlay_ylim_percentiles,
                margin=overlay_ylim_margin,
            )

    plt.show()

# ─── Guardar JSON BIDS-like ───────────────────────────────────────────────────
print(f"\n[6/7] Guardando resultados en: {json_path}")

eog_scores_list             = eog_scores.tolist()             if hasattr(eog_scores,             'tolist') else []
eog_ref_scores_list         = eog_ref_scores.tolist()         if hasattr(eog_ref_scores,         'tolist') else []
muscle_scores_list          = muscle_scores.tolist()          if hasattr(muscle_scores,          'tolist') else []
muscle_ref_scores_list      = muscle_ref_scores.tolist()      if hasattr(muscle_ref_scores,      'tolist') else []
emg_corr_scores_list        = emg_corr_scores.tolist()        if hasattr(emg_corr_scores,        'tolist') else []
emg_ref_corr_scores_list    = emg_ref_corr_scores.tolist()    if hasattr(emg_ref_corr_scores,    'tolist') else []

with open(template_path, 'r', encoding='utf-8') as f:
    result = json.load(f)

result["metadata"]["subject"]       = sub
result["metadata"]["session"] = ses
result["metadata"]["task"] = task
result["metadata"]["run"] = run
result["metadata"]["analysis_date"] = datetime.datetime.now().isoformat()
result["metadata"]["source_file"] = ghiamp_file

result["preprocessing"]["eeg_filter_l_freq"] = 1.0
result["preprocessing"]["eeg_filter_h_freq"] = None
result["preprocessing"]["eog_filter_l_freq"] = 1.0
result["preprocessing"]["eog_filter_h_freq"] = None
result["preprocessing"]["emg_filter_l_freq"] = 1.0
result["preprocessing"]["emg_filter_h_freq"] = None

result["bad_channels"]["auto_detected"] = bad_channels_known

result["ica_settings"]["n_components"] = n_components
result["ica_settings"]["method"] = ica_method
result["ica_settings"]["random_state"] = ica_random_state
result["ica_settings"]["max_iter"] = str(ica_max_iter)
result["ica_settings"]["n_components_found"] = int(ica.n_components_)

result["auto_detected_components"]["eog"]["indices"]           = all_eog_indices
result["auto_detected_components"]["eog"]["scores"]            = eog_scores_list
result["auto_detected_components"]["eog"]["ref_indices"]       = eog_ref_indices
result["auto_detected_components"]["eog"]["ref_scores"]        = eog_ref_scores_list
result["auto_detected_components"]["eog"]["ref_used"]          = eog_ref_used
result["auto_detected_components"]["muscle"]["indices"]             = all_muscle_indices
result["auto_detected_components"]["muscle"]["scores"]              = muscle_scores_list
result["auto_detected_components"]["muscle"]["emg_corr_indices"]    = emg_corr_indices
result["auto_detected_components"]["muscle"]["emg_corr_scores"]     = emg_corr_scores_list
result["auto_detected_components"]["muscle"]["ref_indices"]         = muscle_ref_indices
result["auto_detected_components"]["muscle"]["ref_scores"]          = muscle_ref_scores_list
result["auto_detected_components"]["muscle"]["emg_ref_corr_indices"]= emg_ref_corr_indices
result["auto_detected_components"]["muscle"]["emg_ref_corr_scores"] = emg_ref_corr_scores_list
result["auto_detected_components"]["muscle"]["ref_used"]            = muscle_ref_used
result["ref_paths"]["eog"] = str(eog_ref_path) if eog_ref_path else None
result["ref_paths"]["emg"] = str(emg_ref_path) if emg_ref_path else None

result["components_to_exclude"]["auto_detected"]   = auto_detected
result["components_to_exclude"]["kept_from_auto"]  = components_to_keep
result["components_to_exclude"]["auto"]            = auto_excluded
result["components_to_exclude"]["final"]           = auto_excluded
result["ica_file"]                                 = ica_fif_fname

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

print(f"    JSON guardado en: {json_path}")
print(f"\n    Próximos pasos:")
print(f"    1. Revisar {json_fname} y ajustar:")
print(f"       - 'bad_channels' > 'manual': canales malos adicionales")
print(f"       - 'components_to_exclude' > 'manual': componentes extra a excluir")
print(f"       - 'components_to_exclude' > 'final': lista definitiva para aplicar")
print(f"       - 'notes': observaciones del análisis visual")
print(f"    2. Poner apply_ica=True y volver a correr para ver antes/después.")

# ─── Aplicación ICA y visualización antes/después (Pasada 2) ─────────────────
if apply_ica:
    print(f"\n[7/7] Aplicando ICA — graficando antes/después...")

    with open(json_path, 'r', encoding='utf-8') as f:
        ica_params = json.load(f)

    final_components = ica_params["components_to_exclude"]["final"]
    bad_ch_all       = (
        ica_params["bad_channels"]["auto_detected"]
        + ica_params["bad_channels"]["manual"]
    )
    print(f"    Canales malos: {bad_ch_all}")
    print(f"    Componentes a eliminar: {final_components}")

    raw_clean = filt_raw.copy()
    ica.exclude = final_components
    ica.apply(raw_clean)

    raw_before_plot = _make_ica_plot_raw(filt_raw)
    raw_after_plot = _make_ica_plot_raw(raw_clean)

    # ── 1. Overlay MNE: señal canal por canal antes vs. después
    if show_final_overlay:
        fig_overlay = ica.plot_overlay(
            raw_before_plot, exclude=final_components,
            start=overlay_start, stop=overlay_stop,
            title=f"Overlay — antes vs. después de ICA\n{run_info}"
        )
        if overlay_auto_ylim:
            _clip_overlay_ylim(
                fig_overlay,
                percentiles=overlay_ylim_percentiles,
                margin=overlay_ylim_margin,
            )

    # ── 2. Propiedades de los componentes excluidos finales
    if show_final_properties and final_components:
        ica.plot_properties(raw_before_plot, picks=final_components)

    # ── 3. Comparación de PSD: antes vs. después (media de canales EEG)
    psd_before = raw_before_plot.compute_psd(picks='eeg', fmin=1.0, fmax=40.0)
    psd_after  = raw_after_plot.compute_psd(picks='eeg', fmin=1.0, fmax=40.0)

    data_before, freqs = psd_before.get_data(return_freqs=True)
    data_after, _      = psd_after.get_data(return_freqs=True)

    fig_psd, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(freqs, data_before.mean(axis=0) * 1e12,
                color='steelblue', alpha=0.85, label='Antes de ICA')
    ax.semilogy(freqs, data_after.mean(axis=0) * 1e12,
                color='tomato', alpha=0.85, label='Después de ICA')
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('PSD media (µV²/Hz)')
    ax.set_title(f'Comparación PSD — EEG (promedio de canales)\n{run_info}')
    ax.legend()
    fig_psd.tight_layout()

    # ── 4. Señal antes/después scrollable para inspección final
    raw_before_plot.plot(scalings=scalings, color=color,
                   title=f"Señal antes de ICA — {run_info}", duration=40)
    raw_after_plot.plot(scalings=scalings, color=color,
                   title=f"Señal después de ICA — {run_info}", duration=40)

    # Actualizar JSON: registrar que ICA fue aplicado
    ica_params["ica_applied"]                      = True
    ica_params["components_to_exclude"]["applied"] = final_components
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(ica_params, f, indent=2, ensure_ascii=False)
    print(f"JSON actualizado: ica_applied=true")

    if show_figs:
        plt.show()

else:
    print("\n[7/7] apply_ica=False — se omite la aplicación de ICA.")
    print("    Poner apply_ica=True para aplicar y graficar antes/después.")