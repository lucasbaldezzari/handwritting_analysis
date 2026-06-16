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
sub  = "02"
ses  = "02"
task = "ejecutada"
run  = "05"

type_signal   = "eeg"
path          = f"D:\\dataset\\sub-{sub}\\ses-{ses}"
montage_path  = ".\\ghiamp_montage.sfp"
template_path = ".\\analysis\\ica_results_template.json"
output_path   = path   # dónde se guarda el JSON (misma carpeta que los datos)

n_components       = 30 #se puede usar un número entre 0 y 1 (varianza acumulada)
ica_method         = "fastica"
ica_random_state   = 97
ica_max_iter       = "auto"
bad_channels_known = ["F10"]   # canales malos conocidos a priori

# Componentes auto-detectados que en realidad parecen actividad cerebral.
# Estos se conservan: se excluyen del conjunto final a remover.
components_to_keep = []

apply_ica = True   # True → 2da pasada: aplica ICA y grafica antes/después
show_figs = True

# Grabaciones de referencia opcionales (mismo sujeto/sesión)
# Asignar la ruta al HDF5 correspondiente para mejorar la detección automática.
eog_ref_path = None   # task-eog: movimientos oculares intencionales
emg_ref_path = None   # task-emg: contracciones musculares intencionales

scalings = {'eeg': 30, 'emg': 150, 'eog': 150}
color    = {'eeg': 'steelblue', 'emg': 'forestgreen', 'eog': 'darkorange'}

# ─── Nombres de archivo ───────────────────────────────────────────────────────
ghiamp_file = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_{type_signal}.hdf5"
json_fname  = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_ica.json"
json_path   = os.path.join(output_path, json_fname)
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


def _load_and_filter_ref(hdf5_path):
    """Carga un registro de referencia (task-eog o task-emg) con solo canales EEG y EOG.
    EMG se descarta explícitamente: no se usa en ninguna detección de artefactos."""
    gm  = GHiampDataManager(hdf5_path, normalize_time=True)
    rd  = gm.raw_data.swapaxes(1, 0)
    inf = mne.create_info(ch_names=ch_names, sfreq=gm.sample_rate, ch_types=ch_types)
    raw_ref = mne.io.RawArray(rd, inf)
    raw_ref.set_montage(montage, on_missing="ignore")
    raw_ref.drop_channels(['EMG1'])
    raw_ref.filter(l_freq=1.0, h_freq=None, picks='eeg', fir_design='firwin')
    raw_ref.filter(l_freq=1.0, h_freq=None, picks='eog', fir_design='firwin')
    raw_ref.notch_filter([50])
    raw_ref.info['bads'] = bad_channels_known
    raw_ref.set_eeg_reference('average', projection=True)
    raw_ref.apply_proj()
    return raw_ref


# ─── Preprocesamiento (copia para ICA) ────────────────────────────────────────
# Se usa 1 Hz como corte inferior para todos los canales antes de ICA.
print("\n[2/7] Preprocesando señal para ICA...")
filt_raw = raw_signal.copy()

filt_raw.filter(l_freq=1.0, h_freq=None, picks='eeg', fir_design='firwin')
filt_raw.filter(l_freq=1.0, h_freq=None, picks='eog', fir_design='firwin')
filt_raw.filter(l_freq=1.0, h_freq=None, picks='emg', fir_design='firwin')
filt_raw.notch_filter([50])

filt_raw.info['bads'] = bad_channels_known
filt_raw.set_eeg_reference('average', projection=True)
filt_raw.apply_proj()

print(f"    Canales malos marcados: {filt_raw.info['bads']}")

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

ica_fif_fname = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_ica.fif"
ica_fif_path  = os.path.join(output_path, ica_fif_fname)
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

# ── Músculo: find_bads_muscle detecta potencia de alta frecuencia sin usar EMG1
muscle_indices, muscle_scores = ica.find_bads_muscle(filt_raw)
print(f"    Componentes musculares (registro experimental): {muscle_indices}")

# ── Músculo: registro de referencia (si está disponible)
muscle_ref_indices, muscle_ref_scores = [], np.array([])
muscle_ref_used = False
if emg_ref_path:
    print(f"    Cargando grabación de referencia EMG: {emg_ref_path}")
    filt_emg_ref = _load_and_filter_ref(emg_ref_path)
    muscle_ref_indices, muscle_ref_scores = ica.find_bads_muscle(filt_emg_ref)
    muscle_ref_used = True
    print(f"    Componentes musculares (grabación de referencia): {muscle_ref_indices}")

all_muscle_indices = sorted(set(muscle_indices) | set(muscle_ref_indices))
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

    # 1. Topomapas de todos los componentes
    ica.plot_components(picks=range(n_components), title="Topomapas de componentes ICA")

    # 2. Scores EOG (registro experimental)
    ica.plot_scores(eog_scores, exclude=eog_indices, title="Scores EOG — registro experimental")

    # 3. Scores EOG desde grabación de referencia (si fue usada)
    if eog_ref_used and len(eog_ref_scores) > 0:
        ica.plot_scores(
            eog_ref_scores, exclude=eog_ref_indices,
            title="Scores EOG — grabación de referencia"
        )

    # 4. Scores musculares (registro experimental)
    if len(muscle_scores) > 0:
        ica.plot_scores(
            muscle_scores, exclude=muscle_indices,
            title="Scores musculares — registro experimental"
        )

    # 5. Scores musculares desde grabación de referencia (si fue usada)
    if muscle_ref_used and len(muscle_ref_scores) > 0:
        ica.plot_scores(
            muscle_ref_scores, exclude=muscle_ref_indices,
            title="Scores musculares — grabación de referencia"
        )

    # 6. Propiedades detalladas de los componentes auto-excluidos
    if auto_excluded:
        ica.plot_properties(filt_raw, picks=auto_excluded)

    # 7. Series de tiempo de los componentes (interactivo: click para incluir/excluir)
    ica.plot_sources(filt_raw, title="Series de tiempo de componentes ICA")

    # 8. Overlay señal original vs. reconstruida (solo con componentes auto-detectados)
    if auto_excluded:
        ica.plot_overlay(filt_raw, exclude=auto_excluded,
                         title=f"Overlay auto — antes vs. después de ICA\n{run_info}")

    plt.show()

# ─── Guardar JSON BIDS-like ───────────────────────────────────────────────────
print(f"\n[6/7] Guardando resultados en: {json_path}")

eog_scores_list         = eog_scores.tolist()         if hasattr(eog_scores,         'tolist') else []
eog_ref_scores_list     = eog_ref_scores.tolist()     if hasattr(eog_ref_scores,     'tolist') else []
muscle_scores_list      = muscle_scores.tolist()      if hasattr(muscle_scores,      'tolist') else []
muscle_ref_scores_list  = muscle_ref_scores.tolist()  if hasattr(muscle_ref_scores,  'tolist') else []

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
result["auto_detected_components"]["muscle"]["indices"]        = all_muscle_indices
result["auto_detected_components"]["muscle"]["scores"]         = muscle_scores_list
result["auto_detected_components"]["muscle"]["ref_indices"]    = muscle_ref_indices
result["auto_detected_components"]["muscle"]["ref_scores"]     = muscle_ref_scores_list
result["auto_detected_components"]["muscle"]["ref_used"]       = muscle_ref_used
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

    raw_before_plot = filt_raw.copy()
    raw_after_plot = raw_clean.copy()
    for raw_plot in (raw_before_plot, raw_after_plot):
        raw_plot.filter(l_freq=1.0, h_freq=40.0, picks='eeg', fir_design='firwin')
        raw_plot.filter(l_freq=1.0, h_freq=40.0, picks='eog', fir_design='firwin')
        raw_plot.filter(l_freq=5.0, h_freq=40.0, picks='emg', fir_design='firwin')

    # ── 1. Overlay MNE: señal canal por canal antes vs. después
    ica.plot_overlay(raw_before_plot, exclude=final_components,
                     title=f"Overlay — antes vs. después de ICA\n{run_info}")

    # ── 2. Propiedades de los componentes excluidos finales
    if final_components:
        ica.plot_properties(filt_raw, picks=final_components)

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

    # ── 4. Señal limpia scrollable para inspección final
    raw_after_plot.plot(scalings=scalings, color=color,
                   title=f"Señal limpia post-ICA — {run_info}", duration=40)

    # Actualizar JSON: registrar que ICA fue aplicado
    ica_params["ica_applied"]                      = True
    ica_params["components_to_exclude"]["applied"] = final_components
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(ica_params, f, indent=2, ensure_ascii=False)
    print(f"    JSON actualizado: ica_applied=true")

    plt.show()

else:
    print("\n[7/7] apply_ica=False — se omite la aplicación de ICA.")
    print("    Poner apply_ica=True para aplicar y graficar antes/después.")
