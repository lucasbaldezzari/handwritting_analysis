"""
Remoción de artefactos por ICA sobre UNA SOLA ronda experimento.

Tercer enfoque, complementario a:
  - ica_preprocessing.py : una ICA CONJUNTA concatenando varias rondas.
  - ica_corrmap.py       : una ICA por ronda + corrmap entre soluciones.
  - ica_single_run.py    : (este) ICA ajustada y aplicada sobre el MISMO registro.

Aquí NO se concatena nada ni se comparan soluciones: se ajusta una ICA sobre la
ronda experimento (ejecutada o imaginada) y se detectan los artefactos usando los
canales EOG1/EOG2/EMG1 que la propia ronda ya incluye (ver scripts/analysis/
epocas_escritura.py). Toma como referencia el preprocessing.py del commit anterior
para la detección automática de EOG y EMG, pero aplicado a un único registro.

Remoción automática y/o puramente manual:
  - auto_detect=True  -> find_bads_eog (EOG1/EOG2) + find_bads_muscle + find_bads_eog(EMG1).
  - components_manual  -> lista de componentes a remover marcados a mano (o se edita
                          el JSON entre pasadas). final = (auto ∪ manual) − keep.
  - auto_detect=False y components_manual=[] -> usar las gráficas (plot_sources
    interactivo) para decidir y editar 'final' en el JSON antes de aplicar.

Compatibilidad: guarda EXACTAMENTE los mismos archivos que ica_preprocessing.py
e ica_corrmap.py (sub-..._ica.fif, sub-..._ica.json, sub-..._clean_raw.fif) con la
misma estructura de JSON, de modo que analysis.ica_apply.ICAApplicator los consume
sin distinguir la técnica. Un campo 'method'='single_run' deja traza.

Nota: la señal EEG vive en los .hdf5 (GHiampDataManager); los .xdf solo traen
marcadores, por eso la ICA usa solo los .hdf5.

Flujo de dos pasadas:
  Pasada 1 (apply_ica=False): ajusta ICA, detecta candidatos, grafica y escribe JSON.
  Pasada 2 (apply_ica=True): aplica la ICA y guarda -clean_raw.fif.
"""

import os
import sys
import json
import datetime
import numpy as np

# Consola Windows: forzar UTF-8 para no fallar con caracteres como '->', 'µ', etc.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8')
    except Exception:
        pass
import pandas as pd
import matplotlib.pyplot as plt
from pyhwr.managers import GHiampDataManager
import mne
from mne.preprocessing import ICA

# ─── Parámetros de configuración ─────────────────────────────────────────────
sub  = "01"
ses  = "02"

# Única ronda experimento a analizar / limpiar (ejecutada o imaginada).
target_task = "ejecutada"
target_run  = "06"

type_signal   = "eeg"
path          = f"D:\\dataset\\sub-{sub}\\ses-{ses}"
montage_path  = ".\\analysis\\ghiamp_montage.sfp"
template_path = ".\\analysis\\ica_results_template.json"
output_path   = path            # JSON / fif en la misma carpeta que los datos

# ── ICA (idéntica config que los otros scripts → mismo contrato del .fif)
ica_method         = "fastica"
ica_random_state   = 97
ica_max_iter       = "auto"
bad_channels_known = []         # canales malos conocidos a priori
n_components       = 50 - len(bad_channels_known)

# ── Selección de componentes
auto_detect        = True       # detección automática EOG/EMG sobre la propia ronda
components_manual  = []         # componentes a remover marcados a mano (índices ICA)
components_to_keep = []         # auto-detectados que son actividad cerebral (se conservan)

# ── Filtros de preprocesamiento (antes de ICA)
ica_l_freq  = 1.0
notch_freqs = [50]

# ── Umbrales de detección automática (candidatos; validar y editar 'final')
eog_measure        = "correlation"   # 'correlation' (|r|) o 'zscore'
eog_threshold      = 0.5             # |r| mínimo de correlación con EOG1/EOG2
emg_measure        = "correlation"
emg_corr_threshold = 0.5             # |r| mínimo de correlación con EMG1
muscle_threshold   = 0.85            # find_bads_muscle: 0.5 sobre-detecta; 0.85-0.9 focal
top_n_report       = 8

# ── Flujo / salida
apply_ica  = True              # True → 2da pasada: aplica ICA y guarda clean_raw.fif
save_clean = False

# ── Gráficos
dpi               = 300
show_figs         = False       # mostrar figuras interactivas
save_figs         = True       # guardar PNG en figs_dir con dpi=dpi
show_final_overlay     = True
show_final_properties  = False
figs_dir          = os.path.join(path, "ica_figs")

overlay_auto_ylim        = True
overlay_ylim_percentiles = (1, 90)
overlay_ylim_margin      = None#0.10
overlay_start            = None#1.5
overlay_stop             = None
plot_filter_l_freq       = 1.0
plot_filter_h_freq       = 40.0

scalings = {'eeg': 20, 'emg': 150, 'eog': 150}
color    = {'eeg': 'steelblue', 'emg': 'forestgreen', 'eog': 'darkorange'}

# ─── Estructura de canales y montage ──────────────────────────────────────────
montage_df   = pd.read_csv(montage_path, sep="\t", header=None)
eeg_ch_names = list(montage_df[0])[:64]
ch_names     = eeg_ch_names + ["EMG1"] + ["EOG1", "EOG2"]
ch_types     = ["eeg"] * 64 + ["emg"] + ["eog"] * 2
montage      = mne.channels.read_custom_montage(montage_path)

# ─── Nombres de archivo (idénticos a los otros scripts) ───────────────────────
def _ghiamp_fname(task, run):
    return f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_{type_signal}.hdf5"

base_name     = f"sub-{sub}_ses-{ses}_task-{target_task}_run-{target_run}"
json_fname    = f"{base_name}_ica.json"
json_path     = os.path.join(output_path, json_fname)
ica_fif_fname = f"{base_name}_ica.fif"
ica_fif_path  = os.path.join(output_path, ica_fif_fname)
clean_fname   = f"{base_name}_clean_raw.fif"
clean_path    = os.path.join(output_path, clean_fname)
run_info      = f"Sub-{sub} | Ses-{ses} | Run-{target_run} | Tarea: {target_task}"


# ─── Helpers de carga / preprocesamiento ──────────────────────────────────────
def _drop_present_channels(raw, channels, context):
    present = [ch for ch in channels if ch in raw.ch_names]
    missing = [ch for ch in channels if ch not in raw.ch_names]
    if present:
        raw.drop_channels(present)
        print(f"    Canales removidos {context}: {present}")
    if missing:
        print(f"    Canales no presentes {context}: {missing}")
    return present


def load_run(task, run):
    """Carga un run (hdf5) como mne.io.RawArray con montage aplicado."""
    fname = _ghiamp_fname(task, run)
    gm    = GHiampDataManager(os.path.join(path, fname), normalize_time=True)
    rd    = gm.raw_data.swapaxes(1, 0)
    info  = mne.create_info(ch_names=ch_names, sfreq=gm.sample_rate, ch_types=ch_types)
    raw   = mne.io.RawArray(rd, info)
    raw.set_montage(montage, on_missing="ignore")
    print(f"    [{task} run-{run}] {raw.n_times} muestras | sfreq={gm.sample_rate} Hz")
    return raw


def preprocess_for_ica(raw):
    """Filtra (pasa-altos 1 Hz + notch) y referencia promedio. Mismo preproceso
    que los otros scripts → el .fif resultante es intercambiable."""
    out = raw.copy()
    _drop_present_channels(out, bad_channels_known, "antes de ICA")
    for ch_type in ("eeg", "eog", "emg"):
        if ch_type in out.get_channel_types(unique=True):
            out.filter(l_freq=ica_l_freq, h_freq=None, picks=ch_type, fir_design='firwin')
    out.notch_filter(notch_freqs)
    out.set_eeg_reference('average', projection=True)
    out.apply_proj()
    return out


def _make_ica_plot_raw(raw):
    raw_plot = raw.copy()
    for ch_type in ("eeg", "eog", "emg"):
        if ch_type in raw_plot.get_channel_types(unique=True):
            raw_plot.filter(l_freq=plot_filter_l_freq, h_freq=plot_filter_h_freq,
                            picks=ch_type, fir_design='firwin')
    return raw_plot


def _clip_overlay_ylim(fig, percentiles=(1, 99), margin=0.10):
    lower, upper = percentiles
    margin = 0.0 if margin is None else margin   # None → sin padding extra
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


def finish_fig(fig, name):
    """Guarda (dpi=dpi) y/o deja abierta la figura según save_figs / show_figs."""
    if fig is None:
        return
    figs = fig if isinstance(fig, (list, tuple)) else [fig]
    for i, f in enumerate(figs):
        if f is None:
            continue
        if save_figs:
            os.makedirs(figs_dir, exist_ok=True)
            suffix = f"_{i}" if len(figs) > 1 else ""
            fpath = os.path.join(figs_dir, f"{base_name}_single_{name}{suffix}.png")
            f.savefig(fpath, dpi=dpi, bbox_inches='tight')
            print(f"    Figura guardada: {fpath}")
        if not show_figs:
            plt.close(f)


def _idx_list(indices):
    return sorted(set(int(i) for i in indices))


def _top_candidates(scores, n=top_n_report):
    arr = np.atleast_2d(np.asarray(scores, dtype=float))
    if arr.size == 0:
        return []
    per_comp = np.max(np.abs(arr), axis=0)
    order = np.argsort(per_comp)[::-1][:n]
    return [[int(c), round(float(per_comp[c]), 4)] for c in order]


def _safe_find_muscle(ica_obj, inst, threshold, context):
    """find_bads_muscle con manejo defensivo (bug ocasional de forma en MNE)."""
    try:
        idx, sc = ica_obj.find_bads_muscle(inst, threshold=threshold)
        return _idx_list(idx), sc
    except Exception as e:
        print(f"    [aviso] find_bads_muscle falló {context}: {type(e).__name__}: {e}")
        return [], np.array([])


def _tolist(x):
    return x.tolist() if hasattr(x, 'tolist') else list(x)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ─── [1/6] Carga y preprocesamiento de la ronda ───────────────────────────────
print(f"\n[1/6] Cargando la ronda experimento ({target_task}-{target_run})...")
raw_target = preprocess_for_ica(load_run(target_task, target_run))

# ─── [2/6] Ajuste de ICA sobre la propia ronda ────────────────────────────────
print(f"\n[2/6] Ajustando ICA ({n_components} componentes, método={ica_method})...")
ica = ICA(n_components=n_components, method=ica_method,
          max_iter=ica_max_iter, random_state=ica_random_state)
ica.fit(raw_target, picks='eeg')
print(ica)
ica.save(ica_fif_path, overwrite=True)
print(f"    ICA guardada en: {ica_fif_path}")

# ─── [3/6] Detección automática de artefactos (sobre la misma ronda) ──────────
eog_indices, eog_scores = [], np.array([])
muscle_indices, muscle_scores = [], np.array([])
emg_corr_indices, emg_corr_scores = [], np.array([])
eog_top, muscle_top, emg_corr_top = [], [], []

if auto_detect:
    print("\n[3/6] Detección automática EOG/EMG sobre la propia ronda...")
    # EOG: correlación con los canales EOG1/EOG2 de la ronda
    eog_indices, eog_scores = ica.find_bads_eog(
        raw_target, ch_name=['EOG1', 'EOG2'], threshold=eog_threshold, measure=eog_measure)
    eog_indices = _idx_list(eog_indices)
    eog_top = _top_candidates(eog_scores)
    print(f"    EOG (thr={eog_threshold}): {eog_indices} | top |corr|: {eog_top}")

    # Músculo: criterio espectral/espacial sobre la ronda
    muscle_indices, muscle_scores = _safe_find_muscle(ica, raw_target, muscle_threshold, "en la ronda")
    muscle_top = _top_candidates(muscle_scores)
    print(f"    Músculo espectral (thr={muscle_threshold}): {muscle_indices} | top: {muscle_top}")

    # EMG: correlación con el electrodo EMG1 de la ronda
    emg_corr_indices, emg_corr_scores = ica.find_bads_eog(
        raw_target, ch_name='EMG1', threshold=emg_corr_threshold, measure=emg_measure)
    emg_corr_indices = _idx_list(emg_corr_indices)
    emg_corr_top = _top_candidates(emg_corr_scores)
    print(f"    EMG corr (thr={emg_corr_threshold}): {emg_corr_indices} | top: {emg_corr_top}")
else:
    print("\n[3/6] auto_detect=False — sin detección automática (modo manual).")

all_eog_indices    = eog_indices
all_muscle_indices = sorted(set(muscle_indices) | set(emg_corr_indices))
auto_detected      = sorted(set(all_eog_indices) | set(all_muscle_indices))
manual_list        = _idx_list(components_manual)
final_components   = sorted((set(auto_detected) | set(manual_list)) - set(components_to_keep))
ica.exclude        = final_components
if manual_list:
    print(f"    Componentes manuales: {manual_list}")
if components_to_keep:
    print(f"    Conservados (cerebral): {components_to_keep}")
print(f"    -> Componentes a remover (final = auto ∪ manual − keep): {final_components}")
if auto_detect and len(final_components) > ica.n_components_ / 3:
    print(f"    [AVISO] {len(final_components)}/{ica.n_components_} componentes marcados: los "
          f"umbrales pueden ser permisivos. Revisá los gráficos y editá "
          f"'components_to_exclude.final' en el JSON antes de aplicar (apply_ica=True).")

# ─── [4/6] Gráficas de inspección ─────────────────────────────────────────────
if show_figs or save_figs:
    print("\n[4/6] Generando gráficas de inspección...")
    raw_plot = _make_ica_plot_raw(raw_target)

    n_found   = int(ica.n_components_)
    split_idx = int(np.ceil(n_found / 2))
    finish_fig(ica.plot_components(picks=range(0, split_idx),
               title="Topomapas ICA — 1ª mitad"), "topomaps_1")
    finish_fig(ica.plot_components(picks=range(split_idx, n_found),
               title="Topomapas ICA — 2ª mitad"), "topomaps_2")

    if auto_detect:
        if len(eog_scores) > 0:
            finish_fig(ica.plot_scores(eog_scores, exclude=eog_indices,
                       title=f"Scores EOG — {target_task}-{target_run}"), "scores_eog")
        if len(muscle_scores) > 0:
            finish_fig(ica.plot_scores(muscle_scores, exclude=muscle_indices,
                       title="Scores musculares (espectral)"), "scores_muscle")
        if len(emg_corr_scores) > 0:
            finish_fig(ica.plot_scores(emg_corr_scores, exclude=emg_corr_indices,
                       title="Scores correlación EMG1"), "scores_emg_corr")

    if final_components:
        finish_fig(ica.plot_properties(raw_plot, picks=final_components, show=False),
                   "properties")
        fig_ov = ica.plot_overlay(raw_plot, exclude=final_components,
                                  start=overlay_start, stop=overlay_stop,
                                  title=f"Overlay candidatos — {run_info}")
        if overlay_auto_ylim:
            _clip_overlay_ylim(fig_ov, overlay_ylim_percentiles, overlay_ylim_margin)
        finish_fig(fig_ov, "overlay")

    # Series de tiempo (interactivo: click para incluir/excluir → modo manual).
    if show_figs:
        ica.plot_sources(raw_plot, title="Series de tiempo de componentes ICA")
else:
    print("\n[4/6] show_figs=False y save_figs=False — se omiten gráficas.")

# ─── [5/6] Guardar JSON (mismo contrato + traza de método) ────────────────────
print(f"\n[5/6] Guardando resultados en: {json_path}")

with open(template_path, 'r', encoding='utf-8') as f:
    result = json.load(f)

result["method"] = "single_run"
result["metadata"].update({
    "subject": sub, "session": ses, "task": target_task, "run": target_run,
    "analysis_date": datetime.datetime.now().isoformat(),
    "source_file": _ghiamp_fname(target_task, target_run),
})
result["fit"].update({
    "strategy": "single_run",
    "fit_runs": [f"{target_task}-{target_run}"],
    "eog_run": None, "emg_run": None,
    "n_samples": int(raw_target.n_times), "sfreq": float(raw_target.info['sfreq']),
})
result["preprocessing"].update({
    "eeg_filter_l_freq": ica_l_freq, "eeg_filter_h_freq": None,
    "eog_filter_l_freq": ica_l_freq, "eog_filter_h_freq": None,
    "emg_filter_l_freq": ica_l_freq, "emg_filter_h_freq": None,
})
result["bad_channels"]["auto_detected"] = bad_channels_known
result["ica_settings"].update({
    "n_components": n_components, "method": ica_method,
    "random_state": ica_random_state, "max_iter": str(ica_max_iter),
    "n_components_found": int(ica.n_components_),
})
result["auto_detected_components"]["eog"].update({
    "indices": all_eog_indices, "scores": _tolist(eog_scores),
    "threshold": eog_threshold, "top_candidates": eog_top, "source": "self",
})
result["auto_detected_components"]["muscle"].update({
    "indices": all_muscle_indices, "scores": _tolist(muscle_scores),
    "spectral_indices": muscle_indices, "threshold": muscle_threshold,
    "top_candidates": muscle_top, "source": "self",
    "emg_corr_indices": list(emg_corr_indices), "emg_corr_scores": _tolist(emg_corr_scores),
    "emg_corr_threshold": emg_corr_threshold, "emg_corr_top_candidates": emg_corr_top,
})
result["components_to_exclude"].update({
    "auto_detected": auto_detected, "kept_from_auto": components_to_keep,
    "auto": sorted(set(auto_detected) - set(components_to_keep)),
    "manual": manual_list, "final": final_components,
})
result["ica_file"]   = ica_fif_fname
result["clean_file"] = clean_fname

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
print(f"    JSON guardado. Próximos pasos:")
print(f"    1. Revisar {json_fname} y ajustar 'components_to_exclude.manual/final' y 'notes'.")
print(f"    2. Poner apply_ica=True y volver a correr para aplicar y guardar el EEG limpio.")

# ─── [6/6] Aplicación de ICA y guardado del EEG limpio (Pasada 2) ──────────────
if apply_ica:
    print(f"\n[6/6] Aplicando ICA a la ronda {target_task}-{target_run}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        ica_params = json.load(f)

    final_apply = ica_params["components_to_exclude"]["final"]
    print(f"    Componentes a eliminar (final): {final_apply}")

    raw_clean = raw_target.copy()
    ica.exclude = final_apply
    ica.apply(raw_clean)

    if show_figs or save_figs:
        raw_before_plot = _make_ica_plot_raw(raw_target)
        raw_after_plot  = _make_ica_plot_raw(raw_clean)

        if show_final_overlay and final_apply:
            fig_ov = ica.plot_overlay(raw_before_plot, exclude=final_apply,
                                      start=overlay_start, stop=overlay_stop,
                                      title=f"Overlay — antes/después ICA\n{run_info}")
            if overlay_auto_ylim:
                _clip_overlay_ylim(fig_ov, overlay_ylim_percentiles, overlay_ylim_margin)
            finish_fig(fig_ov, "final_overlay")
        if show_final_properties and final_apply:
            finish_fig(ica.plot_properties(raw_before_plot, picks=final_apply, show=False),
                       "final_properties")

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
        ax.set_title(f'Comparación PSD — EEG (single-run)\n{run_info}')
        ax.legend()
        fig_psd.tight_layout()
        finish_fig(fig_psd, "final_psd")

        if show_figs:
            raw_before_plot.plot(scalings=scalings, color=color,
                                 title=f"Señal antes de ICA — {run_info}", duration=40)
            raw_after_plot.plot(scalings=scalings, color=color,
                                title=f"Señal después de ICA — {run_info}", duration=40)

    if save_clean:
        raw_clean.save(clean_path, overwrite=True)
        print(f"    EEG limpio guardado en: {clean_path}")

    ica_params["ica_applied"] = True
    ica_params["components_to_exclude"]["applied"] = final_apply
    ica_params["clean_file"] = clean_fname if save_clean else None
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(ica_params, f, indent=2, ensure_ascii=False)
    print(f"    JSON actualizado: ica_applied=true")

    if show_figs:
        plt.show()
else:
    print("\n[6/6] apply_ica=False — se omite la aplicación de ICA.")
    print("    Revisá el JSON, ajustá 'final' y poné apply_ica=True para aplicar.")

if show_figs:
    plt.show()
