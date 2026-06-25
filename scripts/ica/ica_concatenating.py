"""
Preprocesamiento ICA conjunto por sesión para limpieza de artefactos en EEG.

Estrategia (ver scripts/ica/prompt_ica.txt y docsAndRefs/links.txt):
  En lugar de ajustar ICA sobre un único run y aplicarlo a otro, se ajusta una
  ICA CONJUNTA concatenando varios runs representativos de la sesión:
    - uno o varios runs ejecutados,
    - uno o varios runs imaginados,
    - segmentos representativos (recortados) de la ronda EOG,
    - segmentos representativos (recortados) de la ronda EMG.
  La matriz de separación se estima así sobre datos que contienen la tarea real.
  Luego se IDENTIFICAN los componentes de artefacto (como CANDIDATOS rankeados;
  la exclusión final la valida el usuario visual y cuantitativamente):
    - find_bads_eog() sobre la ronda EOG (correlación |r| con EOG1/EOG2),
    - find_bads_muscle() sobre el FIT representativo (criterio espectral/espacial;
      sobre la ronda EMG pura sobre-detecta porque casi todo parece muscular),
    - find_bads_eog(EMG1) sobre la ronda EMG (correlación con el electrodo EMG),
    - se valida que esos componentes también aparezcan en el run objetivo.

Nota sobre formatos: la señal EEG vive en los .hdf5 (GHiampDataManager). Los
.xdf solo contienen marcadores de tablet/laptop (LSLDataManager), sin stream
EEG, por lo que la ICA se ajusta exclusivamente con los .hdf5.

Flujo de trabajo en dos pasadas:
  Pasada 1 (apply_ica=False): ajusta ICA conjunta, detecta artefactos con las
    rondas de calibración, muestra/guarda gráficas y escribe un JSON BIDS-like.
  Edición manual: el usuario edita el JSON (manual/final/notes).
  Pasada 2 (apply_ica=True): lee el JSON, aplica ICA al run objetivo, grafica
    antes/después y guarda el EEG limpio (-clean_raw.fif).
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

# Run objetivo: al que se le aplica la ICA y se guarda el EEG limpio.
target_task = "ejecutada"
target_run  = "07"

# Runs para AJUSTAR la ICA conjunta (subconjunto representativo de la sesión).
fit_ejecutada = ["05"]          # uno o varios runs ejecutados
fit_imaginada = ["10"]          # uno o varios runs imaginados
eog_run       = "03"            # task-eog: movimientos oculares intencionales
emg_run       = "02"            # task-emg: contracciones musculares intencionales

# Segmentos representativos de calibración para el fit (segundos; None = run completo).
eog_crop = (None, None)         # ej. (5.0, 60.0): ventana con más movimientos oculares
emg_crop = (None, None)         # ej. (5.0, 60.0): ventana con más contracción

type_signal   = "eeg"
path          = f"D:\\dataset\\sub-{sub}\\ses-{ses}"
montage_path  = ".\\analysis\\ghiamp_montage.sfp"
template_path = ".\\analysis\\ica_results_template.json"
output_path   = path            # dónde se guardan JSON / fif (misma carpeta que los datos)

# ── ICA
ica_method         = "fastica"
ica_random_state   = 97
ica_max_iter       = "auto"
bad_channels_known = []         # canales malos conocidos a priori
n_components       = 30 - len(bad_channels_known)   # int o varianza acumulada (0-1)

# Componentes auto-detectados que en realidad parecen actividad cerebral.
# Estos se conservan: se excluyen del conjunto final a remover.
components_to_keep = []         # ej. [9, 28, 25]

# ── Filtros de preprocesamiento (antes de ICA)
ica_l_freq  = 1.0               # pasa-altos común para eeg/eog/emg antes de ICA
notch_freqs = [50]

# ── Umbrales de detección automática (candidatos; el usuario valida y edita 'final')
# Para rondas de calibración dedicadas conviene 'correlation' (umbral = |r| absoluto,
# físicamente interpretable) en vez de 'zscore' (que aquí no discrimina bien).
eog_measure        = "correlation"   # 'correlation' (|r|) o 'zscore'
eog_threshold      = 0.5             # |r| mínimo de correlación con EOG1/EOG2
emg_measure        = "correlation"
emg_corr_threshold = 0.5             # |r| mínimo de correlación con EMG1
muscle_threshold   = 0.85       # find_bads_muscle: 0.5 sobre-detecta; 0.85-0.9 es focal
top_n_report       = 8          # nº de candidatos rankeados a reportar/guardar por criterio

# ── Flujo / salida
apply_ica  = True              # True → 2da pasada: aplica ICA y guarda clean_raw.fif
save_clean = False               # guardar el EEG limpio del run objetivo

# ── Gráficos
dpi               = 300
show_figs         = True       # mostrar figuras interactivas (inspección pasada 1)
save_figs         = False       # guardar PNG en figs_dir con dpi=dpi
show_final_overlay     = False
show_final_properties  = False
figs_dir          = os.path.join(path, "ica_figs")

overlay_auto_ylim        = True
overlay_ylim_percentiles = (1, 90)
overlay_ylim_margin      = 0.10
overlay_start            = 1.5
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

# ─── Nombres de archivo (output ligado al run objetivo) ───────────────────────
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
    rd    = gm.raw_data.swapaxes(1, 0)          # → canales × muestras
    info  = mne.create_info(ch_names=ch_names, sfreq=gm.sample_rate, ch_types=ch_types)
    raw   = mne.io.RawArray(rd, info)
    raw.set_montage(montage, on_missing="ignore")
    print(f"    [{task} run-{run}] {raw.n_times} muestras | sfreq={gm.sample_rate} Hz")
    return raw


def preprocess_for_ica(raw, crop=None):
    """Filtra (pasa-altos 1 Hz + notch), referencia promedio y opcionalmente
    recorta. Devuelve una copia lista para concatenar / ajustar ICA."""
    out = raw.copy()
    _drop_present_channels(out, bad_channels_known, "antes de ICA")
    for ch_type in ("eeg", "eog", "emg"):
        if ch_type in out.get_channel_types(unique=True):
            out.filter(l_freq=ica_l_freq, h_freq=None, picks=ch_type, fir_design='firwin')
    out.notch_filter(notch_freqs)
    out.set_eeg_reference('average', projection=True)
    out.apply_proj()
    if crop is not None and (crop[0] is not None or crop[1] is not None):
        tmax_data = out.times[-1]
        tmin = 0.0 if crop[0] is None else max(0.0, crop[0])
        tmax = tmax_data if crop[1] is None else min(tmax_data, crop[1])
        out.crop(tmin=tmin, tmax=tmax)
        print(f"    Recorte aplicado: {tmin:.2f}-{tmax:.2f} s")
    return out


def _make_ica_plot_raw(raw):
    """Copia filtrada (1-40 Hz) solo para gráficas ICA."""
    raw_plot = raw.copy()
    present_types = raw_plot.get_channel_types(unique=True)
    for ch_type in ("eeg", "eog", "emg"):
        if ch_type in present_types:
            raw_plot.filter(l_freq=plot_filter_l_freq, h_freq=plot_filter_h_freq,
                            picks=ch_type, fir_design='firwin')
    return raw_plot


def _clip_overlay_ylim(fig, percentiles=(1, 99), margin=0.10):
    """Acota los ejes Y del overlay usando percentiles de las líneas visibles."""
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
    """Guarda (dpi=dpi) y/o deja la figura abierta según save_figs / show_figs."""
    if fig is None:
        return
    figs = fig if isinstance(fig, (list, tuple)) else [fig]
    for i, f in enumerate(figs):
        if save_figs:
            os.makedirs(figs_dir, exist_ok=True)
            suffix = f"_{i}" if len(figs) > 1 else ""
            fpath = os.path.join(figs_dir, f"{base_name}_{name}{suffix}.png")
            f.savefig(fpath, dpi=dpi, bbox_inches='tight')
            print(f"    Figura guardada: {fpath}")
        if not show_figs:
            plt.close(f)


# ─── [1/7] Carga y preprocesamiento de los runs del fit ───────────────────────
print(f"\n[1/7] Cargando runs para la ICA conjunta (sub-{sub} ses-{ses})...")
fit_runs_spec = (
    [("ejecutada", r) for r in fit_ejecutada]
    + [("imaginada", r) for r in fit_imaginada]
)
fit_raws = []
fit_runs_labels = []
for task, run in fit_runs_spec:
    fit_raws.append(preprocess_for_ica(load_run(task, run)))
    fit_runs_labels.append(f"{task}-{run}")

# Segmentos representativos de calibración (recortados) para el fit.
print("    Cargando segmentos de calibración (EOG/EMG) para el fit...")
fit_raws.append(preprocess_for_ica(load_run("eog", eog_run), crop=eog_crop))
fit_runs_labels.append(f"eog-{eog_run}(crop)")
fit_raws.append(preprocess_for_ica(load_run("emg", emg_run), crop=emg_crop))
fit_runs_labels.append(f"emg-{emg_run}(crop)")

# ─── [2/7] Concatenación y ajuste de la ICA conjunta ──────────────────────────
print(f"\n[2/7] Concatenando {len(fit_raws)} segmentos: {fit_runs_labels}")
raw_fit = mne.concatenate_raws([r.copy() for r in fit_raws])
print(f"    Total concatenado: {raw_fit.n_times} muestras "
      f"({raw_fit.n_times / raw_fit.info['sfreq']:.1f} s)")

print(f"\n    Ajustando ICA ({n_components} componentes, método={ica_method})...")
ica = ICA(n_components=n_components, method=ica_method,
          max_iter=ica_max_iter, random_state=ica_random_state)
ica.fit(raw_fit, picks='eeg')
print(ica)
ica.save(ica_fif_path, overwrite=True)
print(f"    ICA guardada en: {ica_fif_path}")

# ─── [3/7] Identificación de componentes con las rondas de calibración ────────
# Los detectores producen CANDIDATOS rankeados por score. En esta sesión las
# correlaciones EOG/EMG son débiles y find_bads_muscle sobre-detecta con su
# umbral por defecto, por eso: (a) umbrales configurables y conservadores,
# (b) find_bads_muscle se corre sobre el FIT representativo (no la ronda pura
# EMG, donde casi todo parece muscular), (c) se guardan los scores y top-N
# candidatos para que el usuario valide visual/cuantitativamente antes de excluir.
print("\n[3/7] Identificando componentes de artefacto (candidatos)...")


def _idx_list(indices):
    return sorted(set(int(i) for i in indices))


def _top_candidates(scores, n=top_n_report):
    """Top-n componentes por |score| (scores 1-D, o 2-D canales×componentes)."""
    arr = np.atleast_2d(np.asarray(scores, dtype=float))
    if arr.size == 0:
        return []
    per_comp = np.max(np.abs(arr), axis=0)
    order = np.argsort(per_comp)[::-1][:n]
    return [[int(c), round(float(per_comp[c]), 4)] for c in order]


def _safe_find_muscle(inst, threshold, context):
    """find_bads_muscle con manejo defensivo (bug ocasional de forma en MNE)."""
    try:
        idx, sc = ica.find_bads_muscle(inst, threshold=threshold)
        return _idx_list(idx), sc
    except Exception as e:
        print(f"    [aviso] find_bads_muscle falló {context}: {type(e).__name__}: {e}")
        return [], np.array([])


# EOG: ronda eog (completa) → correlación con EOG1/EOG2
raw_eog_id = preprocess_for_ica(load_run("eog", eog_run))
eog_indices, eog_scores = ica.find_bads_eog(
    raw_eog_id, ch_name=['EOG1', 'EOG2'], threshold=eog_threshold, measure=eog_measure)
eog_indices = _idx_list(eog_indices)
eog_top = _top_candidates(eog_scores)
print(f"    EOG (thr={eog_threshold}): {eog_indices} | top |corr|: {eog_top}")

# Músculo (espectral): sobre el FIT representativo, umbral focal
muscle_indices, muscle_scores = _safe_find_muscle(raw_fit, muscle_threshold, "en fit")
muscle_top = _top_candidates(muscle_scores)
print(f"    Músculo espectral en fit (thr={muscle_threshold}): {muscle_indices} | top: {muscle_top}")

# EMG (correlación): ronda emg → correlación con el electrodo EMG1
raw_emg_id = preprocess_for_ica(load_run("emg", emg_run))
emg_corr_indices, emg_corr_scores = ica.find_bads_eog(
    raw_emg_id, ch_name='EMG1', threshold=emg_corr_threshold, measure=emg_measure)
emg_corr_indices = _idx_list(emg_corr_indices)
emg_corr_top = _top_candidates(emg_corr_scores)
print(f"    EMG corr (thr={emg_corr_threshold}): {emg_corr_indices} | top: {emg_corr_top}")

all_eog_indices    = eog_indices
all_muscle_indices = sorted(set(muscle_indices) | set(emg_corr_indices))
auto_detected      = sorted(set(all_eog_indices) | set(all_muscle_indices))
auto_excluded      = sorted(set(auto_detected) - set(components_to_keep))
ica.exclude        = auto_excluded
if components_to_keep:
    print(f"    Auto-detectados: {auto_detected} | conservados: {components_to_keep}")
print(f"    -> Componentes auto-excluidos (candidatos): {auto_excluded}")
if len(auto_excluded) > ica.n_components_ / 3:
    print(f"    [AVISO] {len(auto_excluded)}/{ica.n_components_} componentes marcados: los "
          f"umbrales pueden ser permisivos. Revisá los gráficos y editá "
          f"'components_to_exclude.final' en el JSON antes de aplicar (apply_ica=True).")

# ─── [4/7] Validación: ¿esos componentes aparecen en el run objetivo? ─────────
print(f"\n[4/7] Validando presencia en el run objetivo ({target_task}-{target_run})...")
raw_target = preprocess_for_ica(load_run(target_task, target_run))
tgt_eog_indices, _    = ica.find_bads_eog(raw_target, ch_name=['EOG1', 'EOG2'],
                                          threshold=eog_threshold, measure=eog_measure)
tgt_eog_indices       = _idx_list(tgt_eog_indices)
tgt_muscle_indices, _ = _safe_find_muscle(raw_target, muscle_threshold, "en target")
target_detected = sorted(set(tgt_eog_indices) | set(tgt_muscle_indices))
overlap = sorted(set(auto_detected) & set(target_detected))
print(f"    Detectados en target  : EOG={tgt_eog_indices} músculo={tgt_muscle_indices}")
print(f"    Solapamiento con auto : {overlap}")

# ─── [5/7] Gráficas de inspección ─────────────────────────────────────────────
if show_figs or save_figs:
    print("\n[5/7] Generando gráficas de inspección...")
    raw_fit_plot = _make_ica_plot_raw(raw_fit)

    n_found   = int(ica.n_components_)
    split_idx = int(np.ceil(n_found / 2))
    finish_fig(ica.plot_components(picks=range(0, split_idx),
               title="Topomapas ICA — 1ª mitad"), "topomaps_1")
    finish_fig(ica.plot_components(picks=range(split_idx, n_found),
               title="Topomapas ICA — 2ª mitad"), "topomaps_2")

    finish_fig(ica.plot_scores(eog_scores, exclude=eog_indices,
               title=f"Scores EOG — ronda eog-{eog_run}"), "scores_eog")
    if len(muscle_scores) > 0:
        finish_fig(ica.plot_scores(muscle_scores, exclude=muscle_indices,
                   title="Scores musculares (espectral) — fit conjunto"), "scores_muscle")
    if len(emg_corr_scores) > 0:
        finish_fig(ica.plot_scores(emg_corr_scores, exclude=emg_corr_indices,
                   title=f"Scores correlación EMG1 — emg-{emg_run}"), "scores_emg_corr")

    if auto_excluded:
        finish_fig(ica.plot_properties(raw_fit_plot, picks=auto_excluded, show=False),
                   "properties_auto")
        fig_ov = ica.plot_overlay(raw_fit_plot, exclude=auto_excluded,
                                  start=overlay_start, stop=overlay_stop,
                                  title=f"Overlay auto — antes/después ICA\n{run_info}")
        if overlay_auto_ylim:
            _clip_overlay_ylim(fig_ov, overlay_ylim_percentiles, overlay_ylim_margin)
        finish_fig(fig_ov, "overlay_auto")

    # Series de tiempo (interactivo: click para incluir/excluir). Solo si show_figs.
    if show_figs:
        ica.plot_sources(raw_fit_plot, title="Series de tiempo de componentes ICA")
else:
    print("\n[5/7] show_figs=False y save_figs=False — se omiten gráficas.")

# ─── [6/7] Guardar JSON BIDS-like ─────────────────────────────────────────────
print(f"\n[6/7] Guardando resultados en: {json_path}")


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


with open(template_path, 'r', encoding='utf-8') as f:
    result = json.load(f)

result["metadata"].update({
    "subject": sub, "session": ses, "task": target_task, "run": target_run,
    "analysis_date": datetime.datetime.now().isoformat(),
    "source_file": _ghiamp_fname(target_task, target_run),
})
result["fit"].update({
    "fit_runs": fit_runs_labels,
    "eog_run": eog_run, "emg_run": emg_run,
    "eog_crop": list(eog_crop), "emg_crop": list(emg_crop),
    "n_samples": int(raw_fit.n_times), "sfreq": float(raw_fit.info['sfreq']),
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
    "threshold": eog_threshold, "top_candidates": eog_top, "source": "eog_run",
})
result["auto_detected_components"]["muscle"].update({
    "indices": all_muscle_indices, "scores": _tolist(muscle_scores),
    "spectral_indices": muscle_indices, "threshold": muscle_threshold,
    "top_candidates": muscle_top, "source": "fit",
    "emg_corr_indices": list(emg_corr_indices), "emg_corr_scores": _tolist(emg_corr_scores),
    "emg_corr_threshold": emg_corr_threshold, "emg_corr_top_candidates": emg_corr_top,
})
result["validation_on_target"].update({
    "eog_indices": sorted(set(tgt_eog_indices)),
    "muscle_indices": sorted(set(tgt_muscle_indices)),
    "overlap_with_auto": overlap,
})
result["components_to_exclude"].update({
    "auto_detected": auto_detected, "kept_from_auto": components_to_keep,
    "auto": auto_excluded, "final": auto_excluded,
})
result["ica_file"]  = ica_fif_fname
result["clean_file"] = clean_fname

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
print(f"    JSON guardado. Próximos pasos:")
print(f"    1. Revisar {json_fname} y ajustar 'components_to_exclude.manual/final' y 'notes'.")
print(f"    2. Poner apply_ica=True y volver a correr para aplicar y graficar antes/después.")

# ─── [7/7] Aplicación de ICA y guardado del EEG limpio (Pasada 2) ──────────────
if apply_ica:
    print(f"\n[7/7] Aplicando ICA al run objetivo {target_task}-{target_run}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        ica_params = json.load(f)

    final_components = ica_params["components_to_exclude"]["final"]
    bad_ch_all = (ica_params["bad_channels"]["auto_detected"]
                  + ica_params["bad_channels"]["manual"])
    print(f"    Canales malos: {bad_ch_all}")
    print(f"    Componentes a eliminar (final): {final_components}")

    raw_clean = raw_target.copy()
    ica.exclude = final_components
    ica.apply(raw_clean)

    raw_before_plot = _make_ica_plot_raw(raw_target)
    raw_after_plot  = _make_ica_plot_raw(raw_clean)

    if (show_figs or save_figs) and show_final_overlay and final_components:
        fig_ov = ica.plot_overlay(raw_before_plot, exclude=final_components,
                                  start=overlay_start, stop=overlay_stop,
                                  title=f"Overlay — antes/después ICA\n{run_info}")
        if overlay_auto_ylim:
            _clip_overlay_ylim(fig_ov, overlay_ylim_percentiles, overlay_ylim_margin)
        finish_fig(fig_ov, "overlay_final")

    if (show_figs or save_figs) and show_final_properties and final_components:
        finish_fig(ica.plot_properties(raw_before_plot, picks=final_components, show=False),
                   "properties_final")

    # PSD media EEG: antes vs. después
    if show_figs or save_figs:
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
        finish_fig(fig_psd, "psd_before_after")

    if show_figs:
        raw_before_plot.plot(scalings=scalings, color=color,
                             title=f"Señal antes de ICA — {run_info}", duration=40)
        raw_after_plot.plot(scalings=scalings, color=color,
                            title=f"Señal después de ICA — {run_info}", duration=40)

    if save_clean:
        raw_clean.save(clean_path, overwrite=True)
        print(f"    EEG limpio guardado en: {clean_path}")

    ica_params["ica_applied"] = True
    ica_params["components_to_exclude"]["applied"] = final_components
    ica_params["clean_file"] = clean_fname if save_clean else None
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(ica_params, f, indent=2, ensure_ascii=False)
    print(f"    JSON actualizado: ica_applied=true")

    if show_figs:
        plt.show()
else:
    print("\n[7/7] apply_ica=False — se omite la aplicación de ICA.")
    print("    Revisar el JSON, ajustar 'final' y poner apply_ica=True para aplicar.")

if show_figs:
    plt.show()
