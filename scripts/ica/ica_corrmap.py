"""
Remoción de artefactos por ICA INDEPENDIENTE por ronda + corrmap.

Alternativa a ica_preprocessing.py (que ajusta UNA ICA conjunta concatenando
varias rondas). Aquí NO se concatena nada: cada registro obtiene su propia
descomposición ICA y las rondas de calibración se usan como PLANTILLA para
identificar componentes espacialmente equivalentes en la ronda de tarea:

  1. Se ajusta una ICA para la ronda EOG, otra para la ronda EMG y otra para la
     ronda objetivo (ejecutada/imaginada).
  2. En cada ronda de calibración se elige el/los componente(s) de artefacto
     (EOG por correlación con EOG1/EOG2; EMG por criterio muscular espectral).
  3. mne.preprocessing.corrmap() usa esas topografías como plantilla para hallar
     los componentes con topografía similar en la ICA de la ronda objetivo.
  4. Esos componentes (en la ICA propia del objetivo) se excluyen y se limpia
     la señal. NO se aplica la matriz de la calibración sobre la tarea.

Compatibilidad: guarda EXACTAMENTE los mismos archivos que ica_preprocessing.py
(sub-..._ica.fif, sub-..._ica.json, sub-..._clean_raw.fif) con la misma
estructura de JSON, de modo que el código aguas abajo (analysis.ica_apply.
ICAApplicator) los consume sin distinguir qué técnica los generó. El .fif
guardado es la ICA propia de la ronda objetivo (la que ICAApplicator aplica
directamente). Un campo extra 'method'='corrmap_per_round' deja traza de la
técnica sin romper el contrato.

Nota: la señal EEG vive en los .hdf5 (GHiampDataManager); los .xdf solo traen
marcadores, por eso la ICA usa solo los .hdf5.

Flujo de dos pasadas (igual que ica_preprocessing.py):
  Pasada 1 (apply_ica=False): ajusta las ICAs, corre corrmap, detecta candidatos,
    grafica y escribe el JSON. El usuario edita 'components_to_exclude'.
  Pasada 2 (apply_ica=True): aplica la ICA del objetivo y guarda -clean_raw.fif.
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
from mne.preprocessing import ICA, corrmap

# ─── Parámetros de configuración ─────────────────────────────────────────────
sub  = "01"
ses  = "02"

# Run objetivo: el que se descompone, limpia y guarda.
target_task = "ejecutada"
target_run  = "06"

# Rondas de calibración (cada una con su propia ICA).
eog_run = "03"            # task-eog: movimientos oculares intencionales
emg_run = "02"            # task-emg: contracciones musculares intencionales

type_signal   = "eeg"
path          = f"D:\\dataset\\sub-{sub}\\ses-{ses}"
montage_path  = ".\\analysis\\ghiamp_montage.sfp"
template_path = ".\\analysis\\ica_results_template.json"
output_path   = path            # JSON / fif en la misma carpeta que los datos

# ── ICA (idéntica config que ica_preprocessing.py → mismo contrato del .fif)
ica_method         = "fastica"
ica_random_state   = 97
ica_max_iter       = "auto"
bad_channels_known = []         # canales malos conocidos a priori
n_components       = 30 - len(bad_channels_known)

# Componentes detectados que en realidad parecen actividad cerebral: se conservan.
components_to_keep = []

# ── Filtros de preprocesamiento (antes de ICA)
ica_l_freq  = 1.0
notch_freqs = [50]

# ── Selección de plantillas en las rondas de calibración
eog_measure   = "correlation"   # 'correlation' (|r|) o 'zscore'
eog_threshold = 0.5             # |r| mínimo para tomar un componente EOG como plantilla
n_templates_eog = 2             # nº de plantillas EOG (p.ej. vertical y horizontal)
n_templates_emg = 2             # nº de plantillas EMG (por score muscular)

# ── corrmap: umbral de similitud de topografías entre ICAs.
# Como cada ICA se ajusta por separado, la similitud de mapas para una misma
# fuente fisiológica ronda 0.75-0.8 (no ~0.9). 'auto' aquí resulta demasiado
# estricto. 0.75 captura el componente EOG del objetivo de forma focal.
corrmap_threshold = 0.75        # |corr| de mapas; 'auto' deja que MNE lo estime
corrmap_threshold = float(corrmap_threshold) if corrmap_threshold != "auto" else "auto"

# ── Flujo / salida
apply_ica  = True              # True → 2da pasada: aplica ICA y guarda clean_raw.fif
save_clean = False

# ── Gráficos
dpi               = 300
show_figs         = False       # mostrar figuras interactivas
save_figs         = True       # guardar PNG en figs_dir con dpi=dpi
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

# ─── Nombres de archivo (idénticos a ica_preprocessing.py) ────────────────────
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
    que ica_preprocessing.py → el .fif resultante es intercambiable."""
    out = raw.copy()
    _drop_present_channels(out, bad_channels_known, "antes de ICA")
    for ch_type in ("eeg", "eog", "emg"):
        if ch_type in out.get_channel_types(unique=True):
            out.filter(l_freq=ica_l_freq, h_freq=None, picks=ch_type, fir_design='firwin')
    out.notch_filter(notch_freqs)
    out.set_eeg_reference('average', projection=True)
    out.apply_proj()
    return out


def fit_ica(raw, label):
    """Ajusta una ICA (picks='eeg') sobre un registro ya preprocesado."""
    ica = ICA(n_components=n_components, method=ica_method,
              max_iter=ica_max_iter, random_state=ica_random_state)
    print(f"    Ajustando ICA de la ronda {label}...")
    ica.fit(raw, picks='eeg')
    return ica


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
        pad = abs(y_min) * margin if y_min == y_max and y_min != 0 else \
            (margin if y_min == y_max else (y_max - y_min) * margin)
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
            fpath = os.path.join(figs_dir, f"{base_name}_corrmap_{name}{suffix}.png")
            f.savefig(fpath, dpi=dpi, bbox_inches='tight')
            print(f"    Figura guardada: {fpath}")
        if not show_figs:
            plt.close(f)


def _idx_list(indices):
    return sorted(set(int(i) for i in indices))


def _select_templates(scores, n, min_abs):
    """Top-n componentes por |score| con piso min_abs. Si ninguno alcanza el
    piso, devuelve igual el top-1 (para no quedarse sin plantilla) con aviso.
    scores: 1-D (n_comp) o 2-D (canales × n_comp)."""
    arr = np.atleast_2d(np.asarray(scores, dtype=float))
    if arr.size == 0:
        return [], []
    per_comp = np.max(np.abs(arr), axis=0)
    order = np.argsort(per_comp)[::-1]
    passing = [int(c) for c in order if per_comp[c] >= min_abs][:n]
    if not passing:
        passing = [int(order[0])]
        print(f"    [aviso] ningún componente alcanza |score|>={min_abs}; "
              f"se usa el top-1 (comp {passing[0]}, |score|={per_comp[passing[0]]:.3f}).")
    ranked = [[int(c), round(float(per_comp[c]), 4)] for c in order[:max(n, len(passing))]]
    return passing, ranked


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


# ─── [1/6] Carga y preprocesamiento de cada ronda (independiente) ─────────────
print(f"\n[1/6] Cargando rondas (sub-{sub} ses-{ses}) — ICA independiente por ronda...")
raw_target = preprocess_for_ica(load_run(target_task, target_run))
raw_eog    = preprocess_for_ica(load_run("eog", eog_run))
raw_emg    = preprocess_for_ica(load_run("emg", emg_run))

# ─── [2/6] Ajuste de una ICA por ronda ────────────────────────────────────────
print("\n[2/6] Ajustando una ICA por ronda...")
ica_target = fit_ica(raw_target, f"objetivo ({target_task}-{target_run})")
ica_eog    = fit_ica(raw_eog,    f"EOG ({eog_run})")
ica_emg    = fit_ica(raw_emg,    f"EMG ({emg_run})")
ica_target.save(ica_fif_path, overwrite=True)
print(f"    ICA de la ronda objetivo guardada en: {ica_fif_path}")

# ─── [3/6] Selección de plantillas en las rondas de calibración ───────────────
print("\n[3/6] Eligiendo componentes plantilla en EOG y EMG...")

# EOG: componentes correlacionados con EOG1/EOG2 dentro de la ICA del EOG run
eog_cal_idx, eog_cal_scores = ica_eog.find_bads_eog(
    raw_eog, ch_name=['EOG1', 'EOG2'], threshold=eog_threshold, measure=eog_measure)
eog_templates, eog_ranked = _select_templates(eog_cal_scores, n_templates_eog, eog_threshold)
print(f"    Plantillas EOG (comp en ICA_eog): {eog_templates} | ranking |corr|: {eog_ranked}")

# EMG: componentes con mayor score muscular dentro de la ICA del EMG run
try:
    emg_muscle_idx, emg_muscle_scores = ica_emg.find_bads_muscle(raw_emg)
except Exception as e:
    print(f"    [aviso] find_bads_muscle falló en EMG: {type(e).__name__}: {e}")
    emg_muscle_scores = np.array([])
emg_templates, emg_ranked = _select_templates(emg_muscle_scores, n_templates_emg, 0.0)
print(f"    Plantillas EMG (comp en ICA_emg): {emg_templates} | ranking score: {emg_ranked}")

# ─── [4/6] corrmap: hallar componentes equivalentes en la ICA objetivo ────────
print("\n[4/6] Corriendo corrmap (plantillas de calibración -> ICA objetivo)...")
plot_corrmap = show_figs or save_figs


def _run_corrmap(cal_ica, templates, label):
    """Corre corrmap por cada plantilla y devuelve el set de componentes de la
    ICA objetivo (índice 1 de la lista) que matchean."""
    icas = [cal_ica, ica_target]
    matched = set()
    for k, comp in enumerate(templates):
        lab = f"{label}_{k}"
        # corrmap devuelve (template_fig, labelled_ics) si plot=True; None si plot=False.
        res = corrmap(
            icas, template=(0, comp), threshold=corrmap_threshold, label=lab,
            ch_type='eeg', plot=plot_corrmap, show=False)
        hits = ica_target.labels_.get(lab, [])
        matched.update(int(i) for i in hits)
        print(f"    [{label}] plantilla comp {comp} -> objetivo: {sorted(int(i) for i in hits)}")
        if plot_corrmap and res is not None:
            tfig, lfig = res
            finish_fig(tfig, f"{label}_tmpl{comp}_template")
            finish_fig(lfig, f"{label}_tmpl{comp}_matches")
    return sorted(matched)


eog_components = _run_corrmap(ica_eog, eog_templates, "eog")
emg_components = _run_corrmap(ica_emg, emg_templates, "muscle")

auto_detected = sorted(set(eog_components) | set(emg_components))
auto_excluded = sorted(set(auto_detected) - set(components_to_keep))
print(f"    Componentes objetivo EOG : {eog_components}")
print(f"    Componentes objetivo EMG : {emg_components}")
if components_to_keep:
    print(f"    Conservados (cerebral)   : {components_to_keep}")
print(f"    -> Componentes auto-excluidos (candidatos): {auto_excluded}")
if not auto_excluded:
    print("    [AVISO] corrmap no encontró coincidencias. Bajá corrmap_threshold "
          "o revisá las plantillas; editá 'components_to_exclude.final' a mano.")

# ─── Gráficas de inspección de la ICA objetivo ────────────────────────────────
if show_figs or save_figs:
    raw_target_plot = _make_ica_plot_raw(raw_target)
    n_found   = int(ica_target.n_components_)
    split_idx = int(np.ceil(n_found / 2))
    finish_fig(ica_target.plot_components(picks=range(0, split_idx),
               title="Topomapas ICA objetivo — 1ª mitad"), "target_topomaps_1")
    finish_fig(ica_target.plot_components(picks=range(split_idx, n_found),
               title="Topomapas ICA objetivo — 2ª mitad"), "target_topomaps_2")
    if auto_excluded:
        finish_fig(ica_target.plot_properties(raw_target_plot, picks=auto_excluded, show=False),
                   "target_properties")
        fig_ov = ica_target.plot_overlay(raw_target_plot, exclude=auto_excluded,
                                         start=overlay_start, stop=overlay_stop,
                                         title=f"Overlay candidatos — {run_info}")
        if overlay_auto_ylim:
            _clip_overlay_ylim(fig_ov, overlay_ylim_percentiles, overlay_ylim_margin)
        finish_fig(fig_ov, "target_overlay")
    if show_figs:
        ica_target.plot_sources(raw_target_plot, title="Series de tiempo — ICA objetivo")

# ─── [5/6] Guardar JSON (mismo contrato + traza de método) ────────────────────
print(f"\n[5/6] Guardando resultados en: {json_path}")

with open(template_path, 'r', encoding='utf-8') as f:
    result = json.load(f)

result["method"] = "corrmap_per_round"
result["metadata"].update({
    "subject": sub, "session": ses, "task": target_task, "run": target_run,
    "analysis_date": datetime.datetime.now().isoformat(),
    "source_file": _ghiamp_fname(target_task, target_run),
})
result["fit"].update({
    "strategy": "corrmap_per_round",
    "fit_runs": [f"{target_task}-{target_run} (ICA propia)"],
    "eog_run": eog_run, "emg_run": emg_run,
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
    "n_components_found": int(ica_target.n_components_),
})
result["corrmap"] = {
    "threshold": corrmap_threshold,
    "eog_templates_in_cal": eog_templates, "eog_template_ranking": eog_ranked,
    "emg_templates_in_cal": emg_templates, "emg_template_ranking": emg_ranked,
    "eog_matches_in_target": eog_components,
    "muscle_matches_in_target": emg_components,
}
result["auto_detected_components"]["eog"].update({
    "indices": eog_components, "scores": _tolist(eog_cal_scores),
    "threshold": eog_threshold, "source": "corrmap(eog_run -> target)",
})
result["auto_detected_components"]["muscle"].update({
    "indices": emg_components, "scores": _tolist(emg_muscle_scores),
    "source": "corrmap(emg_run -> target)",
})
result["components_to_exclude"].update({
    "auto_detected": auto_detected, "kept_from_auto": components_to_keep,
    "auto": auto_excluded, "final": auto_excluded,
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
    print(f"\n[6/6] Aplicando ICA de la ronda objetivo ({target_task}-{target_run})...")
    with open(json_path, 'r', encoding='utf-8') as f:
        ica_params = json.load(f)

    final_components = ica_params["components_to_exclude"]["final"]
    print(f"    Componentes a eliminar (final): {final_components}")

    raw_clean = raw_target.copy()
    ica_target.exclude = final_components
    ica_target.apply(raw_clean)

    if show_figs or save_figs:
        raw_before_plot = _make_ica_plot_raw(raw_target)
        raw_after_plot  = _make_ica_plot_raw(raw_clean)

        if show_final_overlay and final_components:
            fig_ov = ica_target.plot_overlay(raw_before_plot, exclude=final_components,
                                             start=overlay_start, stop=overlay_stop,
                                             title=f"Overlay — antes/después ICA\n{run_info}")
            if overlay_auto_ylim:
                _clip_overlay_ylim(fig_ov, overlay_ylim_percentiles, overlay_ylim_margin)
            finish_fig(fig_ov, "final_overlay")
        if show_final_properties and final_components:
            finish_fig(ica_target.plot_properties(raw_before_plot, picks=final_components,
                       show=False), "final_properties")

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
        ax.set_title(f'Comparación PSD — EEG (corrmap)\n{run_info}')
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
    ica_params["components_to_exclude"]["applied"] = final_components
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
