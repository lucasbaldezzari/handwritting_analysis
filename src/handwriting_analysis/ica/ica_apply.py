"""
Aplica la solución ICA guardada en un JSON de ica_preprocessing.py a un registro EEG.

Uso típico:
    from analysis.ica_apply import ICAApplicator

    cleaner   = ICAApplicator("ruta/al/sub-XX_..._ica.json")
    cleaner.load_and_fit("ruta/al/sub-XX_..._eeg.hdf5")
    raw_clean = cleaner.apply()          # devuelve mne.io.RawArray limpio
    cleaner.plot_comparison()            # overlay, PSD y señal scrollable
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyhwr.managers import GHiampDataManager
import mne
from mne.preprocessing import ICA


class ICAApplicator:
    """
    Carga los parámetros de un JSON producido por ica_preprocessing.py,
    re-ajusta ICA sobre el registro indicado (determinístico: mismo random_state)
    y aplica la limpieza según la lista de componentes del JSON.

    Métodos:
        load_and_fit(hdf5_path)  — carga datos, preprocesa y re-ajusta ICA
        apply(exclude_from)      — aplica exclusión y devuelve señal limpia
        plot_comparison()        — overlay, PSD y señal scrollable
        apply_to_raw(raw)        — aplica ICA in-place a un Raw ya cargado (para scripts de análisis)
    """

    _MONTAGE_PATH = ".\\analysis\\ghiamp_montage.sfp"

    def __init__(self, json_path: str):
        self._json_path  = json_path
        with open(json_path, 'r', encoding='utf-8') as f:
            self.params = json.load(f)
        self._raw_signal = None
        self._filt_raw   = None
        self._raw_clean  = None
        self._ica        = None

    # ── Carga y ajuste ────────────────────────────────────────────────────────

    def load_and_fit(self, hdf5_path: str,
                     montage_path: str = None) -> None:
        """
        Carga el HDF5, preprocesa con los parámetros del JSON y re-ajusta ICA.

        hdf5_path   : ruta completa al archivo .hdf5 del registro a limpiar
        montage_path: ruta al .sfp (usa _MONTAGE_PATH si no se especifica)
        """
        montage_path = montage_path or self._MONTAGE_PATH

        # ── Montaje y nombres de canales
        montage_df   = pd.read_csv(montage_path, sep="\t", header=None)
        eeg_ch_names = list(montage_df[0])[:64]
        ch_names     = eeg_ch_names + ["EMG1"] + ["EOG1", "EOG2"]
        ch_types     = ["eeg"] * 64 + ["emg"] + ["eog"] * 2

        montage = mne.channels.read_custom_montage(montage_path)

        # ── Carga de datos
        print(f"Cargando: {os.path.basename(hdf5_path)}")
        gmanager = GHiampDataManager(hdf5_path, normalize_time=True)
        raw_data = gmanager.raw_data.swapaxes(1, 0)
        sfreq    = gmanager.sample_rate

        info             = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        self._raw_signal = mne.io.RawArray(raw_data, info)
        self._raw_signal.set_montage(montage, on_missing="ignore")
        print(f"    {self._raw_signal.n_times} muestras | {len(ch_names)} canales | {sfreq} Hz")

        # ── Preprocesamiento (igual que en ica_preprocessing.py)
        bad_channels = (
            self.params["bad_channels"]["auto_detected"]
            + self.params["bad_channels"]["manual"]
        )
        p = self.params["preprocessing"]

        self._filt_raw = self._raw_signal.copy()
        self._filt_raw.filter(l_freq=p["eeg_filter_l_freq"], h_freq=p["eeg_filter_h_freq"],
                               picks='eeg', fir_design=p["fir_design"])
        self._filt_raw.filter(l_freq=p["eog_filter_l_freq"], h_freq=p["eog_filter_h_freq"],
                               picks='eog', fir_design=p["fir_design"])
        if p.get("emg_filter_l_freq") is not None:
            self._filt_raw.filter(l_freq=p["emg_filter_l_freq"], h_freq=p["emg_filter_h_freq"],
                                   picks='emg', fir_design=p["fir_design"])
        self._filt_raw.notch_filter(p["notch_freqs"])
        self._filt_raw.info['bads'] = bad_channels
        self._filt_raw.set_eeg_reference(p["reference"], projection=True)
        self._filt_raw.apply_proj()
        print(f"    Preprocesado. Canales malos: {bad_channels or 'ninguno'}")

        # ── Re-ajuste ICA (determinístico por random_state fijo)
        s = self.params["ica_settings"]
        self._ica = ICA(
            n_components=s["n_components"],
            method=s["method"],
            max_iter=s["max_iter"],
            random_state=s["random_state"],
        )
        print(f"Ajustando ICA ({s['n_components']} componentes, método={s['method']})...")
        self._ica.fit(self._filt_raw, picks=s["fit_picks"])
        print(f"    Listo: {self._ica.n_components_} componentes ajustados")

    # ── Aplicación ────────────────────────────────────────────────────────────

    def apply(self, exclude_from: str = 'final') -> mne.io.RawArray:
        """
        Aplica ICA con los componentes del JSON y devuelve la señal limpia.

        exclude_from: clave de 'components_to_exclude' en el JSON.
                      'final'  — lista definitiva (por defecto)
                      'auto'   — solo componentes auto-detectados
                      'manual' — solo componentes marcados manualmente
        """
        if self._filt_raw is None:
            raise RuntimeError("Llamar a load_and_fit() antes de apply().")

        components = list(self.params["components_to_exclude"][exclude_from])
        self._ica.exclude = components
        print(f"Aplicando ICA — excluidos ({exclude_from}): {components}")

        self._raw_clean = self._filt_raw.copy()
        self._ica.apply(self._raw_clean)
        print("    Señal limpia lista.")
        return self._raw_clean

    # ── Visualización ─────────────────────────────────────────────────────────

    def plot_comparison(self, duration: int = 40) -> None:
        """
        Muestra tres gráficas de comparación antes/después:
          1. Overlay MNE canal por canal
          2. PSD media de canales EEG (1–45 Hz)
          3. Señal limpia scrollable
        """
        if self._raw_clean is None:
            raise RuntimeError("Llamar a apply() antes de plot_comparison().")

        m = self.params["metadata"]
        title_info = (f"Sub-{m['subject']} | Ses-{m['session']} | "
                      f"Run-{m['run']} | Tarea: {m['task']}")

        # 1. Overlay canal por canal
        self._ica.plot_overlay(
            self._filt_raw, exclude=self._ica.exclude,
            title=f"Overlay — antes vs. después de ICA\n{title_info}"
        )

        # 2. Comparación PSD
        psd_before = self._filt_raw.compute_psd(picks='eeg', fmin=1.0, fmax=45.0)
        psd_after  = self._raw_clean.compute_psd(picks='eeg', fmin=1.0, fmax=45.0)

        data_before, freqs = psd_before.get_data(return_freqs=True)
        data_after, _      = psd_after.get_data(return_freqs=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(freqs, data_before.mean(axis=0) * 1e12,
                    color='steelblue', alpha=0.85, label='Antes de ICA')
        ax.semilogy(freqs, data_after.mean(axis=0) * 1e12,
                    color='tomato', alpha=0.85, label='Después de ICA')
        ax.set_xlabel('Frecuencia (Hz)')
        ax.set_ylabel('PSD media (µV²/Hz)')
        ax.set_title(f'Comparación PSD — EEG\n{title_info}')
        ax.legend()
        fig.tight_layout()

        # 3. Señal limpia scrollable
        scalings = {'eeg': 30, 'emg': 150, 'eog': 150}
        color    = {'eeg': 'steelblue', 'emg': 'forestgreen', 'eog': 'darkorange'}
        self._raw_clean.plot(scalings=scalings, color=color,
                             title=f"Señal limpia post-ICA — {title_info}",
                             duration=duration)
        plt.show()

    # ── Aplicación directa sobre Raw existente (para scripts de análisis) ───────

    def apply_to_raw(self, raw: mne.io.RawArray,
                     exclude_from: str = 'final',
                     ica_fif_path: str = None) -> None:
        """
        Carga el objeto ICA desde el .fif y lo aplica in-place sobre un Raw
        ya cargado y croppeado. No re-fitea ICA: usa el modelo guardado.

        raw          : objeto Raw a limpiar (se modifica in-place)
        exclude_from : clave de 'components_to_exclude' en el JSON ('final' por defecto)
        ica_fif_path : ruta explícita al .fif; si None, se construye desde el JSON
        """
        if ica_fif_path is None:
            fname = self.params.get("ica_file")
            if not fname:
                raise ValueError(
                    "'ica_file' no encontrado en el JSON. "
                    "Re-correr ica_preprocessing.py para generar el .fif."
                )
            ica_fif_path = os.path.join(os.path.dirname(self._json_path), fname)

        ica = mne.preprocessing.read_ica(ica_fif_path)
        components = list(self.params["components_to_exclude"][exclude_from])
        ica.exclude = components
        ica.apply(raw)
        print(f"    ICA aplicado in-place — excluidos ({exclude_from}): {components}")

    # ── Propiedad de acceso ───────────────────────────────────────────────────

    @property
    def raw_clean(self) -> mne.io.RawArray:
        """Señal limpia resultante de apply(). None si apply() no fue llamado."""
        return self._raw_clean


# ── Ejemplo de uso ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sub, ses, task, run = "02", "02", "ejecutada", "06"
    base_path = f"D:\\dataset\\sub-{sub}\\ses-{ses}"

    json_path = os.path.join(base_path,
                             f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_ica.json")
    hdf5_path = os.path.join(base_path,
                             f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_eeg.hdf5")

    cleaner   = ICAApplicator(json_path)
    cleaner.load_and_fit(hdf5_path)
    raw_clean = cleaner.apply()           # usa 'final' por defecto
    cleaner.plot_comparison()

    # raw_clean está listo para continuar el análisis:
    # epochs = mne.Epochs(raw_clean, ...)
