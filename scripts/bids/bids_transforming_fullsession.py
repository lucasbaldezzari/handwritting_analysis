"""
Flujo completo para convertir todos los runs de una sesión a BIDS.

Sesión registrada (sub-02, ses-01) — orden real de adquisición:
    run-01: basal
    run-02: emg
    run-03: eog
    run-04: entrenamiento
    run-05: ejecutada
    run-06: ejecutada
    run-07: ejecutada
    run-08: basal
    run-09: imaginada
    run-10: basal
    run-11: imaginada
    run-12: basal
    run-13: imaginada
"""

import os
from pathlib import Path

from neuroia_bids.adapters import GHiampAdapter, GenericXDFAdapter
from neuroia_bids.core.session import BIDSSession, BIDSSessionConfig
from neuroia_bids.io.bids_writer import BIDSWriter, BIDSWriterConfig
from neuroia_bids.builders.sidecar import SidecarConfig
from neuroia_bids import (
    DatasetConfig, DatasetWriter,
    ParticipantInfo, BIDSValidatorRunner,
)

# ── Configuración global ─────────────────────────────────────────────────────
sub = "02"
ses = "01"
subject_folder = "s2"
type_signal = "eeg"

root_path = Path(f"D:\\dataset\\{subject_folder}")
bids_root = Path("d:/testing/")
sfp_path = r"analysis\\ghiamp_montage.sfp"

# Marker names según tipo de tarea.
# basal/emg/eog: no hay trialTablet ni penDown en el HDF5.
# ejecutada/imaginada/entrenamiento: los cuatro marcadores están presentes.
MARKER_NAMES_SIMPLE = {
    1: "startRun",
    4: "trialLaptop",
}
MARKER_NAMES_FULL = {
    1: "startRun",
    2: "trialTablet",
    3: "penDown",
    4: "trialLaptop",
}

TASKS_SIMPLE  = {"basal", "emg", "eog"}
TASKS_WRITING = {"ejecutada", "imaginada", "entrenamiento"}

# ── Registros de la sesión: (task, run) en orden de adquisición ──────────────
recordings = [
    ("basal",          1),
    ("emg",            2),
    ("eog",            3),
    ("entrenamiento",  4),
    ("ejecutada",      5),
    ("ejecutada",      6),
    ("ejecutada",      7),
    ("basal",          8),
    ("imaginada",      9),
    ("basal",         10),
    ("imaginada",     11),
    ("basal",         12),
    ("imaginada",     13),
]

# ── 1. Inicializar dataset raíz (skip-if-exists en archivos y participante) ──
dataset_writer = DatasetWriter(
    root=bids_root,
    config=DatasetConfig(
        name="EEG Escritura NeuroIA",
        authors=["Lucas Baldezzari"],
        license="CC0",
        ethics_approvals=["IRB-UNGS-2024-001"],
    ),
)
dataset_writer.write(write_readme=True, write_changes=True)
result_p = dataset_writer.add_participant(
    ParticipantInfo(sub, age=20, sex="M", handedness="R")
)
if result_p.added:
    print(f"[dataset] Participante sub-{sub} agregado a participants.tsv")
else:
    print(f"[dataset] Participante sub-{sub} ya existe — omitido")

print()

# ── 2. Convertir cada run ────────────────────────────────────────────────────
for task, run in recordings:
    hdf5_name = f"sub-{sub}_ses-{ses}_task-{task}_run-{run:02d}_{type_signal}.hdf5"
    xdf_name  = f"sub-{sub}_ses-{ses}_task-{task}_run-{run:02d}_{type_signal}.xdf"

    marker_names = MARKER_NAMES_SIMPLE if task in TASKS_SIMPLE else MARKER_NAMES_FULL

    eeg_adapter = GHiampAdapter(
        filename=str(root_path / hdf5_name),
        marker_names=marker_names,
        channel_types={
            "EMG1": "emg",
            "EOG1": "eog",
            "EOG2": "eog",
        },
        eeg_channel_names=sfp_path,
    )

    xdf_path = root_path / xdf_name
    event_adapter = (
        GenericXDFAdapter(
            filename=str(xdf_path),
            trial_stream_name="Laptop_Markers",
            tablet_stream_name="Tablet_Markers",
        )
        if xdf_path.exists()
        else None
    )

    # Anchor de sincronización HDF5 <-> XDF.
    # basal/emg/eog → "trialLaptop" (único marcador de trial).
    # escritura/entrenamiento → "trialTablet" (trialLaptop y penDown heredan letra por proximidad).
    xdf_anchor = "trialLaptop" if task in TASKS_SIMPLE else "trialTablet"

    session_obj = BIDSSession(BIDSSessionConfig(
        subject=sub,
        session=ses,
        task=task,
        run=run,
        eeg_adapter=eeg_adapter,
        event_adapter=event_adapter,
        include_xdf_events=event_adapter is not None,
        xdf_anchor_hdf5_marker=xdf_anchor,
    ))

    write_beh = task in TASKS_WRITING

    result = BIDSWriter(
        root=bids_root,
        session=session_obj,
        config=BIDSWriterConfig(
            output_format="brainvision",
            overwrite=False,
            write_beh=write_beh,
            write_task_info=True,
            sidecar_config=SidecarConfig(
                task_name=task,
                eeg_reference="earlobes",
                power_line_frequency=50.0,
                manufacturer="g.tec medical engineering GmbH",
                model_name="g.HIAMP",
            ),
            montage_path=sfp_path,
        ),
    ).write()

    print(f"sub-{sub} ses-{ses} task-{task:15s} run-{run:02d}: {result.n_files} archivos ✓")

# ── 3. Validar ───────────────────────────────────────────────────────────────
print()
validation = BIDSValidatorRunner(bids_root).run()
print(validation.format_report())
assert validation.is_valid, "El dataset no es válido BIDS"
