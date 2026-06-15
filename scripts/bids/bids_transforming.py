import os
from neuroia_bids.adapters import GHiampAdapter, GenericXDFAdapter
from neuroia_bids.core.session import BIDSSession, BIDSSessionConfig
from neuroia_bids.io.bids_writer import BIDSWriter, BIDSWriterConfig
from neuroia_bids.builders.sidecar import SidecarConfig
from neuroia_bids import DatasetConfig, DatasetWriter, ParticipantInfo
from datetime import datetime, timezone
import pandas as pd


sub = "01"
ses = "01"
taskname = "basal"
run = "01"
subject_folder = "s1"
type_signal = "eeg"
root_path = f"D:\\dataset\\{subject_folder}"
bids_root = "d:/testing/"
sfp_path = r"analysis\\ghiamp_montage.sfp"

ghiamp_file = f"sub-{sub}_ses-{ses}_task-{taskname}_run-{run}_{type_signal}.hdf5"
lsl_file = f"sub-{sub}_ses-{ses}_task-{taskname}_run-{run}_{type_signal}.xdf"

# TypeIDs raw del HDF5 (neuroia-bids los lee directamente, sin la normalización de pyhwr).
# pyhwr.GHiampDataManager restaría el mínimo y sumaría 1, exponiendo 1–4 en lugar de 8–11.
if taskname in ["basal", "emg", "eog"]:
    marker_names = {
        1: "startRun",
        4: "trialLaptop",
    }
else:
    marker_names = {
        1: "startRun",
        2: "trialTablet",
        3: "penDown",
        4: "trialLaptop",
    }

# ── Dataset raíz (skip-if-exists en todos los archivos y participantes) ──────
dataset_writer = DatasetWriter(
    root=bids_root,
    config=DatasetConfig(name="EEG Escritura NeuroIA"),
)
dataset_writer.write(write_readme=True, write_changes=True)
result_participant = dataset_writer.add_participant(
    ParticipantInfo(sub, age=20, sex="M", handedness="R")
)
if result_participant.added:
    print(f"[dataset] Participante sub-{sub} agregado a participants.tsv")
else:
    print(f"[dataset] Participante sub-{sub} ya existe — omitido")

# ─────────────────────────────────────────────────────────────────────────────

adapter = GHiampAdapter(
    filename=os.path.join(root_path, ghiamp_file),
    marker_names=marker_names,
    channel_types={
        "EMG1": "emg",
        "EOG1": "eog",
        "EOG2": "eog",
    },
    eeg_channel_names=sfp_path,
)

# Tareas que usan XDF para enriquecer events.tsv con letras y metadatos de trial.
# Para basal/emg/eog no hay trials en el XDF ni se necesita sincronización.
# Laptop_Markers siempre esta presente en el XDF de cualquier tarea.
# tablet_stream_name se pasa siempre; si no existe (basal/emg/eog),
# _merge_tablet_data lo ignora sin error.
xdf_adapter = GenericXDFAdapter(
    filename=os.path.join(root_path, lsl_file),
    trial_stream_name="Laptop_Markers",
    tablet_stream_name="Tablet_Markers",
)

# Ver que trials trae el XDF
for trial in xdf_adapter.trials[:6]:
    print(f"Trial {trial.trial_id}: letra='{trial.letter}', "
          f"pendowns={len(trial.pen_down_ms)}")

# Acceso a datos HDF5
print(adapter.n_channels)
print(adapter.duration_seconds)
print(adapter.markers[:10])
df_markers = adapter.markers_dataframe()
df_channels = adapter.channels_dataframe()

# Anchor de sincronizacion HDF5<->XDF:
#   basal/emg/eog       -> "trialLaptop" (unico marcador de trial en el HDF5)
#   ejecutada/imaginada -> "trialTablet" (step 1: letras en trialTablet;
#                          step 2: trialLaptop y penDown heredan por proximidad)
xdf_anchor = (
    "trialLaptop"
    if taskname in ["basal", "emg", "eog"]
    else "trialTablet"
)

session = BIDSSession(BIDSSessionConfig(
    subject=sub,
    session=ses,
    task=taskname,
    run=int(run),
    eeg_adapter=adapter,
    event_adapter=xdf_adapter,
    include_xdf_events=True,
    xdf_anchor_hdf5_marker=xdf_anchor,
))

# Marcadores HDF5 + columna 'letter' del XDF
df_enriquecido = session.enriched_markers_dataframe()
print(df_enriquecido[:50])

# Metadatos de trials (tiempos XDF convertidos al eje HDF5)
df_trials = session.xdf_trial_metadata(time_reference="session_start")
print(df_trials)
# Columns: trial_id, run_id, letter, trial_start_s, trial_cue_s, ...

# Coordenadas de la tablet (vacio para basal/emg/eog)
df_coords = session.coordinates_dataframe()
print(df_coords)
# Columns: trial_id, letter, x, y, t_rel_s

# Configurar metadatos del sidecar eeg.json
sidecar = SidecarConfig(
    task_name=taskname,                            # campo TaskName (REQUIRED)
    eeg_reference="earlobes",                 # campo EEGReference (REQUIRED)
    power_line_frequency=50.0,                    # Uruguay usa 50 Hz
    manufacturer="g.tec medical engineering GmbH",
    model_name="g.HIAMP",
)

write_beh = taskname in ["ejecutada", "imaginada", "entrenamiento"]
writer = BIDSWriter(
    root=bids_root,
    session=session,
    config=BIDSWriterConfig(
        output_format="brainvision",
        overwrite=False,
        write_beh=write_beh,
        write_task_info=True,
        sidecar_config=sidecar,
        montage_path=sfp_path,
    ),
)

result = writer.write()

print(f"Archivos creados: {result.n_files}")
for f in result.created_files:
    print(f"  {f}")