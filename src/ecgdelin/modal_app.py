"""The Modal app: image, Volume and the functions every experiment step runs in.

All experiment steps run on Modal (CPU functions for loading, NeuroKit2 and scoring; T4 GPU functions
for TimesFM state extraction, training and inference). One Volume holds LUDB, the TimesFM weights,
stored states, checkpoints, predictions and result JSONs.

Scripts under scripts/ import ``app`` from here and add their own functions and local entrypoint
(``modal run scripts/smoke.py`` runs the CPU + T4 smoke test).
"""

import modal

from ecgdelin.paths import timesfm_src

VOLUME_NAME = "ecg-delineation"
VOL = "/vol"  # mount point inside containers
LUDB_IN_VOL = f"{VOL}/ludb"
HF_IN_VOL = f"{VOL}/hf"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Versions pinned to what the 2026-10-10 probe resolved locally (CUDA build on Linux).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.1",
        "numpy==2.5.3",
        "scipy==1.18.1",
        "pandas==2.3.3",
        "wfdb==4.3.1",
        "neurokit2==0.2.13",
        "ts2vg==1.2.4",  # FastNVG R-peaks (ecg_peaks method "emrich2023") for the Emrich 2024 check
        "huggingface_hub==2.2.0",
        "safetensors==0.8.0",
        "tqdm==4.70.1",
    )
    .env(
        {
            "HF_HOME": HF_IN_VOL,
            "LUDB_DIR": LUDB_IN_VOL,
            "TIMESFM_SRC": "/root",  # so timesfm_src() / "timesfm3" is the copied package in containers
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    .add_local_python_source("ecgdelin")
    .add_local_dir(str(timesfm_src() / "timesfm3"), remote_path="/root/timesfm3")
)

app = modal.App("ecg-delineation", image=image)


def _versions() -> dict[str, str]:
    import neurokit2
    import numpy
    import timesfm3
    import torch

    return {
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "neurokit2": neurokit2.__version__,
        "timesfm3": timesfm3.__file__,
    }


@app.function(volumes={VOL: volume}, timeout=600)
def smoke_cpu() -> dict[str, object]:
    import os

    return {**_versions(), "volume_top_level": sorted(os.listdir(VOL))}


@app.function(gpu="T4", volumes={VOL: volume}, timeout=600)
def smoke_gpu() -> dict[str, object]:
    import torch

    return {
        **_versions(),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
    }
