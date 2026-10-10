"""Smoke test of the Modal image and Volume on a CPU container and a T4 container.

    modal run scripts/smoke.py
"""

from ecgdelin.modal_app import app, smoke_cpu, smoke_gpu


@app.local_entrypoint()
def main() -> None:
    print("cpu:", smoke_cpu.remote())
    print("gpu:", smoke_gpu.remote())
