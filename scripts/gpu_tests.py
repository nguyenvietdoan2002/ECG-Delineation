"""Run the tests marked `gpu` (and optionally `data`) inside a T4 container on Modal.

    modal run scripts/gpu_tests.py                 # -m gpu
    modal run scripts/gpu_tests.py --markers "gpu or data"
"""

from ecgdelin.modal_app import VOL, app, volume


@app.function(gpu="T4", volumes={VOL: volume}, timeout=1800, memory=16384)
def run_tests(markers: str) -> tuple[int, str]:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "/root/tests", "-m", markers, "-q", "-rs", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        cwd="/root",
    )
    volume.commit()  # keeps the downloaded TimesFM weights in /vol/hf
    return result.returncode, result.stdout + result.stderr


@app.local_entrypoint()
def main(markers: str = "gpu") -> None:
    code, output = run_tests.remote(markers)
    print(output)
    if code:
        raise SystemExit(code)
