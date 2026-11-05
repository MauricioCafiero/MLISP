"""Run scripts/train.py on a Modal GPU (L4, falling back to A10G).

The image reproduces the README install with CUDA JAX (same jax-0.4.30-era pins), with
third_party/ installed editable so chemutils still finds upstream/descriptor.csv.
scripts/, upstream/ and the data dir are mounted at run time; runs/<name>/ is written to the
Modal volume "mlisp-runs" (fetch with `modal volume get mlisp-runs <name> runs/`).

    modal run --detach scripts/modal_train.py --which vacuum
    modal run --detach scripts/modal_train.py --which solvent
    ... --which delta | embed      (step 1 (a) / (b); embed needs rotx7_vac on the volume)
"""
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
R = "/root/mlisp"
DATA = "data/rotx7"
SOLVENTS = "hexane=25 chloroform=11 acetone=7 dmf=53 water=1 dmso=104"
RUNS = {  # name -> train.py args
    "rotx7_vac": f"--model vacuum --sets gas --epochs 50",
    "rotx7_solv": f"--model solvent --sets {SOLVENTS} --epochs 20",
    # step 1 (a): solvent network from scratch on E_ALPB - E_gas; BE adds rotx7_vac
    "rotx7_delta": f"--model solvent --sets {SOLVENTS} --delta-from gas --init scratch --epochs 40 --lr 1e-3",
    # step 1 (b): rotx7_vac weights in the solvent network, only the solvent embedding trained
    "rotx7_embed": f"--model solvent --sets {SOLVENTS} --init-vacuum runs/rotx7_vac "
                   f"--train-only non_linear_solvent_embedding --epochs 20",
}
WHICH = {"vacuum": "rotx7_vac", "solvent": "rotx7_solv", "delta": "rotx7_delta", "embed": "rotx7_embed"}

# the image is built from the local side only; inside the container ROOT is "/" and unused
lock = [l for l in (ROOT / "requirements.lock.txt").read_text().splitlines()  # dataclasses: py3.6 backport
        if l.strip() and not l.startswith(("#", "jax==", "jaxlib==", "dataclasses=="))] if modal.is_local() else []
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(*lock, "jax[cuda12]==0.4.30", "jaxlib==0.4.30")
    .add_local_dir(ROOT / "third_party", f"{R}/third_party", copy=True)
    .run_commands(f"pip install --no-deps -e {R}/third_party/chemtrain -e {R}/third_party/chemutils")
    .env({"JAX_PLATFORMS": "cuda", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(ROOT / "scripts", f"{R}/scripts")
    .add_local_dir(ROOT / "upstream", f"{R}/upstream")
    .add_local_dir(ROOT / DATA, f"{R}/{DATA}")
)
app = modal.App("mlisp-train", image=image)
runs = modal.Volume.from_name("mlisp-runs", create_if_missing=True)


@app.function(gpu=["L4", "A10G"], timeout=6 * 3600, volumes={f"{R}/runs": runs})
def train(name: str, args: str):
    subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
    cmd = f"python {R}/scripts/train.py {args} --data-dir {R}/{DATA} --name {name}"
    print(cmd, flush=True)
    log = open(f"{R}/runs/{name}.out", "w")
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        if "Warning" not in line:
            print(line, end="", flush=True)
        log.write(line)
    log.close()
    runs.commit()
    assert proc.wait() == 0, f"train.py exited {proc.returncode}"


@app.local_entrypoint()
def main(which: str):
    name = WHICH[which]
    train.remote(name, RUNS[name])
