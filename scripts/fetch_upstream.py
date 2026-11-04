"""Download the published ConSolv files into upstream/ (gitignored).

ConSolv (https://github.com/tummfm/ConSolv, Zhang & Zavadlav, arXiv:2606.24983)
ships no licence for its top-level files, so this repo does not redistribute
them; they are fetched from a pinned commit instead.

    python scripts/fetch_upstream.py
"""
import urllib.request
from pathlib import Path

COMMIT = "06801dd5f53182ed86c31eee1fa03cc011c05b5a"
BASE = f"https://raw.githubusercontent.com/tummfm/ConSolv/{COMMIT}"
FILES = {
    "examples/SFE/output/solvent/config.toml": "solvent/config.toml",
    "examples/SFE/output/solvent/final_params.pkl": "solvent/final_params.pkl",
    "examples/SFE/output/solvent/full_init_params.pkl": "solvent/full_init_params.pkl",
    "examples/SFE/output/vacuum/config.toml": "vacuum/config.toml",
    "examples/SFE/output/vacuum/best_params_1.pkl": "vacuum/best_params_1.pkl",
    "dataset/descriptor.csv": "descriptor.csv",
}

out = Path(__file__).resolve().parents[1] / "upstream"
for src, dst in FILES.items():
    path = out / dst
    if path.exists():
        print(f"have {dst}")
        continue
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"{BASE}/{src}", path)
    print(f"got  {dst} ({path.stat().st_size} B)")
