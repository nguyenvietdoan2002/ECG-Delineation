"""Choose the PTB-XL ECGs for pretraining (decisions 73, 74 and 76) and write splits/ptbxl_pretrain_seed0.json.

    python scripts/ptbxl_pretrain_data.py select

Downloads PTB-XL's ptbxl_database.csv and scp_statements.csv and PTB-XL+'s feature tables from
PhysioNet's S3 mirror into work/ptbxl (skipped when present), fits the label check on folds 1 to 9
(fold 10 is never read), and draws 4,500 ECGs from folds 1 to 8 and 500 from fold 9 among those that
pass, stratified to each pool's mix before the check. Small tables only: the signals and ECGDeli's
fiducial files are fetched on Modal by scripts/ptbxl_fetch.py.
"""

import argparse
import datetime as dt
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd

from ecgdelin.ptbxl import (
    INTERVALS,
    NO_ECGDELI_FILES,
    SUPERCLASSES,
    af_or_flutter,
    failed_intervals,
    fit_check,
    interval_table,
    passes_check,
    strata,
    stratified_sample,
    superclasses,
)

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "work" / "ptbxl"
OUT = REPO / "splits" / "ptbxl_pretrain_seed0.json"
MIRROR = "https://physionet-open.s3.amazonaws.com"
FILES = {
    "ptbxl_database.csv": f"{MIRROR}/ptb-xl/1.0.3/ptbxl_database.csv",
    "scp_statements.csv": f"{MIRROR}/ptb-xl/1.0.3/scp_statements.csv",
    "12sl_features.csv": f"{MIRROR}/ptb-xl-plus/1.0.1/features/12sl_features.csv",
    "unig_features.csv": f"{MIRROR}/ptb-xl-plus/1.0.1/features/unig_features.csv",
    "ecgdeli_features.csv": f"{MIRROR}/ptb-xl-plus/1.0.1/features/ecgdeli_features.csv",
}
N_TRAIN, N_STOP, SEED = 4500, 500, 0
TRAIN_FOLDS, STOP_FOLD = tuple(range(1, 9)), 9


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def fetch_tables() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        path = WORK / name
        if not path.exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, path)


def _read(name: str, index: str) -> pd.DataFrame:
    return pd.read_csv(WORK / name, encoding="utf-8-sig", low_memory=False).set_index(index)


def _mix(ids: pd.Index, sup: pd.Series, af: pd.Series) -> dict[str, float]:
    """Share of ECGs with each superclass (they overlap) and with AF or flutter, in percent."""
    n = len(ids)
    shares = {s: round(100 * sum(s in sup[i] for i in ids) / n, 1) for s in SUPERCLASSES}
    shares["AF_or_flutter"] = round(100 * float(af[ids].mean()), 1)
    return shares


def select() -> None:
    fetch_tables()
    db = _read("ptbxl_database.csv", "ecg_id")
    db = db[db["strat_fold"] != 10]  # decision 73: fold 10 is never read
    scp = pd.read_csv(WORK / "scp_statements.csv", index_col=0)
    table = interval_table(db, _read("12sl_features.csv", "ecg_id"), _read("unig_features.csv", "ecg_id"), _read("ecgdeli_features.csv", "ecg_id"))
    params = fit_check(table, folds=range(1, 10))
    keep = passes_check(table, params)
    failed = failed_intervals(table, params)
    has_ecgdeli = table["de_QRS"].notna() & ~table.index.isin(NO_ECGDELI_FILES)
    eligible = keep & has_ecgdeli

    stratum = strata(db, scp)
    sup = superclasses(db, scp)
    af = af_or_flutter(db)
    folds = db["strat_fold"]
    in_train = folds.isin(TRAIN_FOLDS)
    in_stop = folds == STOP_FOLD
    train = stratified_sample(stratum[in_train & eligible], stratum[in_train], N_TRAIN, SEED)
    stop = stratified_sample(stratum[in_stop & eligible], stratum[in_stop], N_STOP, SEED)

    def summary(mask: pd.Series) -> dict[str, object]:
        ids = db.index[mask]
        return {
            "n": int(mask.sum()),
            "pass_check": int((keep & mask).sum()),
            "no_ecgdeli": int((~has_ecgdeli & mask).sum()),
            "eligible": int((eligible & mask).sum()),
            "failed_by_interval": {k: int((failed[k] & mask).sum()) for k in INTERVALS},
            "mix_before_check": _mix(ids, sup, af),
            "mix_eligible": _mix(db.index[mask & eligible], sup, af),
        }

    result = {
        "name": "ptbxl_pretrain_seed0",
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git": _git_head(),
        "sources": {"ptb-xl": "1.0.3", "ptb-xl-plus": "1.0.1", "mirror": MIRROR},
        "decisions": [73, 74, 76],
        "check": params.to_dict(),
        "folds_1_8": summary(in_train),
        "fold_9": summary(in_stop),
        "sample": {
            "seed": SEED,
            "train_mix": _mix(pd.Index(train), sup, af),
            "stop_mix": _mix(pd.Index(stop), sup, af),
            "train": train,
            "stop": stop,
        },
    }
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    print("check:", json.dumps(params.to_dict()))
    for name in ("folds_1_8", "fold_9"):
        s = result[name]
        print(f"{name}: {s['n']} ECGs, {s['pass_check']} pass, {s['no_ecgdeli']} without ECGDeli, {s['eligible']} eligible, failed {s['failed_by_interval']}")
        print(f"   mix before {s['mix_before_check']}  eligible {s['mix_eligible']}")
    print(f"sample: {len(train)} train, mix {result['sample']['train_mix']}; {len(stop)} stop, mix {result['sample']['stop_mix']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=["select"])
    args = parser.parse_args()
    {"select": select}[args.step]()


if __name__ == "__main__":
    sys.exit(main())
