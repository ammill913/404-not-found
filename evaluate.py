#!/usr/bin/env python3
"""Basic scorer for the hackathon levels. Needs only numpy + pandas.

    python evaluate.py --level1 level1.csv --level2 level2.csv --level3 level3.csv --out results/

Give any subset of the levels. File formats are described in README.md.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GENS = ["musicgen", "musicldm", "audioldm2", "stable_audio_open", "mustango"]
SIX = ["real"] + GENS
SEVEN = SIX + ["suno"]


# ---------------------------------------------------------------- metrics
def confusion(y, p, classes):
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), int)
    for a, b in zip(y, p):
        cm[idx[a], idx[b]] += 1
    return cm


def closed_set(d, col, classes):
    cm = confusion(d.source, d[col], classes)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return {"n_clips": len(d), "accuracy": float(tp.sum() / cm.sum()), "macro_f1": float(f1.mean()),
            "confusion": {"classes": classes, "rows_true_cols_pred": cm.tolist()}}


def real_vs_fake(d, col):
    fake = (d.source != "real").to_numpy()
    pred_fake = (d[col] != "real").to_numpy()
    return float((pred_fake[fake].mean() + (~pred_fake[~fake]).mean()) / 2)


def open_set(d, col, known, unknown_groups, labels):
    p = d[col]
    out = {"n_clips": len(d),
           "detected_as_unknown": {g: float((p[d.source == g] == "unknown").mean()) for g in unknown_groups},
           "false_alarm_on_known": float((p[d.source.isin(known)] == "unknown").mean())}
    cols = labels + ["unknown"]
    out["assignment"] = {s: {c: float((p[d.source == s] == c).mean()) for c in cols}
                         for s in SEVEN if (d.source == s).any()}
    return out


# ---------------------------------------------------------------- input
def read(path, rows, meta, cols, allowed):
    df = pd.read_csv(path)
    for c in ["id"] + cols:
        if c not in df:
            sys.exit(f"{path}: column '{c}' is missing")
    if df.id.duplicated().any():
        sys.exit(f"{path}: duplicated ids")
    if (~df.id.isin(meta.id)).any():
        sys.exit(f"{path}: ids that are not in metadata.csv, e.g. {df.id[~df.id.isin(meta.id)].iloc[0]}")
    missing = sorted(set(rows.id) - set(df.id))
    if missing:
        sys.exit(f"{path}: {len(missing)} of the {len(rows)} required clips are missing, e.g. {missing[0]}")
    d = rows.merge(df[["id"] + cols], on="id")
    for c in cols:
        bad = ~d[c].isin(allowed[c])
        if bad.any():
            sys.exit(f"{path}: column '{c}' has labels outside {sorted(allowed[c])}, e.g. {d[c][bad].iloc[0]!r}")
    return d


def fmt_confusion(e):
    cl, cm = e["confusion"]["classes"], e["confusion"]["rows_true_cols_pred"]
    L = ["  confusion matrix (rows = true, columns = predicted)", "  " + " " * 18 + "".join(f"{c[:9]:>10}" for c in cl)]
    return L + ["  " + f"{c:<18}" + "".join(f"{v:>10d}" for v in row) for c, row in zip(cl, cm)]


def fmt_open(e):
    L = [f"  false alarm on known clips (predicted 'unknown'): {e['false_alarm_on_known']:.3f}"]
    L += [f"  unknown group '{g}' detected as unknown: {v:.3f}" for g, v in e["detected_as_unknown"].items()]
    cols = list(next(iter(e["assignment"].values())))
    L += ["", "  where each source goes (share of its clips)", "  " + " " * 18 + "".join(f"{c[:9]:>10}" for c in cols)]
    return L + ["  " + f"{s:<18}" + "".join(f"{row[c]:>10.2f}" for c in cols) for s, row in e["assignment"].items()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level1")
    ap.add_argument("--level2")
    ap.add_argument("--level3")
    ap.add_argument("--meta", default=str(Path(__file__).resolve().parent / "metadata.csv"))
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if not (a.level1 or a.level2 or a.level3):
        ap.error("give at least one of --level1 --level2 --level3")

    meta = pd.read_csv(a.meta)
    meta = meta[["id", "source", "split"]]
    res, L = {}, []

    if a.level1:
        rows = meta[(meta.split == "test") & meta.source.isin(GENS)]
        d = read(a.level1, rows, meta, ["pred"], {"pred": set(GENS)})
        r = res["level1"] = closed_set(d, "pred", GENS)
        L += ["LEVEL 1 - which generator? (test split)", f"  {r['n_clips']} clips",
              f"  accuracy  {r['accuracy']:.3f}", f"  macro_f1  {r['macro_f1']:.3f}", ""] + fmt_confusion(r) + [""]

    if a.level2:
        rows = meta[(meta.split == "test") & meta.source.isin(SIX)]
        d = read(a.level2, rows, meta, ["pred"], {"pred": set(SIX)})
        r = res["level2"] = closed_set(d, "pred", SIX)
        r["real_vs_fake_balanced_accuracy"] = real_vs_fake(d, "pred")
        L += ["LEVEL 2 - which source, and real vs generated? (test split)", f"  {r['n_clips']} clips",
              f"  accuracy  {r['accuracy']:.3f}", f"  macro_f1  {r['macro_f1']:.3f}",
              f"  real_vs_fake_balanced_accuracy  {r['real_vs_fake_balanced_accuracy']:.3f}", ""] + fmt_confusion(r) + [""]

    if a.level3:
        rows = meta[meta.split == "holdout"]
        have = [c for c in ("pred_gen5", "pred_gen6") if c in pd.read_csv(a.level3, nrows=0).columns]
        if not have:
            sys.exit(f"{a.level3}: needs a 'pred_gen5' and/or 'pred_gen6' column")
        d = read(a.level3, rows, meta, have, {"pred_gen5": set(GENS) | {"unknown"},
                                              "pred_gen6": set(SIX) | {"unknown"}})
        res["level3"] = {}
        if "pred_gen5" in have:
            r = res["level3"]["3A_gen5"] = open_set(d, "pred_gen5", GENS, ["real", "suno"], GENS)
            L += ["LEVEL 3A - gen5 model on the holdout (real and Suno are unknown)", f"  {r['n_clips']} clips"] \
                 + fmt_open(r) + [""]
        if "pred_gen6" in have:
            r = res["level3"]["3B_gen6"] = open_set(d, "pred_gen6", SIX, ["suno"], SIX)
            L += ["LEVEL 3B - gen6 model on the holdout (Suno is unknown)", f"  {r['n_clips']} clips"] \
                 + fmt_open(r) + [""]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores.json").write_text(json.dumps(res, indent=2))
    (out / "report.txt").write_text("\n".join(L))
    print("\n".join(L))
    print(f"written: {out / 'scores.json'}, {out / 'report.txt'}")


if __name__ == "__main__":
    main()
