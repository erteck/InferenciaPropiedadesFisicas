"""Etapa 6: particiones fijas y empaquetado en Zarr listo para entrenar.

Entrada  Datos/raw/manifest.csv, Datos/interim/{normalized,masks,structure_maps}
Salida   Datos/splits/{train,val,test}.csv
         Datos/processed/{train,val,test}.zarr

Cada store Zarr contiene ``image`` (N,H,W), ``mask`` (N,H,W), ``ids`` (N),
un grupo ``labels`` con una columna por etiqueta y un grupo ``structure``
con un arreglo por mapa. Las particiones se hacen sobre las imágenes
disponibles; la etiqueta de estratificación se ignora si no está presente.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import zarr

from .common import INTERIM, PROCESSED, RAW, SPLITS, ensure_dir, load_params, write_json

LABEL_COLUMNS = ["LogMassTaylor", "LogMassNSA", "g_i", "T_11"]


def available_ids() -> list[str]:
    return sorted(f.stem for f in (INTERIM / "normalized").glob("*.npy"))


def assign_splits(ids: list[str], fractions: dict[str, float], seed: int, strata: pd.Series | None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(fractions)
    cum = np.cumsum([fractions[n] for n in names])
    assert abs(cum[-1] - 1.0) < 1e-6, "las fracciones deben sumar 1"
    df = pd.DataFrame({"id": ids})
    df["stratum"] = strata.reindex(ids).fillna("na").values if strata is not None else "na"
    parts = []
    for _, group in df.groupby("stratum", sort=False):
        idx = rng.permutation(len(group))
        bins = np.searchsorted(cum, (idx + 0.5) / len(group))
        parts.append(group.assign(split=[names[b] for b in bins]))
    out = pd.concat(parts).sort_values("id").reset_index(drop=True)
    if len(out) < len(names):
        out["split"] = names[0]
    return out


def pack(split_name: str, ids: list[str], manifest: pd.DataFrame, maps: list[str]) -> None:
    store = zarr.open_group(str(PROCESSED / f"{split_name}.zarr"), mode="w")
    if not ids:
        store.attrs["n"] = 0
        return
    images = np.stack([np.load(INTERIM / "normalized" / f"{i}.npy") for i in ids])
    masks = np.stack([np.load(INTERIM / "masks" / f"{i}.npy") for i in ids])
    chunks = (1,) + images.shape[1:]
    store.create_dataset("image", data=images, chunks=chunks, dtype="float32")
    store.create_dataset("mask", data=masks, chunks=chunks, dtype="uint8")
    store.create_dataset("ids", data=np.array(ids, dtype=str))
    lab = manifest.set_index("MANGAID").reindex(ids)
    labels = store.create_group("labels")
    for col in LABEL_COLUMNS:
        if col in lab:
            labels.create_dataset(col, data=lab[col].to_numpy(dtype="float32", na_value=np.nan))
    structure = store.create_group("structure")
    for name in maps:
        arr = np.stack([np.load(INTERIM / "structure_maps" / f"{i}.npz")[name] for i in ids])
        structure.create_dataset(name, data=arr, chunks=chunks, dtype="float32")
    store.attrs["n"] = len(ids)


def main() -> None:
    p = load_params("split")
    maps = load_params("structure")["maps"]
    ensure_dir(SPLITS)
    ensure_dir(PROCESSED)
    manifest = pd.read_csv(RAW / "manifest.csv")
    ids = available_ids()
    strata = manifest.set_index("MANGAID")[p["stratify_by"]] if p.get("stratify_by") in manifest else None
    assignment = assign_splits(ids, p["fractions"], int(p["seed"]), strata)
    counts = {}
    for name in p["fractions"]:
        members = assignment.loc[assignment["split"] == name, "id"].tolist()
        pd.DataFrame({"id": members}).to_csv(SPLITS / f"{name}.csv", index=False)
        pack(name, members, manifest, maps)
        counts[name] = len(members)
    write_json(SPLITS / "split_summary.json", {"params": p, "counts": counts})
    print(f"[split] {counts}")


if __name__ == "__main__":
    main()
