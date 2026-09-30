"""Re-encode a compiled preference dataset with a fixed-shape Array3D(uint8) schema fast trainers need this.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from datasets import Array3D, Features, load_from_disk


def main():
    src = Path(sys.argv[1]).resolve()
    dst = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else src.with_name(src.name + "_arr")
    if dst.exists():
        sys.exit(f"Destination already exists, refusing to overwrite: {dst}")

    print(f"Source:      {src}")
    print(f"Destination: {dst}")
    ds = load_from_disk(str(src))
    print(f"Rows: {len(ds)}")

    probe = np.asarray(ds[0]["env_0_image"], dtype=np.float32)
    shape = probe.shape
    assert np.asarray(ds[0]["env_1_image"], dtype=np.float32).shape == shape
    print(f"Image shape: {shape}")

    new_features = Features(
        {
            **{k: v for k, v in ds.features.items() if k not in ("env_0_image", "env_1_image")},
            "env_0_image": Array3D(shape=shape, dtype="uint8"),
            "env_1_image": Array3D(shape=shape, dtype="uint8"),
        }
    )

    def to_uint8(batch):
        out = {}
        for col in ("env_0_image", "env_1_image"):
            arr = np.asarray(batch[col], dtype=np.float32)
            scaled = arr * 255.0
            rounded = np.round(scaled)
            if not np.allclose(scaled, rounded, atol=1e-3):
                raise ValueError(f"{col}: values are not exact uint8/255 — refusing lossy conversion")
            if arr.min() < 0.0 or arr.max() > 1.0:
                raise ValueError(f"{col}: values outside [0, 1]")
            out[col] = rounded.astype(np.uint8)
        return out

    nproc = int(os.environ.get("CONVERT_NPROC", "8"))
    converted = ds.map(
        to_uint8,
        batched=True,
        batch_size=256,
        num_proc=nproc,
        features=new_features,
        desc="re-encoding to Array3D(uint8)",
    )
    converted.save_to_disk(str(dst))
    print(f"Saved: {dst}")

    # ---- verification: exact content equality on a random sample of rows
    del converted
    new_ds = load_from_disk(str(dst))
    assert len(new_ds) == len(ds), "row count mismatch"
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), size=min(50, len(ds)), replace=False)
    other_cols = [c for c in ds.features if c not in ("env_0_image", "env_1_image")]
    for i in idxs:
        i = int(i)
        old_row, new_row = ds[i], new_ds[i]
        for col in ("env_0_image", "env_1_image"):
            expected = np.round(np.asarray(old_row[col], dtype=np.float32) * 255.0).astype(np.uint8)
            got = np.asarray(new_row[col])
            if not np.array_equal(expected, got):
                sys.exit(f"VERIFY FAIL: row {i} col {col} differs")
        for col in other_cols:
            if old_row[col] != new_row[col]:
                sys.exit(f"VERIFY FAIL: row {i} col {col}: {old_row[col]!r} != {new_row[col]!r}")
    print(f"VERIFY PASS: {len(idxs)} random rows identical (images via lossless uint8 round-trip)")

    def bench(d, label):
        d = d.with_format("numpy", columns=["env_0_image", "env_1_image"])
        t0 = time.perf_counter()
        for start in range(0, 10 * 256, 256):
            b = d[start : start + 256]
            _ = np.asarray(b["env_0_image"], dtype=np.float32)
            _ = np.asarray(b["env_1_image"], dtype=np.float32)
        dt = time.perf_counter() - t0
        print(f"decode benchmark [{label}]: 10x256 batches in {dt:.2f}s ({dt / 10 * 1000:.0f} ms/batch)")
        return dt

    t_old = bench(ds, "old nested-list")
    t_new = bench(new_ds, "new Array3D uint8")
    print(f"speedup: {t_old / t_new:.1f}x")
    print("CONVERSION_COMPLETE")


if __name__ == "__main__":
    main()
