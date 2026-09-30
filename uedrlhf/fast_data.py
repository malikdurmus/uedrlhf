"""Shared helpers for the faster scripts/train/*_fast.py.

Data lives on the device 
"""

import os

import numpy as np
import jax
import jax.numpy as jnp


def flag_from(env_name: str, cfg: dict, key: str, default: bool = False) -> bool:
    """Boolean setting with env-var override, same parse as train_pixels.py."""
    return str(os.environ.get(env_name, cfg.get(key, default))).lower() in ("1", "true", "yes")


def print_run_banner(**items):
    """Print the run configuration up front, before anything else happens."""
    width = max(len(k) for k in items)
    print("=" * 60)
    print("RUN CONFIG")
    print("-" * 60)
    for k, v in items.items():
        print(f"  {k:<{width}} : {v}")
    print("=" * 60, flush=True)


def three_way_split(ds, seed):
    
    first = ds.train_test_split(test_size=0.1, seed=seed)
    second = first["train"].train_test_split(test_size=1.0 / 9.0, seed=seed)
    return second["train"], second["test"], first["test"]


def _legacy_invert(a):
   
    return np.where(a == 0, 0, 256 - a.astype(np.int64)).astype(np.uint8)


def to_device_uint8(split, chunk_rows=2048):
   
    split = split.with_format("numpy", columns=["env_0_image", "env_1_image", "preference"])
    n = len(split)
    i0_parts, i1_parts, pref_parts = [], [], []
    for start in range(0, n, chunk_rows):
        batch = split[start : start + chunk_rows]
        i0 = np.asarray(batch["env_0_image"])
        i1 = np.asarray(batch["env_1_image"])
        prefs = np.asarray(batch["preference"], dtype=np.float32)
       
        if np.issubdtype(i0.dtype, np.floating):
            assert i0.max() <= 1.0 and i1.max() <= 1.0 and i0.min() >= 0.0,                 f"float images outside [0,1]: {i0.min()}..{i0.max()}"
            i0 = np.round(i0.astype(np.float32) * 255.0).astype(np.uint8)
            i1 = np.round(i1.astype(np.float32) * 255.0).astype(np.uint8)
        elif i0.dtype != np.uint8:
            assert i0.min() >= 0 and i0.max() <= 255 and i1.min() >= 0 and i1.max() <= 255,                 f"integer images outside [0,255]: {i0.min()}..{i0.max()}"
            i0 = i0.astype(np.uint8)
            i1 = i1.astype(np.uint8)
        if flag_from("LEGACY_INVERT_INPUTS", {}, "legacy_invert_inputs"):
           
            i0 = _legacy_invert(i0)
            i1 = _legacy_invert(i1)
        i0_parts.append(jnp.asarray(i0))
        i1_parts.append(jnp.asarray(i1))
        pref_parts.append(jnp.asarray(prefs))
    return (jnp.concatenate(i0_parts), jnp.concatenate(i1_parts),
            jnp.concatenate(pref_parts))


def batches(i0, i1, prefs, order, batch_size):
    n = order.shape[0]
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield i0[idx], i1[idx], prefs[idx]


def to_float(img_u8):
    return img_u8.astype(jnp.float32) / 255.0


def _symmetry_single(k, img):
    rot = jax.lax.switch(
        k % 4,
        [
            lambda x: x,
            lambda x: jnp.rot90(x, 1),
            lambda x: jnp.rot90(x, 2),
            lambda x: jnp.rot90(x, 3),
        ],
        img,
    )
    return jax.lax.cond(k >= 4, lambda x: x[:, ::-1], lambda x: x, rot)


def random_symmetry(key, imgs):
    """Random dihedral transform per image. Square images only; labels unchanged."""
    ks = jax.random.randint(key, (imgs.shape[0],), 0, 8)
    return jax.vmap(_symmetry_single)(ks, imgs)
