"""Compile preference feedback from SQLite into a HuggingFace Dataset.

Usage:
    python scripts/dataset/compile.py
"""

from pathlib import Path

import numpy as np
from datasets import Array3D, Dataset, Features, Value

from uedrlhf.config import FEEDBACK_TYPE, FEEDBACK_SOURCE, LEARNABILITY_MODE, DATA_DIR, IMAGES_DIR, DB_PATH
from uedrlhf.db import get_connection, resolve_table_name
from uedrlhf.utils import load_image_as_nparray


def main():
    table_name = resolve_table_name(FEEDBACK_SOURCE)

    print(f"Feedback type : {FEEDBACK_TYPE}")
    if FEEDBACK_TYPE == "learnability_feedback":
        print(f"Learnability  : {LEARNABILITY_MODE}")
    print(f"Source        : {FEEDBACK_SOURCE} → table '{table_name}'")
    print(f"Database      : {DB_PATH}")

    # Query completed rows
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(f'''
        SELECT env_1_id, env_2_id, preference, split, x_pct, y_pct
        FROM "{table_name}"
        WHERE status = 'completed'
    ''')
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print(f"No completed rows found in '{table_name}'. Nothing to compile.")
        return

    print(f"Found {len(rows)} completed rows\n")

    img_dir = IMAGES_DIR

    def as_uint8(img: np.ndarray) -> np.ndarray:
        
        return np.round(img * 255.0).astype(np.uint8)

    # Probe image shape from the first available pair to declare a fixed schema.
    img_shape = None
    for row in rows:
        p = img_dir / f"{row[0]}.png"
        if p.exists():
            img_shape = load_image_as_nparray(p).shape
            break
    if img_shape is None:
        print("No images found for any row. Nothing to compile.")
        return

    features = Features({
        "env_0_image": Array3D(shape=img_shape, dtype="uint8"),
        "env_1_image": Array3D(shape=img_shape, dtype="uint8"),
        "preference": Value("float64"),
        "split": Value("string"),
        "x_pct": Value("float64"),
        "y_pct": Value("float64"),
    })

    def ds_generator():
        skipped = 0
        for row in rows:
            env_0_id, env_1_id, preference, split, x_pct, y_pct = row

            img_0_path = img_dir / f"{env_0_id}.png"
            img_1_path = img_dir / f"{env_1_id}.png"

            if not all(p.exists() for p in [img_0_path, img_1_path]):
                skipped += 1
                continue

            img_0 = load_image_as_nparray(img_0_path)
            img_1 = load_image_as_nparray(img_1_path)
            yield {
                "env_0_image": as_uint8(img_0),
                "env_1_image": as_uint8(img_1),
                "preference": preference,
                "split": split,
                "x_pct": x_pct,
                "y_pct": y_pct,
            }

        if skipped:
            print(f"Skipped {skipped} rows (missing files)")

    ds = Dataset.from_generator(ds_generator, writer_batch_size=1000, features=features)

    output_dir = DATA_DIR / f"compiled_{table_name}_len_{len(rows)}"
    ds.save_to_disk(str(output_dir))
    print(f"\nDataset saved to {output_dir}  ({len(ds)} examples)")


if __name__ == "__main__":
    main()
