import re
import sqlite3
import pickle
import numpy as np
from pathlib import Path
from datasets import Dataset, Features, Array3D, Value, Sequence
import yaml
import PIL.Image

RESIZE = (64,64)

# -------------------------------- helpers
def sanitize_table_name(model_name: str) -> str:
    """Convert a model name like 'gemini-2.5-flash' into a valid SQLite table name."""
    name = model_name.lower()
    name = re.sub(r"[^a-z0-9]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name


def resolve_table_name(feedback_source: str) -> str:
    """Map the YAML feedback_source value to the actual SQLite table name.
    
    'human'            → 'human_preference_pairs'
    'gemini-2.5-flash' → 'gemini_2_5_flash'
    """
    if feedback_source == "human":
        return "human_preference_pairs"
    return sanitize_table_name(feedback_source)


def load_array(file_path: Path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def state_to_dict(state_obj) -> dict:
    """Converts the custom State PyTree into a plain dictionary of NumPy arrays."""
    return {
        "pos": np.array(state_obj.pos),
        "theta": np.array(state_obj.theta),
        "vel": np.array(state_obj.vel),
        "done": np.array(state_obj.done),
        "term": np.array(state_obj.term),
        "goal_reached": np.array(state_obj.goal_reached),
        "move_term": np.array(state_obj.move_term),
        "step": np.array(state_obj.step),
        "ep_done": np.array(state_obj.ep_done),
        "goal": np.array(state_obj.goal),
        "map_data": np.array(state_obj.map_data),
        "rew_lambda": np.array(state_obj.rew_lambda),
    }


def load_image_as_nparray(file_path: Path) -> np.ndarray:
    img = PIL.Image.open(file_path).resize(RESIZE).convert("RGB")
    return np.array(img).astype(np.float32)

# -------------------------------- helpers TODO move the helpers to some other util file



def main():
    BASE_DIR = Path(__file__).resolve().parent
    config_path = BASE_DIR / "dataset_compilation.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)


    FEEDBACK_TYPE      = config["feedback_type"]
    LEARNABILITY_MODE  = config.get("learnability_mode", "dijkstra")  # default to dijkstra
    FEEDBACK_SOURCE = config["feedback_source"]

    SUBFOLDER_MAP = {
        "difficulty_feedback": "difficulty",
        "learnability_feedback": "learnability",
    }

    VALID_LEARNABILITY_MODES = {"dijkstra", "param"}

    if FEEDBACK_TYPE not in SUBFOLDER_MAP:
        raise ValueError(
            f"Unknown feedback_type '{FEEDBACK_TYPE}' in config.yaml. "
            f"Must be one of: {list(SUBFOLDER_MAP.keys())}"
        )

    # then validate learnability_mode
    if FEEDBACK_TYPE == "learnability_feedback":
        if LEARNABILITY_MODE not in VALID_LEARNABILITY_MODES:
            raise ValueError(
                f"Unknown learnability_mode '{LEARNABILITY_MODE}' in config.yaml. "
                f"Must be one of: {VALID_LEARNABILITY_MODES}"
            )
        print(f"Learnability mode: {LEARNABILITY_MODE}")

    subfolder = SUBFOLDER_MAP[FEEDBACK_TYPE]


    data_dir = BASE_DIR / "data" / subfolder
    img_dir = data_dir / "images"
    arr_dir = data_dir / "arrays"
    db_path = data_dir / "feedback.db"

    # resolve table name based on llm 

    if FEEDBACK_TYPE == "learnability_feedback":
        table_name =  resolve_table_name(FEEDBACK_SOURCE) + f"_{LEARNABILITY_MODE}"  
    else:
        table_name = resolve_table_name(FEEDBACK_SOURCE)

                                        
    print(f"Feedback type : {FEEDBACK_TYPE}")
    print(f"Source        : {FEEDBACK_SOURCE} → table '{table_name}'")
    print(f"Database      : {db_path}")

    # query completed rows from table
    conn = sqlite3.connect(db_path)
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

    
    def ds_generator():
        skipped = 0
        for row in rows:
            env_0_id, env_1_id, preference, split, x_pct, y_pct = row

            #arr_0_path = arr_dir / f"{env_0_id}.pkl"
            #arr_1_path = arr_dir / f"{env_1_id}.pkl"
            img_0_path = img_dir / f"{env_0_id}.png"
            img_1_path = img_dir / f"{env_1_id}.png"

            if not all(p.exists() for p in [img_0_path, img_1_path]):
                skipped += 1
                continue

            #arr_0 = load_array(arr_0_path)
            #arr_1 = load_array(arr_1_path)
            img_0 = load_image_as_nparray(img_0_path)
            img_1 = load_image_as_nparray(img_1_path)

            yield {
                "env_0_image": img_0,
                "env_1_image": img_1,
                "preference": preference,
                "split": split,
                "x_pct": x_pct,
                "y_pct": y_pct,
            }

        if skipped:
            print(f"Skipped {skipped} rows (missing files)")

    ds = Dataset.from_generator(ds_generator, writer_batch_size=1000)
    #ds = ds.with_format("jax")

    output_dir = data_dir / f"compiled_{table_name}_len_{len(rows)}"
    ds.save_to_disk(str(output_dir))
    print(f"\nDataset saved to {output_dir}  ({len(ds)} examples)")


if __name__ == "__main__":
    main()