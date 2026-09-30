import jax
import jax.numpy as jnp
from datasets import load_from_disk
from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav

dataset_path = "data/difficulty/compiled_gemini_2_5_flash_clean"
print(f"Loading dataset from {dataset_path}")

dataset = load_from_disk(dataset_path)
print("Dataset loaded successfully!")
print("Columns:", dataset.column_names)
print("Number of examples:", len(dataset))

# Let's check the first example
if len(dataset) > 0:
    print("First example env_1 state shape:", len(dataset[0]["env_1_state"]))
