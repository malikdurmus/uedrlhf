import jax.numpy as jnp
from datasets import load_from_disk
import numpy as np
from PIL import Image

dataset_path = "data/difficulty/compiled_gemini_2_5_flash_clean"
dataset = load_from_disk(dataset_path)

img_0 = np.array(dataset[0]["env_0_image"])
img_1 = np.array(dataset[10]["env_0_image"])

print("Image shape:", img_0.shape)
print("Image dtype:", img_0.dtype)

# Save the image to inspect it
Image.fromarray((img_0 * 1).astype(np.uint8)).save("tests/img_0.png")
Image.fromarray((img_1 * 1).astype(np.uint8)).save("tests/img_1.png")
