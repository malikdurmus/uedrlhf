import jax
import numpy as np
from PIL import Image
from uedrlhf.config import ENV_CONFIG, ENVIRONMENT, RULESET
from uedrlhf.environments import get_env_backend

def test_resize():
    print(f"Testing Environment: {ENVIRONMENT}")
    print(f"Testing Ruleset: {RULESET}")
    
    backend = get_env_backend(ENVIRONMENT, ENV_CONFIG)
    backend.create_env()
    
    rng = jax.random.PRNGKey(42)
    
    print("Generating 1 environment...")
    states = backend.generate_states(rng, 1)
    
    print("Rendering...")
    images = backend.render_batch(states)
    
    original_img = Image.fromarray(images[0])
    
    # Save original
    orig_path = "resize_original_416x416.png"
    original_img.save(orig_path)
    print(f"Saved original {original_img.size} to {orig_path}")
    
    # Resize 64x64
    img_64 = original_img.resize((64, 64), Image.Resampling.LANCZOS)
    path_64 = "resize_64x64.png"
    img_64.save(path_64)
    print(f"Saved 64x64 to {path_64}")
    
    # Resize 120x120
    img_120 = original_img.resize((120, 120), Image.Resampling.LANCZOS)
    path_120 = "resize_120x120.png"
    img_120.save(path_120)
    print(f"Saved 120x120 to {path_120}")

    # Resize 200x200
    img_200 = original_img.resize((200, 200), Image.Resampling.LANCZOS)
    path_200 = "resize_200x200.png"
    img_200.save(path_200)
    print(f"Saved 200x200 to {path_200}")

if __name__ == "__main__":
    test_resize()

