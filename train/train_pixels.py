import os
#os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
#os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'
#os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
import sqlite3
import numpy as np

from PIL import Image
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from flax.serialization import to_bytes
import optax

# --- CONFIGURATION ---
IMG_SIZE = 64  
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 3e-4
MOMENTUM = 0.9

# --- PATH SETUP ---
script_path = Path(__file__).resolve()
base_dir = script_path.parent.parent  # 'uedrlhf'
img_dir = base_dir / "data" / "images"
db_path = base_dir / "data" / "feedback.db"
print(f"Base Dir: {base_dir}")
print(f"DB Path: {db_path}")

save_dir = base_dir / "train"
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "model_pixels.msgpack"


# --- 1. DATA LOADER ---
def load_dataset():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Grab all completed training pairs
    cursor.execute("""
        SELECT env_1_id, env_2_id, preference 
        FROM preference_pairs 
        WHERE status='completed' AND split='train'
    """)
    rows = cursor.fetchall()
    conn.close()

    if len(rows) == 0:
        raise ValueError("No completed training data found!")

    img1_list, img2_list, pref_list = [], [], []
    
    for env_1_id, env_2_id, pref in rows:
        # Load, resize, and normalize images to [0, 1]
        img1 = Image.open(img_dir / f"{env_1_id}.png").convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img2 = Image.open(img_dir / f"{env_2_id}.png").convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        
        img1_list.append(np.array(img1, dtype=np.float32) / 255.0)
        img2_list.append(np.array(img2, dtype=np.float32) / 255.0)
        pref_list.append(np.float32(pref))

    print(f"Loaded {len(rows)} human-annotated pairs.")
    return np.array(img1_list), np.array(img2_list), np.array(pref_list)


# --- 2. NETWORK DEFINITION ---
class PixelPrefCNN(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(3, 32, kernel_size=(5, 5), rngs=rngs)
        self.batch_norm1 = nnx.BatchNorm(32, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=0.025)
        
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.batch_norm2 = nnx.BatchNorm(64, rngs=rngs)
        
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        
        self.linear1 = nnx.Linear(16384, 256, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=0.025)
        self.linear2 = nnx.Linear(256, 1, rngs=rngs)  

    def __call__(self, x, rngs: nnx.Rngs | None = None):
        x = self.avg_pool(nnx.relu(self.batch_norm1(self.dropout1(self.conv1(x), rngs=rngs))))
        x = self.avg_pool(nnx.relu(self.batch_norm2(self.conv2(x))))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.dropout2(self.linear1(x), rngs=rngs))
        x = self.linear2(x)
        return x.squeeze(-1) # scalar


# --- 3. TRAINING LOGIC ---
def loss_fn(model: PixelPrefCNN, img1_batch, img2_batch, pref_batch, rngs: nnx.Rngs):
    
    score_1 = model(img1_batch, rngs=rngs)
    score_2 = model(img2_batch, rngs=rngs)
    
    # Bradley-Terry Preference Loss
    log_p1_wins = jax.nn.log_sigmoid(score_1 - score_2)
    log_p2_wins = jax.nn.log_sigmoid(score_2 - score_1)
    loss = -jnp.mean(pref_batch * log_p1_wins + (1.0 - pref_batch) * log_p2_wins)
    
    return loss, (score_1, score_2)


@nnx.jit
def train_step(model: PixelPrefCNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, rngs: nnx.Rngs, img1_batch, img2_batch, pref_batch):
    
    # Calculate gradients
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (score_1, score_2)), grads = grad_fn(model, img1_batch, img2_batch, pref_batch, rngs)

    # Calculate custom accuracy metric
    correct = jnp.where(pref_batch >= 0.5, score_1 > score_2, score_2 > score_1)
    accuracy = jnp.mean(correct)
    
    # In-place updates for metrics and optimizer.
    metrics.update(loss=loss, accuracy=accuracy) 
    optimizer.update(model, grads)


# --- 4. EXECUTION ---
def main():
    # Load Data
    X1, X2, Y = load_dataset()
    num_samples = len(Y)

    # Create RNGs for parameter initialization AND dropout noise during training
    rngs = nnx.Rngs(params=0, dropout=1)
    
    # Initialize the model
    model = PixelPrefCNN(rngs=rngs)
    
    # Initialize Optimizer
    optimizer = nnx.Optimizer(
        model, optax.adamw(LEARNING_RATE, b1=MOMENTUM), wrt=nnx.Param
    ) 

    # Create Training and Eval Views
    train_model = nnx.view(model, deterministic=False, use_running_average=False)
    eval_model = nnx.view(model, deterministic=True, use_running_average=True)

    # Initialize Metrics to track averages over the epoch
    # (New code)
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average('loss'),          
        accuracy=nnx.metrics.Average('accuracy') 
    )

    print("Starting Training...")
    
    indices = np.arange(num_samples)
    
    for epoch in range(EPOCHS):
        np.random.shuffle(indices)

        for i in range(0, num_samples, BATCH_SIZE):
            batch_idx = indices[i:i + BATCH_SIZE]
            
            # Skip incomplete final batch if it's too small
            if len(batch_idx) < 2:
                continue
                
            img1_batch = jnp.array(X1[batch_idx])
            img2_batch = jnp.array(X2[batch_idx])
            pref_batch = jnp.array(Y[batch_idx])
            
            # Call train_step! NNX magically updates model, optimizer, and metrics in-place.
            train_step(train_model, optimizer, metrics, rngs, img1_batch, img2_batch, pref_batch)
            
        # --- EPOCH END LOGGING ---
        # Compute the accumulated metrics for the epoch
        metric_values = metrics.compute()
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {metric_values['loss']:.4f} | Accuracy: {metric_values['accuracy']*100:.1f}%")
        
        # Reset metrics for the next epoch
        metrics.reset()

    # --- SAVE THE MODEL ---
    # We extract the state from the base 'model' object, not the view.
    model_state = nnx.state(model)
    with open(save_path, "wb") as f:
        f.write(to_bytes(model_state))
    
    print(f"Model successfully saved to {save_path}")

if __name__ == "__main__":
    main()