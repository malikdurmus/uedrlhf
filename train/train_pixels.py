import os
from pathlib import Path

# --- THE NUCLEAR OPTION FOR JAX/CUDNN CRASHES ---
# 1. Stop XLA from testing heavy algorithms that crash the GPU
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0" 
# 2. Force cuDNN to use stable, deterministic algorithms
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# 3. Only let JAX take 70% of the VRAM. Leave 30% for cuDNN's workspace.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"
os.environ["CUDNN_LOGDEST_DBG"] = "stderr"  # see what cuDNN is actually doing
os.environ["CUDNN_LOGINFO_DBG"] = "1"
# --- SLURM FIX: Reroute the temporary directory before JAX loads ---

os.environ["TF_CUDNN_USE_FRONTEND"] = "0"   # ← KEY: disable cuDNN v8 frontend API

script_path = Path(__file__).resolve()
base_dir = script_path.parent.parent  

import numpy as np
from functools import partial
from torch.utils.data import DataLoader

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from datasets import load_from_disk

# -configs
IMG_SIZE = 64  
BATCH_SIZE = 8
EPOCHS = 20

# paths 
img_dir = base_dir / "data" / "images"
db_path = base_dir / "data" / "feedback.db"
save_dir = base_dir / "train"
save_dir.mkdir(parents=True, exist_ok=True)

class CNN(nnx.Module):
    def __init__(self, * , rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_features= 3 , out_features= 32 , kernel_size = (5,5), padding="VALID",rngs= rngs)
        self.batch_norm1 = nnx.BatchNorm(num_features=32 , rngs = rngs)
        self.dropout1 = nnx.Dropout(rate= 0.025)
        self.conv2 = nnx.Conv(in_features = 32 , out_features= 64 , kernel_size= (3,3), padding="VALID" , rngs= rngs )
        self.batch_norm2 = nnx.BatchNorm(num_features=64 , rngs = rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2,2),strides=(2,2))
        # we shape = (14,14,64)
        self.linear1 = nnx.Linear(in_features=12544, out_features= 256, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=0.025, rngs=rngs)
        self.linear2 = nnx.Linear(in_features=256, out_features=1, rngs= rngs)

    def __call__(self, x, rngs : nnx.Rngs | None = None):
        jax.debug.print("here model called ")
        x = self.conv1(x)
        jax.debug.print("first conv done ")
        x = self.dropout1(x,rngs=rngs)
        x = self.batch_norm1(x)
        x = nnx.relu(x)
        x = self.avg_pool(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = nnx.relu(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0],-1) # flatten
        # now shape # (12544,)
        x = nnx.relu(self.dropout2(self.linear1(x) , rngs= rngs))
        x = self.linear2(x)
        x = x.squeeze()  # return scalar value
        return x


@nnx.jit
def loss_fn(model: CNN, data: tuple):
    #jax.debug.breakpoint()
    img0, img1, preferences = data
    combined_inputs = jnp.concatenate([img0, img1], axis = 0)

    scores = model(combined_inputs)
    scores_input0 = model(img0) 
    scores_input1 = model(img1) # jnp.split(scores, 2, axis = 0 ) 

    logits = (scores_input0 - scores_input1) 
    batch_loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=preferences)
    loss = jnp.mean(batch_loss)
    return loss, logits

@nnx.jit
def train_step(model: CNN, optimizer: nnx.Optimizer , rngs: nnx.Rngs, batch: tuple):
    #jax.debug.breakpoint()
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    jax.debug.print("here 85")
    (loss , logits), grads = grad_fn(model=model, data = batch)
    jax.debug.print("here 87 ")

    optimizer.update(model, grads)
    jax.debug.print("here 95 update done  ")
    # MANUAL ACCURACY CALCULATION
    labels = batch[-1]
    predictions = (logits > 0).astype(jnp.float32) # >0 means img1 won, <=0 means img0 won
    accuracy = jnp.mean(predictions == labels)
    
    return loss, accuracy

@nnx.jit
def eval_step(model: CNN, batch: tuple):
    loss, logits = loss_fn(model, batch)
    
    # MANUAL ACCURACY CALCULATION
    labels = batch[-1]
    predictions = (logits > 0).astype(jnp.float32)
    accuracy = jnp.mean(predictions == labels)
    
    return loss, accuracy

def main():

    ds = load_from_disk("data/difficulty/compiled_gemini_2_5_flash")
    ds_splitted = ds.train_test_split(test_size=0.1)
    
    columns = ['env_0_image', 'env_1_image', 'preference']
    ds_splitted.set_format(type="numpy", columns=columns)

    train_loader = DataLoader(
        ds_splitted["train"], 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        drop_last=True, 
        num_workers=0)

    eval_loader = DataLoader(
        ds_splitted["test"], 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        drop_last=True,
        num_workers=0)

    rngs = nnx.Rngs(0)
    model = CNN(rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    
    train_model = nnx.view(model, deterministic=True, use_running_average=False)
    eval_model = nnx.view(model, deterministic=True, use_running_average=True)
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        
        epoch_train_losses = []
        epoch_train_accs = []

        # --- TRAIN ---
        for batch_dict in train_loader:
            img0 = jnp.array(batch_dict['env_0_image'])
            img1 = jnp.array(batch_dict['env_1_image'])
            
            img0_resized = jax.image.resize(img0, (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3), method='bilinear')
            img1_resized = jax.image.resize(img1, (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3), method='bilinear')

            img0_resized = img0_resized / 255.0
            img1_resized = img1_resized / 255.0

            batch_tuple = (img0_resized, img1_resized, jnp.array(batch_dict['preference']))
            
            loss, acc = train_step(model=train_model, optimizer=optimizer, rngs=rngs, batch=batch_tuple)

            epoch_train_losses.append(loss.item())
            epoch_train_accs.append(acc.item())

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {np.mean(epoch_train_losses):.4f}, Train Acc: {np.mean(epoch_train_accs):.4f}")

        epoch_eval_losses = []
        epoch_eval_accs = []

        # --- EVAL ---
        for batch_dict in eval_loader:
            img0 = jnp.array(batch_dict['env_0_image'])
            img1 = jnp.array(batch_dict['env_1_image'])
            
            img0_resized = jax.image.resize(img0, (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3), method='bilinear')
            img1_resized = jax.image.resize(img1, (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3), method='bilinear')
        
            img0_resized = img0_resized / 255.0
            img1_resized = img1_resized / 255.0


            batch_tuple = (img0_resized, img1_resized, jnp.array(batch_dict['preference']))
            
            loss, acc = eval_step(eval_model, batch_tuple)
            
            epoch_eval_losses.append(loss.item())
            epoch_eval_accs.append(acc.item())
                
        print(f"Epoch {epoch+1}/{EPOCHS} - Eval Loss: {np.mean(epoch_eval_losses):.4f}, Eval Acc: {np.mean(epoch_eval_accs):.4f}")

    print("Training done")

if __name__ == "__main__":
    main()