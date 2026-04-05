import os
from pathlib import Path
import pdb
from queue import Queue
import threading
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
BATCH_SIZE = 512
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
        self.dropout1 = nnx.Dropout(rate= 0.025,rngs= rngs)
        self.conv2 = nnx.Conv(in_features = 32 , out_features= 64 , kernel_size= (3,3), padding="VALID" , rngs= rngs )
        self.batch_norm2 = nnx.BatchNorm(num_features=64 , rngs = rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2,2),strides=(2,2))
        # we shape = (14,14,64)
        self.linear1 = nnx.Linear(in_features=12544, out_features= 256, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=0.025, rngs=rngs)
        self.linear2 = nnx.Linear(in_features=256, out_features=1, rngs= rngs)

    def __call__(self, x, rngs : nnx.Rngs | None = None):
        x = self.conv1(x)
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
    img0, img1, preferences = data
    combined_inputs = jnp.concatenate([img0, img1], axis = 0)

    scores = model(combined_inputs)
    scores_input0 , scores_input1 = jnp.split(scores, 2, axis = 0 ) 

    logits = (scores_input0 - scores_input1) 
    batch_loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=preferences)
    loss = jnp.mean(batch_loss)
    return loss, logits

@nnx.jit
def train_step(model: CNN, optimizer: nnx.Optimizer , rngs: nnx.Rngs, batch: tuple):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss , logits), grads = grad_fn(model=model, data = batch)

    optimizer.update(model, grads)

    labels = batch[-1]
    predictions = (logits > 0).astype(jnp.float32) # >0 means img1 won, <=0 means img0 won
    accuracy = jnp.mean(predictions == labels)
    
    return loss, accuracy

@nnx.jit
def eval_step(model: CNN, batch: tuple):
    loss, logits = loss_fn(model, batch)
    
    labels = batch[-1]
    predictions = (logits > 0).astype(jnp.float32)
    accuracy = jnp.mean(predictions == labels)
    
    return loss, accuracy


def prefetch_to_gpu(dataloader, queue_size = 2 ):
    queue = Queue(maxsize= queue_size)

    def producer():
        for batch_dict in dataloader:
            batch = (
                jnp.array(batch_dict['env_0_image']),
                jnp.array(batch_dict['env_1_image']),
                jnp.array(batch_dict['preference'])
            )
            queue.put(batch)
        
        queue.put(None) # signal
    t = threading.Thread(target=producer, daemon=True)
    t.start()

    while True:
        batch = queue.get()
        if batch is None:
            break
        yield batch



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
        num_workers=4)

    eval_loader = DataLoader(
        ds_splitted["test"], 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        drop_last=False,
        num_workers=4)

    rngs = nnx.Rngs(0)
    model = CNN(rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    
    train_model = nnx.view(model, deterministic=False, use_running_average=False)
    eval_model = nnx.view(model, deterministic=True, use_running_average=False)
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        
        epoch_train_losses = []
        epoch_train_accs = []

        # --- TRAIN ---
        model.train()
        for batch_tuple in prefetch_to_gpu(train_loader):
                  
            loss, acc = train_step(model=model, optimizer=optimizer, rngs=rngs, batch=batch_tuple)

            epoch_train_losses.append(loss.item())
            epoch_train_accs.append(acc.item())

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {np.mean(epoch_train_losses):.4f}, Train Acc: {np.mean(epoch_train_accs):.4f}")

        epoch_eval_losses = []
        epoch_eval_accs = []

        # --- EVAL ---
        model.eval()

        for batch_tuple in prefetch_to_gpu(eval_loader):
            loss, acc = eval_step(model, batch_tuple)
            epoch_eval_losses.append(loss.item())
            epoch_eval_accs.append(acc.item())
        print(f"Epoch {epoch+1}/{EPOCHS} - Eval Loss: {np.mean(epoch_eval_losses):.4f}, Eval Acc: {np.mean(epoch_eval_accs):.4f}")

    print("Training done")

if __name__ == "__main__":

    main()