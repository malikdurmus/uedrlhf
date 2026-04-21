import time
import os
from pathlib import Path
import pdb
from queue import Queue
import threading
import orbax.checkpoint as ocp
script_path = Path(__file__).resolve()
base_dir = script_path.parent.parent  

import numpy as np
from functools import partial
from torch.utils.data import DataLoader
import wandb
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from datasets import load_from_disk

# -configs
IMG_SIZE = 64  
BATCH_SIZE = 512
EPOCHS = 50
dataset_path= "data/difficulty/compiled_gemini_2_5_flash_20k"
model_type= dataset_path[5:]

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

    wandb.init(
        project="uedrlhf",          # groups all runs for this project
        name="cnn-difficulty-20k",   # name for this specific run
        config={                      
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "img_size": IMG_SIZE,
            "learning_rate": 1e-3,
            "optimizer": "adam",
            "dropout": 0.025,
        }
    )
    ds = load_from_disk(dataset_path)
    ds_splitted = ds.train_test_split(test_size=0.1)
    

    columns = ['env_0_image', 'env_1_image', 'preference']
    ds_splitted.set_format(type="numpy", columns=columns)

    train_loader = DataLoader(
        ds_splitted["train"], 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        drop_last=False, 
        num_workers=4)

    eval_loader = DataLoader(
        ds_splitted["test"], 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        drop_last=False,
        num_workers=4)

    rngs = nnx.Rngs(0)
    model = CNN(rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=1e-3), wrt=nnx.Param)
    
    train_model = nnx.view(model, deterministic=False, use_running_average=False)
    eval_model = nnx.view(model, deterministic=True, use_running_average=False)

    best_eval_loss = float("inf") # keeps track of the best model/ loss 

    # ----- TRAINING & EVAL LOOP
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
        
        # ------ avg in batch
        train_loss_avg= np.mean(epoch_train_losses)
        train_acc_avg = np.mean(epoch_train_accs)

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc_avg:.4f}")

        epoch_eval_losses = []
        epoch_eval_accs = []

        # --- EVAL ---
        model.eval() # switch to eval view

        for batch_tuple in prefetch_to_gpu(eval_loader):
            loss, acc = eval_step(model, batch_tuple)
            epoch_eval_losses.append(loss.item())
            epoch_eval_accs.append(acc.item())

        eval_loss_avg = np.mean(epoch_eval_losses)
        eval_acc_avg = np.mean(epoch_eval_accs)
        if eval_loss_avg < best_eval_loss:
            best_eval_loss = eval_loss_avg


        print(f"Epoch {epoch+1}/{EPOCHS} - Eval Loss: {eval_loss_avg:.4f}, Eval Acc: {eval_acc_avg:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss_avg,
            "train/accuracy": train_acc_avg,
            "eval/loss": eval_loss_avg,
            "eval/accuracy": eval_acc_avg,
            "eval/loss_gap": eval_loss_avg - train_loss_avg,  # overfitting signal
            "best_eval_loss": best_eval_loss,
        })


    wandb.finish()      
    _, state = nnx.split(model)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = base_dir / "orbaxexport" / timestamp / model_type

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(save_path, state)
    checkpointer.wait_until_finished()

    print("Training done")
    

if __name__ == "__main__":
    main()