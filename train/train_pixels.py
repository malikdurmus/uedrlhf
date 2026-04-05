import os
import sqlite3
import numpy as np
from PIL import Image
from pathlib import Path
from functools import partial
import matplotlib.pyplot as plt
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
LEARNING_RATE = 3e-4
MOMENTUM = 0.9

train_steps = 1200
eval_every = 200

# path 
script_path = Path(__file__).resolve()
base_dir = script_path.parent.parent  
img_dir = base_dir / "data" / "images"
db_path = base_dir / "data" / "feedback.db"

save_dir = base_dir / "train"
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "model_pixels.msgpack"


# model takes a batch of images : n, 64, 64 , 3 (64x64 image and 3 channels )
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
def train_step(model: CNN, optimizer: nnx.Optimizer ,metrics: nnx.MultiMetric, rngs: nnx.Rngs, batch: tuple):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss , logits), grads = grad_fn(model=model, data = batch)
    
    metric_labels = batch[-1].astype(jnp.int32)
    metrics.update(loss = loss, logits= logits, labels=metric_labels)
    optimizer.update(model, grads)

@nnx.jit
def eval_step(model: CNN, metrics, batch ):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch[-1]) 

def main():
    metrics_history = {
        'train_loss': [], 'train_accuracy': [],
        'test_loss': [], 'test_accuracy': [],
    }

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
        drop_last=True,
        num_workers=4)

    rngs = nnx.Rngs(0)
    model = CNN(rngs=rngs)
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(threshold=0.5),
        loss=nnx.metrics.Average('loss'),
        
    )
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    train_model = nnx.view(model, deterministic=True, use_running_average=False)
    eval_model = nnx.view(model, deterministic=True, use_running_average=True)
    
    print("Starting training...")
    for epoch in range (EPOCHS):

        # train
        for batch_dict in train_loader:
            
            img0 = jnp.array(batch_dict['env_0_image'])
            img1 = jnp.array(batch_dict['env_1_image'])
            
            
            img0_resized = jax.image.resize(img0, (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3), method='bilinear')
            img1_resized = jax.image.resize(img1, (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3), method='bilinear')
            
            batch_tuple = (
                img0_resized,
                img1_resized,
                jnp.array(batch_dict['preference'])
            )
            
            train_step(model=train_model, optimizer=optimizer, metrics=metrics, rngs=rngs, batch=batch_tuple)

        train_metrics = metrics.compute()
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        metrics.reset()

        for batch_dict in eval_loader:
            
            img0 = jnp.array(batch_dict['env_0_image'])
            img1 = jnp.array(batch_dict['env_1_image'])
            
            img0_resized = jax.image.resize(img0, (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3), method='bilinear')
            img1_resized = jax.image.resize(img1, (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3), method='bilinear')

            batch_tuple = (
                img0_resized,
                img1_resized,
                jnp.array(batch_dict['preference'])
            )
            eval_step(eval_model, metrics, batch_tuple)
                
        eval_metrics = metrics.compute()
        print(f"Epoch {epoch+1}/{EPOCHS} - Eval Loss: {eval_metrics['loss']:.4f}, Eval Acc: {eval_metrics['accuracy']:.4f}")
        metrics.reset()

    print("Training done")

if __name__ == "__main__":
    main()