import os
import sqlite3
from pyexpat import features
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from functools import partial
import matplotlib.pyplot as plt
from flax.nnx import Rngs
from keras.src.ops import dtype
from orbax.export import JaxModule, ExportManager, ServingConfig
import jax
import jax.numpy as jnp
from flax import nnx
from flax.serialization import to_bytes
import optax
from datasets import load_from_disk


# -configs
IMG_SIZE = 64  
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 3e-4
MOMENTUM = 0.9


tf.random.set_seed(0)  # Set the random seed for reproducibility.

train_steps = 1200
eval_every = 200
batch_size = 32



# path TODO: rework LATER
script_path = Path(__file__).resolve()
base_dir = script_path.parent.parent  # 'uedrlhf'
img_dir = base_dir / "data" / "images"
db_path = base_dir / "data" / "feedback.db"
print(f"Base Dir: {base_dir}")
print(f"DB Path: {db_path}")

save_dir = base_dir / "train"
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "model_pixels.msgpack"

# model takes a batch of images : n, 64, 64 , 3 (64x64 image and 3 channels )
class CNN(nnx.Module):

    def __init__(self, * , rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_features= 3 , out_features= 32 , kernel_size = (5,5),rngs= rngs)
        self.batch_norm1 = nnx.BatchNorm(num_features=32 , rngs = rngs)
        self.dropout1 = nnx.Dropout(rate= 0.025)
        self.conv2 = nnx.Conv(in_features = 32 , out_features= 64 , kernel_size= (3,3) , rngs= rngs )
        self.batch_norm2 = nnx.BatchNorm(num_features=64 , rngs = rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2,2),strides=(2,2))
        # we shape = (14,14,64)
        self.linear1 = nnx.Linear(in_features=12544, out_features= 256, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=0.025, rngs=rngs)
        self.linear2 = nnx.Linear(in_features=256, out_features=1, rngs= rngs)

    def __call__(self, x, rngs : nnx.Rngs | None = None):
        x= self.avg_pool(
            nnx.relu(
                self.batch_norm1(
                    self.dropout1(
                        self.conv1(x),rngs) ) ) )
        # now shape 30,30,32

        x = self.avg_pool(
            nnx.relu(
                self.batch_norm2(
                        self.conv2(x), rngs) ) )
        # now shape 14,14,64
        x = x.reshape(x.shape[0],-1) # flatten
        # now shape # (12544,)
        x = nnx.relu(self.dropout2(self.linear1(x , rngs)))
        x = self.linear2(x)
        x = x.squeeze()  # return scalar value
        return x

def loss_fn(model: CNN, data: tuple):
    """
    data: (image0: Image, image1: Image , preference: jax.Array)
    
    preferences: (Batch,) where 0.0 means img0 is preferred, 1 for img1
    """
    
    img0, img1, preferences = data

    combined_inputs = jnp.concatenate([img0, img1], axis = 0)

    scores = model(combined_inputs)

    scores_input0 , scores_input1 = jnp.split(scores, 2, axis = 0 ) 
    #score for image 0 , score for image 1


    # raw scores (optimized for calculation)
    logits = (scores_input0 - scores_input1) # >0 if image 0 is preffered <0 otherwise

    batch_loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=preferences)
    loss = jnp.mean(batch_loss)
    
    return loss, logits

def train_step(model: CNN, optimizer: nnx.Optimizer ,metrics: nnx.MultiMetric, rngs: nnx.Rngs,
               batch : tuple):

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss , logits), grads = grad_fn(model=model, data = batch)
    metrics.update(loss = loss, logits= logits, labels=batch[-1])
    
    optimizer.update(model, grads)

def eval_step(model: CNN, metrics, batch ):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch)

def main():
    
    metrics_history = {
    'train_loss': [],
    'train_accuracy': [],
    'test_loss': [],
    'test_accuracy': [],
    }
    

    ds = load_from_disk("data/difficulty/compiled_gemini_2_5_flash")

    # TODO will change this later to run (Nested)-CV, without relying on the split attribute
    ds_splitted = ds.train_test_split(test_size=0.1)
    #ds_train_raw = ds.filter(lambda x : x["split"] == "train")
    #ds_train_raw = ds.filter(lambda x : x["split"] == "test")

    columns = ['env_0_image', 'env_1_image', 'preference']
    def resize_batch(batch_dict):
        # tf.image.resize expects shape [batch, height, width, channels]
        # It automatically squashes the 500x500 images down to 64x64
        batch_dict['env_0_image'] = tf.image.resize(batch_dict['env_0_image'], [IMG_SIZE, IMG_SIZE])
        batch_dict['env_1_image'] = tf.image.resize(batch_dict['env_1_image'], [IMG_SIZE, IMG_SIZE])
        return batch_dict


    # create training loader
    train_loader = ds_splitted["train"].to_tf_dataset(
        columns=columns,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_remainder=True,
        prefetch=tf.data.AUTOTUNE
    ).map(resize_batch, num_parallel_calls=tf.data.AUTOTUNE)

    # create eval loader
    eval_loader = ds_splitted["test"].to_tf_dataset(
        columns=columns,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_remainder=True,
        prefetch=tf.data.AUTOTUNE
    ).map(resize_batch, num_parallel_calls=tf.data.AUTOTUNE)


    rngs = nnx.Rngs(0)
    model = CNN(rngs=rngs)
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average('loss'),
    )
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    train_model = nnx.view(model, deterministic=False, use_running_average=False)
    eval_model = nnx.view(model, deterministic=True, use_running_average=True)
    
    
    for epoch in range (EPOCHS):

        for batch_dict in train_loader.as_numpy_iterator():
            batch_tuple = (
                jnp.array(batch_dict['env_0_image']),
                jnp.array(batch_dict['env_1_image']),
                jnp.array(batch_dict['preference'])
                )
            train_step(model= train_model , optimizer= optimizer, metrics= metrics, rngs= rngs, batch= batch_tuple)

        train_metrics = metrics.compute()
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        # reset metrics for evaluation
        metrics.reset()

        for batch_dict in eval_loader.as_numpy_iterator():
            batch_tuple = (
                jnp.array(batch_dict['env_0_image']),
                jnp.array(batch_dict['env_1_image']),
                jnp.array(batch_dict['preference'])
                )
            eval_step(eval_model, metrics, batch_tuple)
                
        eval_metrics = metrics.compute()

        print(f"Epoch {epoch+1}/{EPOCHS} - Eval Loss: {eval_metrics['loss']:.4f}, Eval Acc: {eval_metrics['accuracy']:.4f}")
        metrics.reset()


    print("Training done")

main()


