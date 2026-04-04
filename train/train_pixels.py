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

# -config
IMG_SIZE = 64  
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 3e-4
MOMENTUM = 0.9


tf.random.set_seed(0)  # Set the random seed for reproducibility.

train_steps = 1200
eval_every = 200
batch_size = 32



# path To do: rework
script_path = Path(__file__).resolve()
base_dir = script_path.parent.parent  # 'uedrlhf'
img_dir = base_dir / "data" / "images"
db_path = base_dir / "data" / "feedback.db"
print(f"Base Dir: {base_dir}")
print(f"DB Path: {db_path}")

save_dir = base_dir / "train"
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "model_pixels.msgpack"



def load_dataset(split: str = "train"):
    """Load image pairs and preferences for a given split ('train' or 'test')."""
    """returns: img1 shape (n, IMG_SIZE ,IMG_SIZE, 3)
                img2 shape (n, IMG_SIZE ,IMG_SIZE, 3)
                pref shape (n, IMG_SIZE ,IMG_SIZE, 3)
     """

    # LOAD data
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT env_1_id, env_2_id, preference 
        FROM preference_pairs 
        WHERE status='completed' AND split=?
    """, (split,))
    rows = cursor.fetchall()
    conn.close()
    num_rows = len(rows)
    if len(rows) == 0:
        raise ValueError(f"No completed {split} data found!")

    # initialize arrays that hold the data
    img1_array = np.empty( shape = (num_rows, IMG_SIZE, IMG_SIZE, 3), dtype = np.float32 )
    img2_array = np.empty( shape = (num_rows, IMG_SIZE, IMG_SIZE, 3), dtype = np.float32 )
    pref_array = np.empty(shape = (num_rows, IMG_SIZE, IMG_SIZE, 3) , dtype = np.float32 )

    # loop over and load the data
    for i, (env_1_id, env_2_id, pref) in enumerate(rows):

        img1 = Image.open(img_dir / f"{env_1_id}.png").convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img2 = Image.open(img_dir / f"{env_2_id}.png").convert("RGB").resize((IMG_SIZE, IMG_SIZE))

        img1_array[i] = img1
        img2_array[i] = img2
        pref_array[i] = pref


    print(f"Loaded {len(rows)} pairs from '{split}' split.")
    return np.array(img1_array), np.array(img2_array), np.array(pref_array)





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
        self.linear1 = nnx.Linear(in_features=57344, out_features= 256, rngs=rngs)
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

    combined_inputs = jnp.concatenate(img0, img1, axis = 0)

    scores = model(combined_inputs) # this is done for efficient computing


   
    scores_input0 , scores_input1 = jnp.split(combined_inputs, 2, axis = 0 ) 
    #score for image 0 , score for image 1


    # raw scores (optimized for calculation)
    logits = (scores_input0-scores_input0) # >0 if image 0 is preffered <0 otherwise

    loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=preferences)
    
    return loss, logits

def train_step(model: CNN, optimizer: nnx.Optimizer ,metrics: nnx.MultiMetric, rngs: nnx.Rngs,
               batch : jax.Array
               ):
    grad_fn = nnx.value_and_grad(loss_fn)
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
    


    ds_train = load_dataset("train")
    ds_test  = load_dataset("test")

    rngs = nnx.Rngs(0)
    model = CNN(rngs=rngs)
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average('loss'),
    )
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    train_model = nnx.view(model, deterministic=False, use_running_average=False)
    eval_model = nnx.view(model, deterministic=True, use_running_average=True)
    


    for step, batch in enumerate(ds_train.as_numpy_iterator()):
    
  # Run the optimization for one step and make a stateful update to the following:
  # - The train state's model parameters
  # - The optimizer state
  # - The training loss and accuracy batch metrics
        train_step(train_model, optimizer, metrics, rngs, batch)

        if step > 0 and (step % eval_every == 0 or step == train_steps - 1):  # One training epoch has passed.
            # Log the training metrics.
            # for metric, value in metrics.compute().items():  # Compute the metrics.
            metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
            metrics.reset()  # Reset the metrics for the test set.

            # Compute the metrics on the test set after each training epoch.
            for test_batch in ds_train.as_numpy_iterator():
                eval_step(eval_model, metrics, test_batch)

            # Log the test metrics.
            for metric, value in metrics.compute().items():
                metrics_history[f'test_{metric}'].append(value)
            metrics.reset()  # Reset the metrics for the next training epoch.

            # Plot loss and accuracy in subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            ax1.set_title('Loss')
            ax2.set_title('Accuracy')
            for dataset in ('train', 'test'):
                ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
                ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
            ax1.legend()
            ax2.legend()
            plt.show()





main()




