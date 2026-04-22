"""Neural network architectures for reward model training."""

from functools import partial
import jax.numpy as jnp
from flax import nnx


class CNN(nnx.Module):
    """Preference prediction CNN.

    Takes a (B, H, W, 3) image batch and produces a scalar score per image.
    Used in pairwise preference learning: score(env_0) - score(env_1) → logit.
    """

    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_features=3, out_features=32, kernel_size=(5, 5), padding="VALID", rngs=rngs)
        self.batch_norm1 = nnx.BatchNorm(num_features=32, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=0.025, rngs=rngs)
        self.conv2 = nnx.Conv(in_features=32, out_features=64, kernel_size=(3, 3), padding="VALID", rngs=rngs)
        self.batch_norm2 = nnx.BatchNorm(num_features=64, rngs=rngs)
        self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        # After conv+pool layers on 64x64 input: shape = (14, 14, 64) = 12544
        self.linear1 = nnx.Linear(in_features=12544, out_features=256, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=0.025, rngs=rngs)
        self.linear2 = nnx.Linear(in_features=256, out_features=1, rngs=rngs)

    def __call__(self, x, rngs: nnx.Rngs | None = None):
        x = self.conv1(x)
        x = self.dropout1(x, rngs=rngs)
        x = self.batch_norm1(x)
        x = nnx.relu(x)
        x = self.avg_pool(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = nnx.relu(x)
        x = self.avg_pool(x)

        x = x.reshape(x.shape[0], -1)  # flatten

        x = nnx.relu(self.dropout2(self.linear1(x), rngs=rngs))
        x = self.linear2(x)
        x = x.squeeze()  # return scalar value
        return x

