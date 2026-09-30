import xminigrid
import jax

key = jax.random.key(0)
key, reset_key = jax.random.split(key=key)

env, env