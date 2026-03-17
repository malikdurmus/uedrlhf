import jax
import jax.numpy as jnp

from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav
from jaxrobotarium.robotarium_env import RobotariumEnv, State


def main():
    print("--- Starting JAX Smoke Test ---")
    print(jax.devices())

    # 1. Initialize the Environment
    # This might fail immediately if imports are broken or if
    # JAX versions conflict with class structures.
    try:
        env = JaxNav(num_agents=2, map_id="Grid-Rand-Poly")
        print("Successfully initialized JaxNav environment.")
    except Exception as e:
        print(f"\n[INIT ERROR] Initialization failed:\n{e}")
        return

    # 2. Setup Random Keys
    # This is where strict PRNGKey typing in JAX 0.9 often causes issues.
    rng = jax.random.PRNGKey(42)
    rng, key_reset = jax.random.split(rng)

    # 3. Attempt Reset
    print("\nAttempting environment reset...")
    try:
        obs, state = env.reset(key_reset)
        print("Successfully reset environment.")
    except Exception as e:
        print(f"\n[RESET ERROR] Reset failed:\n{e}")
        return

    # 4. Attempt a Step
    print("\nAttempting environment step...")
    try:
        # Create dummy actions for 2 agents
        # 'agent_0', 'agent_1' are the names JaxNav uses
        actions = {"agent_0": jnp.array([0.5, 0.0]), "agent_1": jnp.array([0.5, 0.0])}
        rng, key_step = jax.random.split(rng)

        # Taking a step involves JIT compilation usually.
        # This is where most JAX-specific tracer errors occur.
        obs, state, reward, done, info = env.step(key_step, state, actions)

        print("Successfully took a step!")
        print(f"Step Count: {state.step}")
        print(f"Rewards: {reward}")

    except Exception as e:
        print(f"\n[STEP ERROR] Step failed:\n{e}")
        # Hint: If this fails with a 'Tracer' error, it means we are
        # mixing Python side-effects with JAX JIT compilation.


if True:
    main()
