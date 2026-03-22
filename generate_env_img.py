from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav
from jaxmarl.environments.jaxnav.jaxnav_viz import JaxNavVisualizer
import jax
import matplotlib.pyplot as plt
import uuid
import os 

env = JaxNav(num_agents=1)

rng = jax.random.PRNGKey(0)
reset_rng = jax.random.split(rng, 100 )
obs, env_state = jax.vmap(env.reset) (reset_rng)
rng , _rng = jax.random.split(rng)

#obs_list = [obs]
#env_state_list = [env_state]
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

for i in range(11):
    state_id = uuid.uuid

    #extract a single state
    state = jax.tree_util.tree_map(lambda x : x[i] , env_state)
    # render the env image
    env.init_render(ax, state , lidar=False,  # plot lidar?
                    agent=False,  # plot agents?
                    goal=False, )
    # save env image
    if not os.path.exists("images"):
        os.makedirs("images")

    plt.savefig("images/" + str(i) + "__"+ str(state_id))


#viz = JaxNavVisualizer(env, obs_list, env_state_list)
#viz.animate("test.gif")
