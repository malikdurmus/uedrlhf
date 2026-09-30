"""
test_minigrid _generation.py — Generate and render multiple XLand-MiniGrid levels for Easy, Medium, and Difficult rulesets.

Uses XLand-MiniGrid-R4-13x13.
Renders each layout with the JAX tile-cache renderer (jax_renderer.py),
saves PNG images, and prints LLM-ready prompts for each difficulty level.
"""

import os
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

import xminigrid
from xminigrid.types import RuleSet
from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
from xminigrid.core.goals import AgentHoldGoal
from xminigrid.core.rules import TileNearRule, AgentHoldRule, AgentNearRule
from xminigrid.rendering.text_render import print_ruleset

# Import the JAX renderer from the local file
from uedrlhf.jax_renderer import build_tile_caches, jax_render


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
ENV_ID = "XLand-MiniGrid-R4-13x13"
NUM_LEVELS_PER_RULESET = 6
TILE_SIZE = 32          # pixels per grid cell
VIEW_SIZE = 5           # agent FOV (used for highlight)
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rendered_levels")
SEED = 42

# ──────────────────────────────────────────────────────────────
# Define tiles
# ──────────────────────────────────────────────────────────────
yellow_ball    = TILES_REGISTRY[Tiles.BALL,    Colors.YELLOW]
blue_key       = TILES_REGISTRY[Tiles.KEY,     Colors.BLUE]
green_star     = TILES_REGISTRY[Tiles.STAR,    Colors.GREEN]
purple_hex     = TILES_REGISTRY[Tiles.HEX,     Colors.PURPLE]
pink_square    = TILES_REGISTRY[Tiles.SQUARE,  Colors.PINK]

# Distractor objects (placed on grid but not part of the rule chain)
brown_pyramid  = TILES_REGISTRY[Tiles.PYRAMID, Colors.BROWN]
orange_square  = TILES_REGISTRY[Tiles.SQUARE,  Colors.ORANGE]
grey_hex       = TILES_REGISTRY[Tiles.HEX,     Colors.GREY]
red_pyramid    = TILES_REGISTRY[Tiles.PYRAMID, Colors.RED]

# ──────────────────────────────────────────────────────────────
# Define Easy Ruleset
# ──────────────────────────────────────────────────────────────
easy_goal = AgentHoldGoal(tile=yellow_ball)

easy_ruleset = RuleSet(
    goal=easy_goal.encode(),
    rules=jnp.zeros((0, 7), dtype=jnp.int32),  # No rules
    init_tiles=jnp.array([
        yellow_ball,       # target
        orange_square,     # distractor
    ]),
)

easy_prompt = """You are evaluating the difficulty of grid-world environments for a navigation agent.

## Task Description
The agent (shown as a red triangle) must complete the following task:
1. **Pick up the yellow ball** (circle shape, yellow color) - completing the goal.

## Environment Details
- The grid is a 13×13 four-room layout with walls (grey blocks) dividing it into 4 quadrants.
- Rooms are connected by colored doors (closed but unlockable by toggling).
- The agent can: move forward, turn left/right, pick up objects, put down objects, and toggle doors.
- The agent can only hold ONE object at a time.
- **Distractor objects** (orange square) are present but irrelevant to the task.
"""

# ──────────────────────────────────────────────────────────────
# Define Medium Ruleset
# ──────────────────────────────────────────────────────────────
med_goal = AgentHoldGoal(tile=purple_hex)

med_rule1 = TileNearRule(tile_a=yellow_ball, tile_b=blue_key, prod_tile=green_star)
med_rule2 = AgentHoldRule(tile=green_star, prod_tile=purple_hex)

med_ruleset = RuleSet(
    goal=med_goal.encode(),
    rules=jnp.vstack([med_rule1.encode(), med_rule2.encode()]),
    init_tiles=jnp.array([
        yellow_ball,       
        blue_key,          
        brown_pyramid,       
        orange_square,     
    ]),
)

med_prompt = """You are evaluating the difficulty of grid-world environments for a navigation agent.

## Task Description
The agent (shown as a red triangle) must complete the following multi-step task:
1. **Pick up the yellow ball** (circle shape, yellow color)
2. **Navigate to the blue key** (key shape, blue color)
3. **Place the yellow ball adjacent to the blue key** — transforms both into a **green star** (hexagram shape)
4. **Pick up the green star** — transforms into a **purple hexagon** in the agent's pocket, completing the goal.

## Environment Details
- The grid is a 13×13 four-room layout with walls (grey blocks) dividing it into 4 quadrants.
- Rooms are connected by colored doors (closed but unlockable by toggling).
- The agent can: move forward, turn left/right, pick up objects, put down objects, and toggle doors.
- The agent can only hold ONE object at a time.
- **Distractor objects** (brown pyramid, orange square) are present but irrelevant to the task.
"""

# ──────────────────────────────────────────────────────────────
# Define Difficult Ruleset
# ──────────────────────────────────────────────────────────────
diff_goal = AgentHoldGoal(tile=pink_square)

# 1. Agent gets near red pyramid -> spawns blue key
diff_rule1 = AgentNearRule(tile=red_pyramid, prod_tile=blue_key)
# 2. Place blue key near yellow ball -> spawns green star
diff_rule2 = TileNearRule(tile_a=blue_key, tile_b=yellow_ball, prod_tile=green_star)
# 3. Place green star near brown pyramid -> spawns purple hex
diff_rule3 = TileNearRule(tile_a=green_star, tile_b=brown_pyramid, prod_tile=purple_hex)
# 4. Agent picks up purple hex -> transforms to pink square
diff_rule4 = AgentHoldRule(tile=purple_hex, prod_tile=pink_square)

diff_ruleset = RuleSet(
    goal=diff_goal.encode(),
    rules=jnp.vstack([
        diff_rule1.encode(), 
        diff_rule2.encode(), 
        diff_rule3.encode(), 
        diff_rule4.encode()
    ]),
    init_tiles=jnp.array([
        red_pyramid,       
        yellow_ball,
        brown_pyramid,
        orange_square,     
        grey_hex,
    ]),
)

diff_prompt = """You are evaluating the difficulty of grid-world environments for a navigation agent.

## Task Description
The agent (shown as a red triangle) must complete the following highly complex multi-step task:
1. **Navigate adjacent to the red pyramid** — this rule drops a **blue key** on the floor.
2. **Pick up the blue key** and navigate to the **yellow ball**.
3. **Place the blue key adjacent to the yellow ball** — transforms both into a **green star**.
4. **Pick up the green star** and navigate to the **brown pyramid**.
5. **Place the green star adjacent to the brown pyramid** — transforms both into a **purple hexagon**.
6. **Pick up the purple hexagon** — transforms into a **pink square** in the agent's pocket, completing the goal.

## Environment Details
- The grid is a 13×13 four-room layout with walls (grey blocks) dividing it into 4 quadrants.
- Rooms are connected by colored doors (closed but unlockable by toggling).
- The agent can: move forward, turn left/right, pick up objects, put down objects, and toggle doors.
- The agent can only hold ONE object at a time.
- **Distractor objects** (orange square, grey hexagon) are present but irrelevant to the task.
"""

# ──────────────────────────────────────────────────────────────
# Group them up
# ──────────────────────────────────────────────────────────────
RULESETS = {
    "easy": (easy_ruleset, easy_prompt),
    "medium": (med_ruleset, med_prompt),
    "difficult": (diff_ruleset, diff_prompt),
}

common_evaluation_criteria = """
## What to Evaluate
For each level image shown, assess:
1. How far apart are the required objects? (spatial distance)
2. Are they in the same room or different rooms?
3. How many doors must the agent open to reach the required objects?
4. Where is the agent starting relative to the required objects?
5. Are distractor objects blocking efficient paths?

Rate each level's difficulty on a scale of 1 (trivial) to 5 (very hard).

## Visual Legend
| Shape | Tile Type |
|-------|-----------|
| Circle (●) | Ball |
| Square (■) | Square |
| Triangle (▲) | Pyramid |
| Hexagram (✡) | Star |
| Hexagon (⬢) | Hexagon |
| Key shape (🔑) | Key |
| Solid colored block with handle | Closed Door |
| Grey solid block | Wall |
| Dark floor | Empty floor |
| Red triangle (agent) | Agent (points in facing direction) |
"""

# ──────────────────────────────────────────────────────────────
# Main execution
# ──────────────────────────────────────────────────────────────
def main():
    print("Building tile caches...")
    tile_cache, agent_tile_cache = build_tile_caches(tile_size=TILE_SIZE)
    
    key = jax.random.PRNGKey(SEED)

    for diff_level, (ruleset, prompt_text) in RULESETS.items():
        print("=" * 60)
        print(f"GENERATING: {diff_level.upper()} RULESET")
        print("=" * 60)
        print_ruleset(ruleset)
        print()

        # Create output dir
        out_dir = os.path.join(BASE_OUTPUT_DIR, diff_level)
        os.makedirs(out_dir, exist_ok=True)

        # Create env with this ruleset
        env, env_params = xminigrid.make(ENV_ID, view_size=VIEW_SIZE)
        # Pad ruleset rules to max length of 4 so they can be jitted/processed if needed, though not strictly required here.
        # But wait, env.reset doesn't care about static shape of rules if we just run it directly.
        env_params = env_params.replace(ruleset=ruleset)

        print(f"Generating {NUM_LEVELS_PER_RULESET} level layouts for {diff_level}...")
        for i in range(NUM_LEVELS_PER_RULESET):
            key, reset_key = jax.random.split(key)
            timestep = env.reset(env_params, reset_key)

            img = jax_render(
                grid=timestep.state.grid,
                agent=timestep.state.agent,
                tile_cache=tile_cache,
                agent_tile_cache=agent_tile_cache,
                view_size=VIEW_SIZE,
                tile_size=TILE_SIZE,
                sight_highlight=True,
            )

            img_np = np.asarray(img)
            img_pil = Image.fromarray(img_np)
            path = os.path.join(out_dir, f"level_{i:02d}.png")
            img_pil.save(path)

            agent_pos = tuple(np.asarray(timestep.state.agent.position))
            agent_dir = int(np.asarray(timestep.state.agent.direction))
            dir_names = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
            print(f"  Level {i}: agent at {agent_pos} facing {dir_names[agent_dir]} → {path}")

        # Save prompt
        full_prompt = prompt_text + common_evaluation_criteria
        prompt_path = os.path.join(out_dir, "task_prompt.txt")
        with open(prompt_path, "w") as f:
            f.write(full_prompt)
        print(f"  📝 Prompt saved to: {prompt_path}\n")


if __name__ == "__main__":
    main()
