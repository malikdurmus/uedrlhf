"""Custom rulesets for XLand-MiniGrid."""

import jax.numpy as jnp
from xminigrid.types import RuleSet
from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
from xminigrid.core.goals import AgentHoldGoal
from xminigrid.core.rules import TileNearRule, AgentHoldRule, AgentNearRule

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
white_pyramid    = TILES_REGISTRY[Tiles.PYRAMID, Colors.WHITE]

# ──────────────────────────────────────────────────────────────
# Easy Ruleset
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
# Medium Ruleset
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
# Difficult Ruleset
# ──────────────────────────────────────────────────────────────
diff_goal = AgentHoldGoal(tile=pink_square)

# 1. Agent gets near white pyramid -> spawns blue key
diff_rule1 = AgentNearRule(tile=white_pyramid, prod_tile=blue_key)
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
        white_pyramid,       
        yellow_ball,
        brown_pyramid,
        orange_square,     
        grey_hex,
    ]),
)

diff_prompt = """You are evaluating the difficulty of grid-world environments for a navigation agent.

## Task Description
The agent (shown as a red triangle) must complete the following highly complex multi-step task:
1. **Navigate adjacent to the white pyramid** — this rule drops a **blue key** on the floor.
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

# Group

RULESETS = {
    "easy": (easy_ruleset, easy_prompt),
    "medium": (med_ruleset, med_prompt),
    "difficult": (diff_ruleset, diff_prompt),
}

common_evaluation_criteria = """
## What to Evaluate
Compare the two images provided. Which environment is HARDER for the agent to navigate and complete the task?
Consider:
1. How far apart are the required objects? (spatial distance)
2. Are they in the same room or different rooms?
3. How many doors must the agent open to reach the required objects?
4. Where is the agent starting relative to the required objects?
5. Are distractor objects blocking efficient paths?

Choose:
  A  — if Image A is harder
  B  — if Image B is harder
  TIE — if they are equally difficult

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


def get_ruleset(name: str) -> RuleSet:
    """Return the ruleset by name."""
    if name not in RULESETS:
        raise ValueError(f"Ruleset '{name}' not found. Available: {list(RULESETS.keys())}")
    return RULESETS[name][0]

def get_ruleset_prompt(name: str) -> str:
    """Return the prompt + evaluation criteria for the ruleset."""
    from uedrlhf.config import JSON_INSTRUCTION
    if name not in RULESETS:
        raise ValueError(f"Ruleset '{name}' not found. Available: {list(RULESETS.keys())}")
    return RULESETS[name][1] + "\n" + common_evaluation_criteria + "\n" + JSON_INSTRUCTION
