"""Prompt registry for XLand-MiniGrid environments.

Maps env_id → feedback_type → prompt string.  When a new environment
is added to the pipeline, define its prompts here.

Usage (from xland_backend.py):
    from uedrlhf.environments.xland_prompts import get_xland_prompt
    prompt = get_xland_prompt(env_id, feedback_type)
"""

from uedrlhf.config import JSON_INSTRUCTION


# Used as a fallback when the env_id has no dedicated entry.

_DEFAULT_PROMPTS: dict[str, str] = {
    "difficulty_feedback": (
        "You are evaluating two grid-world environments for an RL agent. "
        "Each image shows a top-down view of a MiniGrid environment with coloured tiles, "
        "objects (keys, doors, balls), walls, and an agent (red triangle) that must reach a goal.\n\n"
        "Your task: Which environment is HARDER for the agent to solve?\n\n"
        "Consider factors like:\n"
        "- Number and complexity of objects and rules (keys needed to open doors, etc.)\n"
        "- Grid size and maze complexity\n"
        "- Distance and obstacles between the agent and the goal\n"
        "- Whether multiple steps of reasoning are needed (pick up key → open door → reach goal)\n\n"
        "Choose:\n"
        "  A  — if the First Image (Image A) is harder\n"
        "  B  — if the Second Image (Image B) is harder\n"
        "  TIE — if they are equally difficult\n\n"
        + JSON_INSTRUCTION
    ),
}


# ── Per-environment prompt overrides ────────────────────────────────────
# Only the feedback types that differ from the default need to be listed.
# Missing keys fall back to _DEFAULT_PROMPTS.

ENV_PROMPTS: dict[str, dict[str, str]] = {

    # ── MiniGrid-BlockedUnlockPickUp ────────────────────────────────────
    "MiniGrid-BlockedUnlockPickUp": {
        "difficulty_feedback": (
            "You are evaluating two MiniGrid-BlockedUnlockPickUp environments for an RL agent.\n\n"
            "In this task, the agent (red triangle) must figure out how to navigate between several objects to reach the final goal:\n"
            "  - The agent must pick up a target object (a box) located in a second room.\n"
            "  - The rooms are separated by a locked door, which requires a key to open.\n"
            "  - A ball is blocking the path or the door.\n"
            "  - The agent can only carry ONE item at a time, meaning it must juggle moving the ball out of the way and transporting the key to the door.\n\n"
            "Each image shows a top-down grid view with two rooms separated by a wall. "
            "You will see coloured balls, keys, doors, and boxes.\n\n"
            "Your task: Which environment is HARDER for a random-exploring RL agent to solve?\n\n"
            "Consider factors like:\n"
            "- Distance and Backtracking: How far apart are the agent, key, ball, and door? (Greater distance means more steps required to juggle items back and forth)\n"
            "- Maneuvering space: Is the ball wedged in a tight corridor where the agent has little room to drop it out of the way?\n"
            "- Overall path length and complexity of wall obstructions.\n\n"
            "Choose:\n"
            "  A  — if the First Image (Image A) is harder\n"
            "  B  — if the Second Image (Image B) is harder\n"
            "  TIE — if they are equally difficult\n\n"
            + JSON_INSTRUCTION
        ),
    },

    # ── XLand-MiniGrid-R4-9x9 ──────────────────────────────────────────
    # Uses the generic default prompt (multi-rule grid world).
    # Uncomment and customise below if a specialised prompt is needed.
    #
    # "XLand-MiniGrid-R4-9x9": {
    #     "difficulty_feedback": "...",
    # },
}


def get_xland_prompt(env_id: str, feedback_type: str) -> str:
    """Return the prompt for a given env_id and feedback_type.

    Lookup order:
        1. RULESET specific prompt (if config.RULESET is set)
        2. ENV_PROMPTS[env_id][feedback_type]   (env-specific override)
        3. _DEFAULT_PROMPTS[feedback_type]       (generic fallback)

    Raises KeyError if the feedback_type is not found in either dict.
    """
    from uedrlhf.config import RULESET
    if RULESET:
        from uedrlhf.environments.xland_rulesets import get_ruleset_prompt
        return get_ruleset_prompt(RULESET)
        
    env_entry = ENV_PROMPTS.get(env_id, {})
    prompt = env_entry.get(feedback_type)
    if prompt is not None:
        return prompt
    return _DEFAULT_PROMPTS[feedback_type]