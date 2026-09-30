import numpy as np
import matplotlib.pyplot as plt

from uedrlhf.config import BASE_DIR, FEEDBACK_TYPE, LEARNABILITY_MODE, DB_PATH
from uedrlhf.db import get_connection, resolve_table_name


def load_preferences(cursor, table_name):
    """Load completed pairs as {pair_id: preference}."""
    cursor.execute(f"""
        SELECT pair_id, preference FROM "{table_name}"
        WHERE status = 'completed' AND preference IS NOT NULL
    """)
    return {row[0]: row[1] for row in cursor.fetchall()}


def compare(human_prefs: dict, llm_prefs: dict):
    """Compare human and LLM preferences on shared pairs."""

    common_ids = sorted(set(human_prefs.keys()) & set(llm_prefs.keys()))
    if not common_ids:
        print("No common completed pairs found.")
        return

    human = np.array([human_prefs[i] for i in common_ids])
    llm = np.array([llm_prefs[i] for i in common_ids])

    # Exclude ties from agreement calculation
    no_tie_mask = (human != 0.5) & (llm != 0.5)
    human_no_tie = human[no_tie_mask]
    llm_no_tie = llm[no_tie_mask]

    agree = (human_no_tie == llm_no_tie).sum()
    total_no_tie = no_tie_mask.sum()
    agreement_rate = agree / total_no_tie if total_no_tie > 0 else 0.0

    # Tie breakdown
    both_tie = ((human == 0.5) & (llm == 0.5)).sum()
    human_tie_only = ((human == 0.5) & (llm != 0.5)).sum()
    llm_tie_only = ((human != 0.5) & (llm == 0.5)).sum()

    print(f"Common pairs:         {len(common_ids)}")
    print(f"Pairs without ties:   {total_no_tie}")
    print(f"Agreement rate:       {agreement_rate:.3f}  ({100*agreement_rate:.1f}%)")
    print(f"Both tie:             {both_tie}")
    print(f"Human tie only:       {human_tie_only}")
    print(f"LLM tie only:         {llm_tie_only}")

    # Confusion matrix values
    both_0 = ((human == 0.0) & (llm == 0.0)).sum()
    both_1 = ((human == 1.0) & (llm == 1.0)).sum()
    h0_l1 = ((human == 0.0) & (llm == 1.0)).sum()
    h1_l0 = ((human == 1.0) & (llm == 0.0)).sum()

    print(f"\nConfusion matrix (no ties):")
    print(f"  Human=0, LLM=0 (agree env_0): {both_0}")
    print(f"  Human=1, LLM=1 (agree env_1): {both_1}")
    print(f"  Human=0, LLM=1 (disagree):    {h0_l1}")
    print(f"  Human=1, LLM=0 (disagree):    {h1_l0}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Confusion matrix heatmap
    cm = np.array([[both_0, h0_l1],
                   [h1_l0, both_1]])
    im = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["LLM=0", "LLM=1"])
    axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["Human=0", "Human=1"])
    axes[0].set_title(f"Confusion matrix\nAgreement: {100*agreement_rate:.1f}%")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    plt.colorbar(im, ax=axes[0])

    # Agreement bar chart
    categories = ["Agree\n(env_0)", "Agree\n(env_1)", "Disagree\nH=0,L=1", "Disagree\nH=1,L=0",
                  "Both tie", "Human\ntie only", "LLM\ntie only"]
    values = [both_0, both_1, h0_l1, h1_l0, both_tie, human_tie_only, llm_tie_only]
    colors = ["steelblue", "steelblue", "coral", "coral", "gray", "sienna", "sienna"]

    axes[1].bar(categories, values, color=colors, edgecolor="white")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Agreement breakdown")

    plt.suptitle(f"{FEEDBACK_TYPE} — Human vs LLM alignment", fontsize=11)
    plt.tight_layout()

    out_dir = BASE_DIR / "outputs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    if FEEDBACK_TYPE == "learnability_feedback":
        out_path = out_dir / f"alignment_{FEEDBACK_TYPE}_{LEARNABILITY_MODE}.png"
    else:
        out_path = out_dir / f"alignment_{FEEDBACK_TYPE}.png"

    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"\nPlot saved to {out_path}")


def main():
    conn = get_connection()
    cursor = conn.cursor()

    # Human feedback table
    human_table = resolve_table_name("human")
    # LLM feedback table
    llm_table = resolve_table_name("gemini-2.5-flash")

    print(f"Human table : {human_table}")
    print(f"LLM table   : {llm_table}")
    print(f"Database    : {DB_PATH}\n")

    human_prefs = load_preferences(cursor, human_table)
    llm_prefs = load_preferences(cursor, llm_table)

    print(f"Human completed pairs : {len(human_prefs)}")
    print(f"LLM completed pairs   : {len(llm_prefs)}\n")

    conn.close()

    compare(human_prefs, llm_prefs)


if __name__ == "__main__":
    main()
