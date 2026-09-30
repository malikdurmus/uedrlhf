import jax
import numpy as np
import matplotlib.pyplot as plt
from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav

from uedrlhf.config import BASE_DIR


def plot_dijkstra_distribution(env, num_samples=3000, rng=jax.random.PRNGKey(0)):
    keys = jax.random.split(rng, num_samples)

    # vmap over sample_test_case
    map_data_batch, case_batch = jax.vmap(env._map_obj.sample_test_case)(keys)

    # Extract starts and goals
    starts = case_batch[:, 0, 0, :2]  # (num_samples, 2)
    goals = case_batch[:, 0, 1, :2]   # (num_samples, 2)

    # vmap over dijkstra
    valid_batch, d_batch = jax.vmap(env._map_obj.dikstra_path)(map_data_batch, starts, goals)

    # Convert to numpy for plotting
    valid_batch = np.array(valid_batch)
    d_batch = np.array(d_batch)

    distances = d_batch[valid_batch]
    unsolvable = (~valid_batch).sum()

    print(f"\nTotal sampled:  {num_samples}")
    print(f"Unsolvable:     {unsolvable} ({100*unsolvable/num_samples:.1f}%)")
    print(f"Solvable:       {len(distances)} ({100*len(distances)/num_samples:.1f}%)")
    print(f"\nDistance stats:")
    print(f"  min:    {distances.min():.1f}")
    print(f"  max:    {distances.max():.1f}")
    print(f"  mean:   {distances.mean():.1f}")
    print(f"  median: {np.median(distances):.1f}")

    percentiles = [0, 25, 50, 75, 100]
    boundaries = np.percentile(distances, percentiles)
    print(f"\nPercentile-based bin boundaries:")
    for i in range(len(boundaries) - 1):
        low, high = boundaries[i], boundaries[i + 1]
        count = np.sum((distances >= low) & (distances < high))
        print(f"  Bin {i} [{low:.1f}, {high:.1f}): {count} environments")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(distances, bins=50, color='steelblue', edgecolor='white')
    for b in boundaries[1:-1]:
        axes[0].axvline(b, color='red', linestyle='--', linewidth=1.5, label=f'{b:.1f}')
    axes[0].set_xlabel('Dijkstra path length (grid cells)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Dijkstra distances\n(red lines = percentile-based bin boundaries)')
    axes[0].legend(title='Boundaries')

    sorted_d = np.sort(distances)
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    axes[1].plot(sorted_d, cdf, color='steelblue')
    for b in boundaries[1:-1]:
        axes[1].axvline(b, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel('Dijkstra path length (grid cells)')
    axes[1].set_ylabel('Cumulative probability')
    axes[1].set_title('CDF of Dijkstra distances')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_dir = BASE_DIR / "outputs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / 'dijkstra_distribution.png', dpi=150)
    plt.show()

    return distances, boundaries


if __name__ == "__main__":
    env = JaxNav(num_agents=1)
    rng = jax.random.PRNGKey(2)
    distances, boundaries = plot_dijkstra_distribution(env, num_samples=5000, rng=rng)
