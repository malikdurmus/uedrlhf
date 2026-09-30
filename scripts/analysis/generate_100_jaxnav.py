"""Generate 100 JaxNav environments, render them as images, and compute difficulty metrics.

Usage:
    python scripts/analysis/generate_100_jaxnav.py [--num_envs 100] [--output_dir outputs/jaxnav_100_levels]
"""

import argparse
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav
from uedrlhf.config import BASE_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Generate JaxNav environment images and difficulty metrics.")
    parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to generate.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(BASE_DIR / "outputs" / "jaxnav_100_levels"),
        help="Directory to save images and reports.",
    )
    parser.add_argument("--seed", type=int, default=None, help="PRNG seed.")
    return parser.parse_args()


def generate_html_gallery(output_dir: Path, env_records: list[dict]):
    """Creates an interactive HTML gallery to browse and sort levels easily."""
    records_json = json.dumps(env_records)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JaxNav 100 Environments Gallery</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 24px;
        }}
        h1 {{
            margin-top: 0;
            font-size: 28px;
            color: #38bdf8;
        }}
        .controls {{
            display: flex;
            gap: 12px;
            align-items: center;
            margin-bottom: 24px;
            flex-wrap: wrap;
        }}
        button {{
            background-color: #1e293b;
            color: #f8fafc;
            border: 1px solid #475569;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        button:hover, button.active {{
            background-color: #0284c7;
            border-color: #38bdf8;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
            gap: 18px;
        }}
        .card {{
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
            transition: transform 0.2s, border-color 0.2s;
        }}
        .card:hover {{
            transform: translateY(-3px);
            border-color: #38bdf8;
        }}
        .card.easy {{
            border: 2px solid #22c55e;
        }}
        .card.hard {{
            border: 2px solid #ef4444;
        }}
        .card img {{
            width: 100%;
            aspect-ratio: 1 / 1;
            object-fit: cover;
            background-color: #000;
            border-bottom: 1px solid #334155;
        }}
        .info {{
            padding: 12px;
            font-size: 13px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
            text-transform: uppercase;
            margin-bottom: 4px;
        }}
        .badge-easy {{ background-color: #15803d; color: #fff; }}
        .badge-hard {{ background-color: #b91c1c; color: #fff; }}
        .metric {{
            display: flex;
            justify-content: space-between;
            color: #94a3b8;
        }}
        .metric span.val {{
            font-weight: bold;
            color: #f1f5f9;
        }}
    </style>
</head>
<body>
    <h1>JaxNav Generated Environments Gallery</h1>
    <div class="controls">
        <span style="color: #94a3b8; font-size: 14px;">Sort by:</span>
        <button id="btn-easy" class="active" onclick="sortCards('easiest')">Easiest First (Shortest Path)</button>
        <button id="btn-hard" onclick="sortCards('hardest')">Hardest First (Longest Path & Detour)</button>
        <button id="btn-detour" onclick="sortCards('detour')">Highest Detour Ratio</button>
        <button id="btn-id" onclick="sortCards('id')">Environment Index (0–99)</button>
    </div>
    <div id="gallery" class="grid"></div>

    <script>
        const data = {records_json};
        
        // Find easiest and hardest
        const sortedByDiff = [...data].sort((a, b) => (b.dijkstra_dist * b.detour_ratio) - (a.dijkstra_dist * a.detour_ratio));
        const hardestId = sortedByDiff[0].id;
        const easiestId = sortedByDiff[sortedByDiff.length - 1].id;

        function render(items) {{
            const container = document.getElementById('gallery');
            container.innerHTML = items.map(item => {{
                let extraClass = "";
                let badge = "";
                if (item.id === easiestId) {{
                    extraClass = "easy";
                    badge = '<span class="badge badge-easy">★ Recommended Easiest</span>';
                }} else if (item.id === hardestId) {{
                    extraClass = "hard";
                    badge = '<span class="badge badge-hard">★ Recommended Hardest</span>';
                }}
                return `
                <div class="card ${{extraClass}}">
                    <a href="${{item.image_file}}" target="_blank">
                        <img src="${{item.image_file}}" alt="Environment ${{item.id}}" loading="lazy">
                    </a>
                    <div class="info">
                        ${{badge}}
                        <div style="font-weight: bold; font-size: 14px; margin-bottom: 4px;">Env #${{String(item.id).padStart(3, '0')}}</div>
                        <div class="metric"><span>Shortest Path:</span><span class="val">${{item.dijkstra_dist.toFixed(1)}}</span></div>
                        <div class="metric"><span>Straight Line:</span><span class="val">${{item.euclidean_dist.toFixed(1)}}</span></div>
                        <div class="metric"><span>Detour Ratio:</span><span class="val">${{item.detour_ratio.toFixed(2)}}x</span></div>
                        <div class="metric"><span>Obstacles:</span><span class="val">${{item.obstacle_pct.toFixed(0)}}%</span></div>
                    </div>
                </div>
                `;
            }}).join('');
        }}

        function sortCards(criteria) {{
            document.querySelectorAll('.controls button').forEach(b => b.classList.remove('active'));
            let sorted = [...data];
            if (criteria === 'easiest') {{
                document.getElementById('btn-easy').classList.add('active');
                sorted.sort((a, b) => (a.dijkstra_dist * a.detour_ratio) - (b.dijkstra_dist * b.detour_ratio));
            }} else if (criteria === 'hardest') {{
                document.getElementById('btn-hard').classList.add('active');
                sorted.sort((a, b) => (b.dijkstra_dist * b.detour_ratio) - (a.dijkstra_dist * a.detour_ratio));
            }} else if (criteria === 'detour') {{
                document.getElementById('btn-detour').classList.add('active');
                sorted.sort((a, b) => b.detour_ratio - a.detour_ratio);
            }} else if (criteria === 'id') {{
                document.getElementById('btn-id').classList.add('active');
                sorted.sort((a, b) => a.id - b.id);
            }}
            render(sorted);
        }}

        sortCards('easiest');
    </script>
</body>
</html>
"""
    with open(output_dir / "index.html", "w") as f:
        f.write(html_content)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else int(time.time())
    print(f"Generating {args.num_envs} JaxNav environments (seed={seed})...")

    map_params = {
        "map_size": (11, 11),
        "min_fill": 0.0,
        "max_fill": 0.35,
        "valid_path_check": True,
    }

    env = JaxNav(num_agents=1, map_params=map_params)

    rng = jax.random.PRNGKey(seed)
    keys = jax.random.split(rng, args.num_envs)

    print("Sampling environments (vmap reset)...")
    _, env_states = jax.vmap(env.reset)(keys)

    records = []
    print(f"Rendering {args.num_envs} environments and computing difficulty metrics...")

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    for i in tqdm(range(args.num_envs), desc="Rendering"):
        state = jax.tree_util.tree_map(lambda x: x[i], env_states)

        # Compute shortest path via Dijkstra
        valid, d_dist = env._map_obj.dikstra_path(state.map_data, state.pos[0], state.goal[0])
        dijkstra_dist = float(d_dist)
        euclidean_dist = float(jnp.linalg.norm(state.pos[0] - state.goal[0]))
        detour_ratio = dijkstra_dist / max(euclidean_dist, 1e-4)

        # Obstacle fill percentage of internal 9x9 walkable zone
        inner_map = np.array(state.map_data[1:-1, 1:-1])
        obstacle_pct = float(np.mean(inner_map) * 100.0)

        img_filename = f"env_{i:03d}.png"
        img_path = output_dir / img_filename

        ax.clear()
        env.init_render(ax, state, lidar=False, agent=True, goal=True)
        plt.savefig(img_path, bbox_inches="tight")

        records.append({
            "id": i,
            "image_file": img_filename,
            "dijkstra_dist": dijkstra_dist,
            "euclidean_dist": euclidean_dist,
            "detour_ratio": detour_ratio,
            "obstacle_pct": obstacle_pct,
            "composite_score": round(dijkstra_dist * detour_ratio, 2),
        })

    plt.close(fig)

    # Save metadata JSON
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(records, f, indent=2)

    # Generate HTML gallery
    generate_html_gallery(output_dir, records)

    # Sort for summary
    sorted_by_easy = sorted(records, key=lambda r: r["composite_score"])
    easiest = sorted_by_easy[0]
    hardest = sorted_by_easy[-1]

    # Save Markdown summary
    summary_md = f"""# JaxNav Environments Summary ({args.num_envs} levels)

All {args.num_envs} environments have been generated and rendered to `{output_dir}`.

## Recommended Picks:
- **Easiest Level:** `env_{easiest['id']:03d}.png`
  - Shortest Path: {easiest['dijkstra_dist']:.1f}
  - Straight Line: {easiest['euclidean_dist']:.1f}
  - Detour Ratio: {easiest['detour_ratio']:.2f}x
  - Obstacles: {easiest['obstacle_pct']:.0f}%

- **Hardest Level:** `env_{hardest['id']:03d}.png`
  - Shortest Path: {hardest['dijkstra_dist']:.1f}
  - Straight Line: {hardest['euclidean_dist']:.1f}
  - Detour Ratio: {hardest['detour_ratio']:.2f}x
  - Obstacles: {hardest['obstacle_pct']:.0f}%

## Interactive Gallery:
Open `index.html` in your browser to view all {args.num_envs} environments with interactive sorting.
"""
    with open(output_dir / "summary.md", "w") as f:
        f.write(summary_md)

    print("\n" + "=" * 60)
    print(f"Successfully generated {args.num_envs} images in: {output_dir}/")
    print(f"Open gallery: {output_dir / 'index.html'}")
    print("=" * 60)
    print(f"\nTop 3 Easiest Environments:")
    for r in sorted_by_easy[:3]:
        print(f"  - env_{r['id']:03d}.png: Path={r['dijkstra_dist']:.1f}, Straight={r['euclidean_dist']:.1f}, Detour={r['detour_ratio']:.2f}x, Obstacles={r['obstacle_pct']:.0f}%")

    print(f"\nTop 3 Hardest Environments:")
    for r in sorted_by_easy[-3:][::-1]:
        print(f"  - env_{r['id']:03d}.png: Path={r['dijkstra_dist']:.1f}, Straight={r['euclidean_dist']:.1f}, Detour={r['detour_ratio']:.2f}x, Obstacles={r['obstacle_pct']:.0f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
