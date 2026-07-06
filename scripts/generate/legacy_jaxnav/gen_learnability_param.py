"""Generate environment pairs for learnability feedback using configuration parameter binning.

TODO: Implement 4-bin pairing based on environment configuration parameters
      (fill ratio, start-goal distance, etc.)

Usage:
    python scripts/generate/gen_learnability_param.py
"""

from uedrlhf.config import DATA_DIR, IMAGES_DIR, ARRAYS_DIR
from uedrlhf.db import get_connection, create_pair_table, resolve_table_name


def main():
    raise NotImplementedError(
        "Parameter-based learnability binning is not yet implemented. "
        "Use gen_learnability_dijkstra.py for Dijkstra-based binning."
    )


if __name__ == "__main__":
    main()
