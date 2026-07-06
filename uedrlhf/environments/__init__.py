"""Environment backend registry.

Usage:
    from uedrlhf.environments import get_env_backend
    backend = get_env_backend("jaxnav", config_dict)
"""

from uedrlhf.environments.base import EnvBackend


_REGISTRY = {
    "jaxnav": "uedrlhf.environments.jaxnav_backend.JaxNavBackend",
    "xland_minigrid": "uedrlhf.environments.xland_backend.XLandMiniGridBackend",
}


def get_env_backend(name: str, config: dict) -> EnvBackend:
    """Instantiate an environment backend by name.

    Backends are imported lazily so that JaxNav is not pulled in when
    running XLand-MiniGrid experiments and vice versa.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown environment '{name}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )

    module_path, class_name = _REGISTRY[name].rsplit(".", 1)

    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    return cls(config)
