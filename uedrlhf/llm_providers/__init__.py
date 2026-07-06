"""LLM provider registry.

Usage:
    from uedrlhf.llm_providers import get_llm_provider
    provider = get_llm_provider("gemini", config_dict)
"""

from uedrlhf.llm_providers.base import LLMProvider


_REGISTRY = {
    "gemini": "uedrlhf.llm_providers.gemini_provider.GeminiProvider",
    "openwebui": "uedrlhf.llm_providers.openwebui_provider.OpenWebUIProvider",
}


def get_llm_provider(name: str, config: dict) -> LLMProvider:
    """Instantiate an LLM provider by name.

    Providers are imported lazily so that unused SDKs (e.g. google-genai)
    are not loaded when using a different provider.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown LLM provider '{name}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )

    module_path, class_name = _REGISTRY[name].rsplit(".", 1)

    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    return cls(config)
