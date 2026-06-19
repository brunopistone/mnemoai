class BaseModelController:
    """Shared base for the LLM and vision model controllers.

    Per-provider inference-parameter handling lives in
    ``models.provider_params`` (the single source of truth, consumed by both
    controllers via ``build_kwargs``). This base is intentionally minimal — it
    exists so the two controllers share a common type.
    """
