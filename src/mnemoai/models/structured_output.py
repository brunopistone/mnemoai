"""Optional native JSON constraints without changing the model-message interface."""

from copy import deepcopy
from typing import get_args, get_origin


def _accepts_object(annotation):
    if annotation is dict or get_origin(annotation) is dict:
        return True
    return any(_accepts_object(arg) for arg in get_args(annotation))


def with_json_schema(model, schema):
    """Use an SDK's declared object-format capability when it is available.

    The normal text prompt and strict response validator remain authoritative
    for every provider. Do not send speculative provider parameters or switch
    the model to an output-parser interface that loses usage/stop metadata.
    """
    inner = getattr(model, "bound", model)
    field = getattr(type(inner), "model_fields", {}).get("format")
    if field is not None and _accepts_object(getattr(field, "annotation", None)):
        return model.bind(format=deepcopy(schema))
    return model
