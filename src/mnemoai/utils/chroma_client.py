"""Embedded Chroma storage with embeddings supplied exclusively by MnemoAI."""

from copy import deepcopy
from functools import wraps

import chromadb
from chromadb.config import Settings


class _PrecomputedEmbeddings:
    def __call__(self, input):
        raise ValueError("MnemoAI must supply precomputed embeddings to ChromaDB")

    def embed_query(self, input):
        return self(input)

    @staticmethod
    def name():
        # Chroma uses this sentinel to accept existing collections without
        # interpreting their saved embedding-function configuration.
        return "default"

    @staticmethod
    def is_legacy():
        # Persist no executable embedding-function configuration.
        return True


PRECOMPUTED_EMBEDDINGS = _PrecomputedEmbeddings()


def _precomputed_collection(collection):
    """Remove executable configuration from a private, in-memory model snapshot.

    Chroma deserializes schema embedding functions even on an add with supplied
    vectors, while checking sparse metadata. Never pass those definitions to its
    deserializers. Keep the database and all other schema/index settings intact.
    """
    model = deepcopy(collection._model)
    model.configuration_json.pop("embedding_function", None)
    schema = getattr(model, "serialized_schema", None)
    if schema is not None:
        # Iterate the value types, not arbitrary JSON keys: a user's metadata
        # field named "embedding_function" is a schema key and must be retained.
        for value_types in [schema.get("defaults", {}), *schema.get("keys", {}).values()]:
            for value_type, index_type in (
                ("float_list", "vector_index"),
                ("sparse_vector", "sparse_vector_index"),
            ):
                index = value_types.get(value_type, {}).get(index_type, {})
                index.get("config", {}).pop("embedding_function", None)
    collection._model = model
    collection._embedding_function = PRECOMPUTED_EMBEDDINGS
    return collection


class _PrecomputedClient:
    """Keep every collection acquisition behind the same deserialization guard."""

    def __init__(self, client):
        self._client = client

    def __getattr__(self, name):
        # Resolve first so optional APIs (e.g. get_collection_by_id) remain
        # absent on older supported Chroma versions.
        method = getattr(self._client, name)
        if name in {
            "create_collection", "get_collection", "get_or_create_collection",
            "get_collection_by_id",
        }:
            @wraps(method)
            def collection(*args, **kwargs):
                kwargs["embedding_function"] = PRECOMPUTED_EMBEDDINGS
                return _precomputed_collection(method(*args, **kwargs))
            return collection
        if name == "list_collections":
            @wraps(method)
            def collections(*args, **kwargs):
                return [_precomputed_collection(item) for item in method(*args, **kwargs)]
            return collections
        return method


def persistent_client(path):
    # PersistentClient otherwise accepts CHROMA_API_IMPL from the environment,
    # which can redirect local memory to the vulnerable HTTP server interface.
    return _PrecomputedClient(chromadb.PersistentClient(
        path=str(path),
        settings=Settings(chroma_api_impl="chromadb.api.rust.RustBindingsAPI"),
    ))
