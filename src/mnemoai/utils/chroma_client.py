"""Embedded Chroma storage with embeddings supplied exclusively by MnemoAI."""

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


def persistent_client(path):
    # PersistentClient otherwise accepts CHROMA_API_IMPL from the environment,
    # which can redirect local memory to the vulnerable HTTP server interface.
    return chromadb.PersistentClient(
        path=str(path),
        settings=Settings(chroma_api_impl="chromadb.api.rust.RustBindingsAPI"),
    )
