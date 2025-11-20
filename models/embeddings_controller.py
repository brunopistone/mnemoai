import boto3
import hashlib
import json
import numpy as np
import ollama
from typing import List, Any
from utils.config import config
from utils.logger import logger


class EmbeddingsController:
    """Controller for embedding model operations across different providers."""

    def __init__(self, embed_model_config: dict = None) -> None:
        """Initialize embeddings controller.

        Args:
            embed_model_config: Optional embedding model configuration
        """
        self.embed_model_config = embed_model_config or config.get("EMBED_MODEL_ID", {})
        self.embed_model_type = self.embed_model_config.get("TYPE", "ollama")
        self.embed_model_name = self.embed_model_config.get("NAME", "mxbai-embed-large")
        self.region = self.embed_model_config.get("REGION", "us-east-1")
        self.dim = None  # Will be set from first embedding

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts using configured provider (Ollama, Bedrock, or SageMaker).

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of embeddings with shape (n, dim)
        """
        logger.debug(
            f"Embedding {len(texts)} texts using {self.embed_model_type} model '{self.embed_model_name}'"
        )
        if not texts:
            logger.warning("Empty text list provided to embed()")
            return np.array([], dtype=np.float32).reshape(0, self.dim or 768)

        if self.embed_model_type == "ollama":
            return self._embed_ollama(texts)
        elif self.embed_model_type == "bedrock":
            return self._embed_bedrock(texts)
        elif self.embed_model_type == "sagemaker":
            return self._embed_sagemaker(texts)
        else:
            raise ValueError(
                f"Unsupported embedding model type: {self.embed_model_type}"
            )

    def _embed_ollama(self, texts: List[str]) -> np.ndarray:
        """Embed using Ollama.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of embeddings
        """
        try:
            resp = ollama.embed(model=self.embed_model_name, input=texts)
            emb = self._extract_embeddings_from_response(resp)
            return np.array(emb, dtype=np.float32)
        except Exception:
            logger.exception(
                "Ollama embed failed, falling back to deterministic embeddings"
            )
            return self._embed_fallback(texts)

    def _embed_bedrock(self, texts: List[str]) -> np.ndarray:
        """Embed using AWS Bedrock.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of embeddings
        """
        try:
            client = boto3.client("bedrock-runtime", region_name=self.region)

            embeddings = []
            for text in texts:
                response = client.invoke_model(
                    modelId=self.embed_model_name, body=json.dumps({"inputText": text})
                )
                result = json.loads(response["body"].read())
                embeddings.append(result.get("embedding", []))

            return np.array(embeddings, dtype=np.float32)
        except Exception:
            logger.exception(
                "Bedrock embed failed, falling back to deterministic embeddings"
            )
            return self._embed_fallback(texts)

    def _embed_sagemaker(self, texts: List[str]) -> np.ndarray:
        """Embed using AWS SageMaker.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of embeddings
        """
        try:
            client = boto3.client("sagemaker-runtime", region_name=self.region)

            response = client.invoke_endpoint(
                EndpointName=self.embed_model_name,
                ContentType="application/json",
                Body=json.dumps({"inputs": texts}),
            )
            result = json.loads(response["Body"].read())
            embeddings = result.get("embeddings", result)

            return np.array(embeddings, dtype=np.float32)
        except Exception:
            logger.exception(
                "SageMaker embed failed, falling back to deterministic embeddings"
            )
            return self._embed_fallback(texts)

    def _embed_fallback(self, texts: List[str]) -> np.ndarray:
        """Fallback to deterministic SHA256-based embeddings.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of deterministic embeddings
        """
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            v = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            if self.dim and len(v) < self.dim:
                v = np.resize(v, self.dim)
            elif self.dim:
                v = v[: self.dim]
            v = v / (np.linalg.norm(v) + 1e-12)
            out.append(v)
        return np.vstack(out)

    def _extract_embeddings_from_response(self, resp: Any) -> List[List[float]]:
        """Extract embeddings from Ollama response.

        Args:
            resp: Ollama API response

        Returns:
            List of embedding vectors
        """
        # Handle dict response
        if isinstance(resp, dict):
            if "embeddings" in resp:
                return resp["embeddings"]
            elif "embedding" in resp:
                emb = resp["embedding"]
                return [emb] if isinstance(emb[0], (int, float)) else emb

        # Handle object with attributes
        if hasattr(resp, "embeddings"):
            raw = resp.embeddings
            if isinstance(raw, list):
                return raw
        elif hasattr(resp, "embedding"):
            raw = resp.embedding
            if isinstance(raw, list):
                return [raw] if isinstance(raw[0], (int, float)) else raw

        raise ValueError(f"Failed to extract embeddings from response: {type(resp)}")
