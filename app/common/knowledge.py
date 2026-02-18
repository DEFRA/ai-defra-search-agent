import logging
import re

import httpx

logger = logging.getLogger(__name__)


def _to_snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _convert_keys(data):
    if isinstance(data, dict):
        return {_to_snake_case(k): _convert_keys(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_convert_keys(i) for i in data]
    return data


class KnowledgeRetriever:
    def __init__(self, base_url: str, similarity_threshold: float = 0.5):
        self.base_url = base_url.rstrip("/")
        self.similarity_threshold = similarity_threshold

    RAG_ERROR_MESSAGE = (
        "RAG lookup failed. Knowledge base sources could not be retrieved."
    )

    def search(
        self, group_id: str, query: str, max_results: int = 5
    ) -> tuple[list[dict], str | None]:
        """Returns (docs, error_message). error_message is non-None when RAG lookup failed."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.post(
                    f"{self.base_url}/snapshots/query",
                    json={
                        "groupId": group_id,
                        "query": query,
                        "maxResults": max_results,
                    },
                )
                response.raise_for_status()
                raw_docs = _convert_keys(response.json())
                return self._filter_relevant_docs(raw_docs), None
        except httpx.HTTPStatusError as e:
            try:
                body = e.response.json()
            except Exception:
                body = e.response.text
            logger.error(
                "RAG Lookup failed: %s %s - %s",
                e.response.status_code,
                e.response.reason_phrase,
                body,
            )
            return [], self.RAG_ERROR_MESSAGE
        except Exception as e:
            logger.error("RAG Lookup failed: %s", e)
            return [], self.RAG_ERROR_MESSAGE

    def _filter_relevant_docs(self, docs: list[dict]) -> list[dict]:
        similar_docs = [
            d for d in docs if d["similarity_score"] >= self.similarity_threshold
        ]
        if num_filtered := len(docs) - len(similar_docs):
            logger.info(
                "Filtered %s docs to %s docs with similarity threshold %s",
                num_filtered,
                len(similar_docs),
                self.similarity_threshold,
            )
        return similar_docs
