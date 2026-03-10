import dataclasses
import logging

import httpx

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class KnowledgeDoc:
    content: str
    file_name: str
    s3_key: str
    score: float


@dataclasses.dataclass(frozen=True)
class Source:
    name: str
    location: str
    snippet: str
    score: float


class KnowledgeRetriever:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    RAG_ERROR_MESSAGE = (
        "RAG lookup failed. Knowledge base sources could not be retrieved."
    )

    def search(
        self, group_ids: list[str], user_id: str, query: str, max_results: int = 5
    ) -> tuple[list[KnowledgeDoc], str | None]:
        """Returns (docs, error_message). error_message is non-None when RAG lookup failed."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.post(
                    f"{self.base_url}/rag/search",
                    json={
                        "knowledge_group_ids": group_ids,
                        "query": query,
                        "max_results": max_results,
                    },
                    headers={"user-id": user_id},
                )
                response.raise_for_status()
                raw_docs = response.json()
                return [
                    KnowledgeDoc(
                        content=d["content"],
                        file_name=d.get("file_name", ""),
                        s3_key=d.get("s3_key", ""),
                        score=d["similarity_score"],
                    )
                    for d in raw_docs
                ], None
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
