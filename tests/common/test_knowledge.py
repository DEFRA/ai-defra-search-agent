import logging

from app.common.knowledge import KnowledgeRetriever


class TestKnowledgeRetriever:
    def test_filter_relevant_docs_filters_below_threshold(self):
        retriever = KnowledgeRetriever(base_url="http://test", similarity_threshold=0.6)
        docs = [
            {"id": 1, "similarity_score": 0.7},
            {"id": 2, "similarity_score": 0.6},  # Inclusive check
            {"id": 3, "similarity_score": 0.59},
            {"id": 4, "similarity_score": 0.9},
        ]

        result = retriever._filter_relevant_docs(docs)

        assert len(result) == 3
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2
        assert result[2]["id"] == 4

    def test_filter_relevant_docs_logs_when_filtering(self, caplog):
        retriever = KnowledgeRetriever(base_url="http://test", similarity_threshold=0.8)
        docs = [
            {"id": 1, "similarity_score": 0.9},
            {"id": 2, "similarity_score": 0.7},  # Should be filtered
        ]

        with caplog.at_level(logging.INFO):
            result = retriever._filter_relevant_docs(docs)

        assert len(result) == 1
        assert "Filtered 1 docs to 1 docs" in caplog.text

    def test_filter_relevant_docs_empty_input(self):
        retriever = KnowledgeRetriever(base_url="http://test")
        assert retriever._filter_relevant_docs([]) == []
