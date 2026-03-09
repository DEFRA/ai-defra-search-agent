import logging

import httpx

from app.common.knowledge import KnowledgeRetriever


class TestKnowledgeRetriever:
    def test_search_returns_filtered_documents_from_knowledge_service(self, mocker):
        base_url = "http://test"
        retriever = KnowledgeRetriever(base_url=base_url, similarity_threshold=0.5)

        mock_response = mocker.Mock()
        mock_response.json.return_value = [
            {"similarity_score": 0.9, "content": "foo", "document_id": "doc1"},
            {"similarity_score": 0.4, "content": "bar", "document_id": "doc2"},
        ]
        mock_response.raise_for_status.return_value = None

        mock_client = mocker.patch("httpx.Client")
        mock_client_instance = mock_client.return_value
        mock_client_instance.__enter__.return_value = mock_client_instance
        mock_client_instance.post.return_value = mock_response

        docs, error = retriever.search(
            group_ids=["group1"], user_id="user-1", query="query"
        )

        mock_client_instance.post.assert_called_once_with(
            f"{base_url}/rag/search",
            json={
                "knowledge_group_ids": ["group1"],
                "query": "query",
                "max_results": 5,
            },
            headers={"user-id": "user-1"},
        )
        assert error is None
        assert len(docs) == 1
        assert docs[0]["similarity_score"] == 0.9
        assert docs[0]["content"] == "foo"

    def test_search_passes_max_results(self, mocker):
        retriever = KnowledgeRetriever(base_url="http://test")
        mock_response = mocker.Mock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None

        mock_client = mocker.patch("httpx.Client")
        mock_client_instance = mock_client.return_value
        mock_client_instance.__enter__.return_value = mock_client_instance
        mock_client_instance.post.return_value = mock_response

        retriever.search(
            group_ids=["group1"], user_id="user-1", query="query", max_results=10
        )

        mock_client_instance.post.assert_called_once_with(
            f"{retriever.base_url}/rag/search",
            json={
                "knowledge_group_ids": ["group1"],
                "query": "query",
                "max_results": 10,
            },
            headers={"user-id": "user-1"},
        )

    def test_search_returns_empty_list_on_http_error(self, caplog, mocker):
        retriever = KnowledgeRetriever(base_url="http://test")

        mock_response = mocker.Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error", request=mocker.Mock(), response=mocker.Mock()
        )

        mock_client = mocker.patch("httpx.Client")
        mock_client_instance = mock_client.return_value
        mock_client_instance.__enter__.return_value = mock_client_instance
        mock_client_instance.post.return_value = mock_response

        docs, error = retriever.search(
            group_ids=["group1"], user_id="user-1", query="query"
        )

        assert docs == []
        assert error == KnowledgeRetriever.RAG_ERROR_MESSAGE
        assert "RAG Lookup failed" in caplog.text

    def test_search_logs_json_body_on_http_error_when_available(self, caplog, mocker):
        retriever = KnowledgeRetriever(base_url="http://test")
        mock_http_response = mocker.Mock()
        mock_http_response.status_code = 500
        mock_http_response.reason_phrase = "Internal Server Error"
        mock_http_response.json.return_value = {"error": "Something went wrong"}
        mock_http_response.text = "fallback"

        mock_response = mocker.Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error", request=mocker.Mock(), response=mock_http_response
        )

        mock_client = mocker.patch("httpx.Client")
        mock_client_instance = mock_client.return_value
        mock_client_instance.__enter__.return_value = mock_client_instance
        mock_client_instance.post.return_value = mock_response

        docs, error = retriever.search(
            group_ids=["group1"], user_id="user-1", query="query"
        )

        assert docs == []
        assert error == KnowledgeRetriever.RAG_ERROR_MESSAGE
        assert "Something went wrong" in caplog.text

    def test_search_falls_back_to_text_on_http_error_when_json_unavailable(
        self, caplog, mocker
    ):
        retriever = KnowledgeRetriever(base_url="http://test")
        mock_http_response = mocker.Mock()
        mock_http_response.status_code = 500
        mock_http_response.reason_phrase = "Internal Server Error"
        mock_http_response.json.side_effect = ValueError("Invalid JSON")
        mock_http_response.text = "HTML error page"

        mock_response = mocker.Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error", request=mocker.Mock(), response=mock_http_response
        )

        mock_client = mocker.patch("httpx.Client")
        mock_client_instance = mock_client.return_value
        mock_client_instance.__enter__.return_value = mock_client_instance
        mock_client_instance.post.return_value = mock_response

        docs, error = retriever.search(
            group_ids=["group1"], user_id="user-1", query="query"
        )

        assert docs == []
        assert error == KnowledgeRetriever.RAG_ERROR_MESSAGE
        assert "HTML error page" in caplog.text

    def test_search_returns_empty_list_on_connection_error(self, caplog, mocker):
        retriever = KnowledgeRetriever(base_url="http://test")

        mock_client = mocker.patch("httpx.Client")
        mock_client_instance = mock_client.return_value
        mock_client_instance.__enter__.return_value = mock_client_instance
        mock_client_instance.post.side_effect = httpx.ConnectError("Connection failed")

        docs, error = retriever.search(
            group_ids=["group1"], user_id="user-1", query="query"
        )

        assert docs == []
        assert error == KnowledgeRetriever.RAG_ERROR_MESSAGE
        assert "RAG Lookup failed" in caplog.text

    def test_filter_relevant_docs_filters_below_threshold(self):
        retriever = KnowledgeRetriever(base_url="http://test", similarity_threshold=0.6)
        docs = [
            {"id": 1, "similarity_score": 0.7},
            {"id": 2, "similarity_score": 0.6},
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
            {"id": 2, "similarity_score": 0.7},
        ]

        with caplog.at_level(logging.INFO):
            result = retriever._filter_relevant_docs(docs)

        assert len(result) == 1
        assert "Filtered 1 docs to 1 docs" in caplog.text

    def test_filter_relevant_docs_empty_input(self):
        retriever = KnowledgeRetriever(base_url="http://test")
        assert retriever._filter_relevant_docs([]) == []
