import logging

import httpx

from app.common.knowledge import KnowledgeRetriever, _convert_keys, _to_snake_case


class TestHelpers:
    def test_to_snake_case(self):
        assert _to_snake_case("camelCase") == "camel_case"
        assert _to_snake_case("PascalCase") == "pascal_case"
        assert _to_snake_case("simple") == "simple"
        assert _to_snake_case("PDFLoader") == "p_d_f_loader"

    def test_convert_keys(self):
        data = {"camelCase": 1, "nested": {"PascalCase": 2}, "list": [{"camelCase": 3}]}
        expected = {
            "camel_case": 1,
            "nested": {"pascal_case": 2},
            "list": [{"camel_case": 3}],
        }
        assert _convert_keys(data) == expected

    def test_convert_keys_non_dict_list(self):
        assert _convert_keys("string") == "string"
        assert _convert_keys(123) == 123


class TestKnowledgeRetriever:
    def test_search_success(self, mocker):
        base_url = "http://test"
        retriever = KnowledgeRetriever(base_url=base_url)

        mock_response = mocker.Mock()
        mock_response.json.return_value = [
            {"similarityScore": 0.9, "content": "foo"},
            {"similarityScore": 0.1, "content": "bar"},
        ]
        mock_response.raise_for_status.return_value = None

        mock_client = mocker.patch("httpx.Client")
        mock_client_instance = mock_client.return_value
        mock_client_instance.__enter__.return_value = mock_client_instance
        mock_client_instance.post.return_value = mock_response

        results = retriever.search("group1", "query")

        mock_client_instance.post.assert_called_once_with(
            f"{base_url}/snapshots/query",
            json={
                "groupId": "group1",
                "query": "query",
                "maxResults": 5,
            },
        )

        assert len(results) == 1
        assert results[0]["similarity_score"] == 0.9
        assert results[0]["content"] == "foo"

    def test_search_api_error(self, caplog, mocker):
        retriever = KnowledgeRetriever(base_url="http://test")

        mock_response = mocker.Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error", request=mocker.Mock(), response=mocker.Mock()
        )

        mock_client = mocker.patch("httpx.Client")
        mock_client_instance = mock_client.return_value
        mock_client_instance.__enter__.return_value = mock_client_instance
        mock_client_instance.post.return_value = mock_response

        results = retriever.search("group1", "query")

        assert results == []
        assert "RAG Lookup failed" in caplog.text

    def test_search_connection_error(self, caplog, mocker):
        retriever = KnowledgeRetriever(base_url="http://test")

        mock_client = mocker.patch("httpx.Client")
        mock_client_instance = mock_client.return_value
        mock_client_instance.__enter__.return_value = mock_client_instance
        mock_client_instance.post.side_effect = httpx.ConnectError("Connection failed")

        results = retriever.search("group1", "query")

        assert results == []
        assert "RAG Lookup failed" in caplog.text

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
