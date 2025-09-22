from threading import Lock

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore

from app.common.http_client import get_proxies
from app.lib.bedrock_embedding_client import embedding_bedrock

class VectorStoreClient:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.vector_store = InMemoryVectorStore(embedding_bedrock())

        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )

        self.last_loaded_urls: list[str] | None = None
        self.last_docs = None
        self.last_doc_splits = None

    def load_documents(self, urls, metadata_key="source", metadata_value="defra-ai"):
        doc_ids = []
        docs = [WebBaseLoader(url, proxies=get_proxies()).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        for doc in docs_list:
            doc.metadata[metadata_key] = metadata_value

        self.last_loaded_urls = list(urls)
        self.last_docs = docs_list

        doc_splits = self.text_splitter.split_documents(docs_list)

        self.last_doc_splits = doc_splits

        self.vector_store.add_documents(doc_splits)
        print(f"Loaded {len(doc_splits)} document splits.")
        return doc_ids

    def get_last_doc_splits(self):
        return self.last_doc_splits or []

    def similarity_search(
        self, query, k=1, filter_key="source", filter_value="defra-ai"
    ):
        def _filter_function(doc):
            return doc.metadata.get(filter_key) == filter_value

        return self.vector_store.similarity_search(
            query=query, k=k, filter=_filter_function
        )

    def delete_documents(self, ids):
        self.vector_store.delete(ids)
        print(f"Deleted documents with ids {ids}.")

    def clear_vector_store(self):
        self.vector_store = InMemoryVectorStore(embedding_bedrock())
        print("Vector store cleared.")

    def has_content(self):
        try:
            test_results = self.vector_store.similarity_search("test", k=1)
            return len(test_results) > 0
        except Exception:
            return False

    def get_document_count(self):
        try:
            all_results = self.vector_store.similarity_search("", k=10000)
            return len(all_results)
        except Exception:
            return 0

    def as_retriever(self):
        return self.vector_store.as_retriever()
