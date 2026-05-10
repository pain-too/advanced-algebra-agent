from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config


class VectorStoreService():
    def __init__(self,embedding):
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name = config.collection_name,
            persist_directory = config.persist_directory,
            embedding_function = DashScopeEmbeddings(model = config.embedding_name)
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs = {"k":config.similarity_threshold})