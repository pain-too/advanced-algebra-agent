from utils.config_handler import chroma_conf
from utils.logger_handler import logger
from data.model.factory import embedding_model
from data.rag.KnowledgeBaseService import KnowledgeBaseService


class VectorStoreService:
    """
    向量库检索服务（轻量化，不重复造轮子）
    依赖 KnowledgeBaseService 实现加载、去重、分片
    """

    def __init__(self, embedding=None):
        logger.info("初始化 VectorStoreService...")

        self.embedding = embedding_model

        # ===================== 核心：复用 KBS =====================
        self.kbs = KnowledgeBaseService()
        self.vector_store = self.kbs.chroma  # 直接用KBS的向量库

        logger.info("VectorStoreService 初始化完成 ✅")

    def get_retriever(self, search_kwargs=None):
        if search_kwargs is None:
            search_kwargs = {"k": chroma_conf.get("k", 3)}

        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def get_vector_store(self):
        return self.vector_store