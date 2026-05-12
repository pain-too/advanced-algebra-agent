# 系统模块
import os
from typing import List, Optional
# 第三方库
from langchain_core.documents import Document
# Agent 项目模块
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from model.factory import embedding_model
# RAG 核心模块
from rag.KnowledgeBaseService import KnowledgeBaseService
from rag.vector_store import VectorStoreService


class DSRagService:
    def __init__(self, data_path: Optional[str] = None) -> None:
        logger.info("=" * 60)
        logger.info("开始初始化 DSRagService...")


        # ===================== 配置读取 & 异常捕获 =====================
        try:
            self.k_default_k: int = chroma_conf.get("k", 3)
            self.k_data_path: str = chroma_conf.get("data_path", "./data")
            logger.info(f"配置读取成功 | k={self.k_default_k}, data_path={self.k_data_path}")

        except Exception as e:
            logger.error(f"读取 chroma_conf 配置失败！错误信息：{str(e)}")
            raise RuntimeError("DSRagService 初始化失败：配置加载异常") from e

        # ===================== 初始化核心服务 =====================
        try:
            self.kb_service: KnowledgeBaseService = KnowledgeBaseService()
            logger.info("KnowledgeBaseService 初始化成功")

            self.vector_service: VectorStoreService = VectorStoreService(embedding=embedding_model)
            self.vector_service.vector_store = self.kb_service.chroma
            logger.info("VectorStoreService 绑定向量库成功")

        except Exception as e:
            logger.error(f"服务初始化失败：{str(e)}")
            raise RuntimeError("DSRagService 服务初始化异常") from e

        # ===================== 自动加载PDF（保持不变） =====================
        if data_path is None:
            data_path = get_abs_path(self.k_data_path)

        logger.info(f"自动加载知识库目录：{data_path}")
        self.pdf_upload_folder_with_md5(data_path)

        logger.info("DSRagService 初始化完成 ✅")
        logger.info("=" * 60)

    def pdf_upload_folder_with_md5(self, folder_path: str) -> None:
        try:
            abs_folder = get_abs_path(folder_path)
            file_list = os.listdir(abs_folder)
            logger.info(f"扫描目录文件总数：{len(file_list)} 个")

            pdf_count = 0
            for file_name in file_list:
                file_path = os.path.join(abs_folder, file_name)

                if not os.path.isfile(file_path):
                    continue
                if not file_name.lower().endswith(".pdf"):
                    continue

                pdf_count += 1
                logger.info(f"正在处理 PDF：{file_name}")
                result = self.kb_service.upload_entire_pdf(file_path, file_name)
                logger.info(f"处理完成：{file_name} | 结果：{result}")

            logger.info(f"本次共加载 {pdf_count} 个PDF文件")

        except Exception as e:
            logger.error(f"批量加载PDF失败：{str(e)}")

    def format_docs(self, docs: List[Document]) -> str:
        """
        将检索到的文档格式化为带页码溯源的字符串
        Args:
            docs: Document列表
        Returns:
            格式化后的参考资料文本
        """
        if not docs:
            logger.warning("无文档可格式化")
            return "未找到相关资料"

        logger.info(f"开始格式化 {len(docs)} 个文档")
        formatted_list = []

        for idx, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知文件")
            page = doc.metadata.get("page_num") or doc.metadata.get("page") or 1
            page = max(int(page), 1)
            content = doc.page_content.strip()

            item = f"【参考资料{idx} | {source} 第{page}页】\n{content}"
            formatted_list.append(item)

        logger.info("格式化完成，返回带页码溯源文本")
        return "\n\n".join(formatted_list)

    def search(self, query: str, k: Optional[int] = None) -> str:
        """
        对外核心检索接口：返回带页码的格式化结果
        Args:
            query: 查询问题
            k: 召回数量，为空则使用配置默认值
        Returns:
            格式化后的参考资料
        """
        logger.info("=" * 50)
        logger.info(f"用户检索查询：{query}")

        try:
            k = k or self.k_default_k
            logger.info(f"召回数量 k = {k}")

            retriever = self.vector_service.get_retriever(search_kwargs={"k": k})
            docs = retriever.invoke(query)

            if not docs:
                logger.warning("检索结果为空")
                return "未在王道408数据结构知识库中找到相关内容"

            logger.info(f"检索成功，命中 {len(docs)} 条结果")
            return self.format_docs(docs)

        except Exception as e:
            logger.error(f"检索执行异常：{str(e)}")
            return f"检索服务暂时不可用：{str(e)}"

    def search_with_scores(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        内部使用接口：返回原始 Document 对象（供后续结构化处理）
        Args:
            query: 查询语句
            k: 召回数量
        Returns:
            Document 列表
        """
        try:
            k = k or self.k_default_k
            retriever = self.vector_service.get_retriever(search_kwargs={"k": k})
            return retriever.invoke(query)
        except Exception as e:
            logger.error(f"search_with_scores 执行失败：{str(e)}")
            return []


# ===================== 单例模式 =====================
_rag_service_instance: Optional[DSRagService] = None


def get_rag_service() -> DSRagService:
    global _rag_service_instance

    if _rag_service_instance is None:
        _rag_service_instance = DSRagService()
        logger.info("RAG 服务单例创建完成")

    return _rag_service_instance