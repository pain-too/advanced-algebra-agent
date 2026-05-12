import os
import hashlib
from langchain_chroma import Chroma
import chromadb
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 你的统一工具 & 配置
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from model.factory import embedding_model  # 从工厂取模型（软编码）


class KnowledgeBaseService:
    """
    王道408数据结构知识库服务
    统一PDF加载、MD5去重、向量库存储
    """

    def __init__(self):
        # ===================== 全配置读取（无任何硬编码） =====================
        self.persist_directory = get_abs_path(chroma_conf.get("persist_directory", "./chroma_db"))
        self.collection_name = chroma_conf.get("collection_name", "data_structure_408")
        self.chunk_size = chroma_conf.get("chunk_size", 500)
        self.chunk_overlap = chroma_conf.get("chunk_overlap", 50)
        self.max_split_char_number = chroma_conf.get("max_split_char_number", 1000)
        self.md5_path = get_abs_path(chroma_conf.get("md5_path", "./md5_record.txt"))

        # 分隔符
        self.separators = chroma_conf.get("separators", ["\n\n", "\n", "。", "！", "？", "；", "，", " "])

        # ===================== 向量库初始化 =====================
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.chroma = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=embedding_model,
        )

        # 文本分片器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )

        logger.info(f"✅ KnowledgeBaseService 初始化完成")
        logger.info(f"📂 向量库路径：{self.persist_directory}")
        logger.info(f"📄 集合名称：{self.collection_name}")

    # ===================== MD5 校验 =====================
    def calculate_md5(self, file_path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def load_md5_record(self) -> set:
        if not os.path.exists(self.md5_path):
            return set()
        with open(self.md5_path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    def save_md5_record(self, md5_value: str):
        with open(self.md5_path, "a", encoding="utf-8") as f:
            f.write(md5_value + "\n")

    # ===================== PDF 按页上传（你原有核心逻辑不动） =====================
    def upload_entire_pdf(self, file_path: str, file_name: str) -> str:
        try:
            md5 = self.calculate_md5(file_path)
            existed = self.load_md5_record()

            if md5 in existed:
                logger.info(f"⏭️ 文件已存在（MD5 去重）：{file_name}")
                return "已存在，跳过"

            loader = PyPDFLoader(file_path)
            pages = loader.load()
            docs = []

            for page in pages:
                page_num = page.metadata.get("page", 0) + 1
                page.metadata["source"] = file_name
                page.metadata["page_num"] = page_num

                content = page.page_content.strip()
                if not content:
                    continue

                # 配置项：max_split_char_number 从配置读取
                if len(content) > self.max_split_char_number:
                    splitted = self.text_splitter.split_text(content)
                    for t in splitted:
                        new_doc = Document(page_content=t, metadata=page.metadata)
                        docs.append(new_doc)
                else:
                    docs.append(page)

            if docs:
                self.chroma.add_documents(docs)
                self.save_md5_record(md5)
                logger.info(f"✅ PDF 入库成功：{file_name}，总页数：{len(pages)}")
                return "上传成功"
            else:
                logger.warning(f"⚠️ 无有效内容：{file_name}")
                return "无有效内容"

        except Exception as e:
            logger.error(f"❌ 处理 PDF 失败：{file_name}，原因：{str(e)}")
            return f"处理失败：{str(e)}"

    # ===================== 清空向量库（你说不动，我就不动） =====================
    def clear_all_data(self):
        try:
            collections = self.client.list_collections()
            for coll in collections:
                self.client.delete_collection(coll.name)
            logger.info("🗑️ 已清空所有向量库数据")
        except Exception as e:
            logger.error(f"❌ 清空失败：{str(e)}")