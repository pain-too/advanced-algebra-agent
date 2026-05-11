# 内置库
import hashlib
import os
from datetime import datetime
# 第三方库
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 自定义模块（完整路径）
import config_data as config


def check_md5(md5_hex:str):
    if not os.path.exists(config.md5_path):
        with open(config.md5_path, "w",encoding="utf-8") as f:
            pass
        return False
    else:
        with open(config.md5_path, "r",encoding="utf-8") as f:
            for line in f:
                if line.strip() == md5_hex:
                    return True
            return False


def save_md5(md5_hex) -> None:
    with open(config.md5_path, "a", encoding="utf-8") as f:
        f.write(md5_hex + "\n")


def get_string_md5(input_str) -> str:
    bytes = input_str.encode()
    md5_obj = hashlib.md5()
    md5_obj.update(bytes)
    md5_hex = md5_obj.hexdigest()
    return md5_hex



def get_file_md5(file_path: str) -> str:
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


#==========================================KnowledgeBaseService====================================
class KnowledgeBaseService(object):
    def __init__(self):
        os.makedirs(config.persist_directory, exist_ok=True)

        self.chroma = Chroma(
            collection_name = config.collection_name,
            embedding_function = DashScopeEmbeddings(model = "text-embedding-v4"),
            persist_directory = config.persist_directory,
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = config.chunk_size,
            chunk_overlap = config.chunk_overlap,
            separators = config.separators,
            length_function = len,
        )



    def upload_by_str_with_page(self, data: str, filename: str, page_num: int):
        if len(data) > config.max_split_char_number:
            knowledge_chunks: list[str] = self.splitter.split_text(data)
        else:
            knowledge_chunks = [data]


        metadata = {
            "source": filename,
            "page_num": page_num,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "操作者1"
        }

        self.chroma.add_texts(
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )

        return "【成功】 内容已加载到向量库"


    # =============================================================================
    # 【 核心优化 —— 整文件 MD5 校验 + 按页加载 】
    # 逻辑：
    # 1. 计算MD5                      <--不变
    # 2. 存在  整文件跳过               <--不变
    # 3. 不存在  逐页加载、分片、保存页码  <--新增
    # =============================================================================
    def upload_entire_pdf(self, pdf_path: str, filename: str):
        file_md5 = get_file_md5(pdf_path)
        if check_md5(file_md5):
            return f"【跳过】{filename}（文件已入库）"


        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        for page_num, page_doc in enumerate(pages, start=1):
            content = page_doc.page_content
            self.upload_by_str_with_page(content, filename, page_num)


        save_md5(file_md5)
        return f"【成功】{filename} 全部页面已入库"

    # 清空脏数据
    def clear_all_data(self):
        self.chroma.delete_collection()
        self.chroma = Chroma(
            collection_name=config.collection_name,
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
            persist_directory=config.persist_directory,
        )
        if os.path.exists(config.md5_path):
            os.remove(config.md5_path)
        print("✅ 已清空所有脏数据！")