# 内置库
import hashlib
import os
from datetime import datetime
# 第三方库
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
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




#==========================================基础函数已定义，现定义KnowledgeBaseService类====================================
class KnowledgeBaseService(object):
    def __init__(self):
        #如果文件夹不存在则创建
        os.makedirs(config.persist_directory, exist_ok=True)

        self.chroma = Chroma(
        #需要配置两个属性，config.collection_name和config.persist_directory，均写在config_data.py中
            collection_name = config.collection_name, #数据库的表名称
            embedding_function = DashScopeEmbeddings(model = "text-embedding-v4"),
            persist_directory = config.persist_directory,   #数据库本地存储文件夹
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = config.chunk_size,             #每个分割后的文本段的最大长度
            chunk_overlap = config.chunk_overlap,       #字符重合数
            separators = config.separators,              #分隔符（标点）
            length_function = len           #默认用python自带的len函数统计长度
        )

    def upload_by_str(self, data: str, filename):
        md5_hex = get_string_md5(data)

        if check_md5(md5_hex):  # 不论是否分割，都得到列表套字符串
            return "【跳过】，内容已存在"

        if len(data) > config.max_split_char_number:  # 文本达到一定规模才分割
            knowledge_chunks: list[str] = self.splitter.split_text(data)
        else:
            knowledge_chunks = [data]

        metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 转换为人眼常见的时间格式
            "operator": "操作者1"
        }
        self.chroma.add_texts(  # 内容加载到向量库中
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )

        # 表明已经把处理后的数据存入md5
        save_md5(md5_hex)
        return "【成功】内容已加载到向量库"
