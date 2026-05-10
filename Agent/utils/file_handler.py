# 系统库
import os
import hashlib

# 第三方库
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# 本地工具
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


#获取文件的md5十六进制指纹
#成功返回字符串md5，失败返回None
def get_file_md5_hex(file_path:str) -> str | None:
    file_path = get_abs_path(file_path)
    if not os.path.exists(file_path):
        logger.error(f"[md5计算]文件{file_path}不存在")
        return None
    if not os.path.isfile(file_path):
        logger.error(f"[md5计算]文件{file_path}不是文件")
        return None

    md5_obj = hashlib.md5()
    chunk_size = 4096     #4KB分片，避免文件过大
    try:
        with open(file_path, "rb") as f:        #rb专门用来读非文本文件(图片，视频...)
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
            """
                海象运算符       :=     把读和判断写在一起

                等价表示
                chunk = f.read(chunk_size)      # 先读一次
                while chunk:                    # 再判断循环
                    md5_obj.update(chunk)       # 记忆刚读到的 4KB ，不输出结果
                    chunk = f.read(chunk_size)  # 继续读下一片（文件对象 f 内部自带文件指针，自动读下一段）
            """
        hex_md5 = md5_obj.hexdigest()
        return hex_md5
    except Exception as e:
        logger.error(f"计算文件 {file_path} 的md5失败,{str(e)}")
        return None

#返回文件夹内的文件列表
def listdir_with_allowed_type(path:str,allowed_type:tuple[str]) -> list[str]:    #tuple元组
    files = []
    path = get_abs_path(path)

    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return files

    for f in os.listdir(path):
        # 拼接完整路径，用于判断是否为普通文件
        full_path = os.path.join(path, f)
        full_path = get_abs_path(full_path)
        # 增加判断：只筛选普通文件，跳过子文件夹，避免把文件夹当文件加载
        if os.path.isfile(full_path) and f.endswith(allowed_type):
            files.append(full_path)

    return files


# 给密码参数设默认值空字符串，调用时可不传
def pdf_loader(file_path:str,password:str = "") -> list[Document]:
    file_path = get_abs_path(file_path)
    try:
        return PyPDFLoader(file_path,password).load()
    except Exception as e:
        logger.error(f"加载PDF失败 {file_path}: {str(e)}")
        return []

def txt_loader(file_path:str) -> list[Document]:
    file_path = get_abs_path(file_path)
    try:
        return TextLoader(file_path, encoding="utf-8").load()
    except Exception as e:
        logger.error(f"加载TXT失败 {file_path}: {str(e)}")
        return []