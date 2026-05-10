#======================================先把KnowledgeBaseService类导入===================================================
import hashlib
import os
from langchain_chroma import Chroma
import config_data as config
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import time

#检查传入的md5字符串是否已经被处理过
def check_md5(md5_str:str):
    if not os.path.exists(config.md5_path):     #进入if表示文件不存在，未被处理过
        with open(config.md5_path, 'w',encoding = 'utf-8'):     #python的w，如果没有文件会自动创建，再关闭
            pass

    else:                                       #进入else则直接读取文件作比较
        for line in open(config.md5_path, 'r',encoding = 'utf-8').readlines():
            line = line.strip()     #去掉首尾换行空格
            if line == md5_str:
                return True
        return False        #如果文件存在，但是每一句都找不到，则return False


#将传入的md5字符串写入文件
def save_md5(md5_str:str) ->None :
    with open(config.md5_path,'a',encoding = 'utf-8') as f:
        f.write(md5_str + '\n')

#将传入的字符串转为md5字符串
def get_string_md5(input_str:str,encoding = 'utf-8') ->str :
    str_bytes = input_str.encode(encoding=encoding)  #把字符串转换为二进制数据（md5只认二进制）
    md5_obj = hashlib.md5()         #给hashlib.md5类创建实例
    md5_obj.update(str_bytes)       #更新内容（传入即将要转换的字节数组）
    md5_hex = md5_obj.hexdigest()   #得到md5的十六进制字符串
    return md5_hex




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
            "operator": "让千万人失业的黑马"
        }
        self.chroma.add_texts(  # 内容加载到向量库中
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )

        # 表明已经把处理后的数据存入md5
        save_md5(md5_hex)
        return "【成功】内容已加载到向量库"







#======================================再复制添加了st.session_state字典的01代码=============================================
'''修改如下
①每次运行都会创建新的KnowledgeBaseService类对象，所以把service对象存进st.session_state字典
②最后两行，调用class
'''
import streamlit as st
st.title("知识库更新服务")

#上传文件
uploaded_file = st.file_uploader(
    label = "请上传TXT文件",
    type = "txt",
    accept_multiple_files = False
)


#使用字典st.session_state
if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()


#输出基本信息
if uploaded_file is not None:
    file_name = uploaded_file.name
    file_size = uploaded_file.size / 1024
    file_type = uploaded_file.type

    st.subheader(f"文件名：{file_name}")
    st.write(f"文件类型：{file_type} |文件大小：{file_size:.2f}KB")

#获取具体内容（通过解码器转为字符串）
    text = uploaded_file.getvalue().decode("utf-8")

    with st.spinner("载入知识库中..."):
        time.sleep(1)
        res = st.session_state["service"].upload_by_str(text,file_name)    #通过这一行调用创建的类
        st.write(res)#write可以在网页看到结果
