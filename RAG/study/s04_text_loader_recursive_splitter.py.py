'''
        TextLoader读取文本文件，并放入一个Document对象中。
                     》如果文件过长《
 使用文档分割器RecursiveCharacterTextSplitter,按照自然段落分割
'''
from langchain_community.document_loaders import TextLoader



loader = TextLoader(
    file_path ="/RAG4.20-/material/text.txt"
)
docs = loader.load()
print(docs,len(docs))#输出长度是1,只有一个document

#======================================文档分割=====================================
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,    #每段的最大字符数
    chunk_overlap = 50, #允许重复的字符数（保证上下文连贯）
    separators = ["\n\n","/n""!","?","？"," " "","，","。"] ,
    #遇到这些符号就分段，按需执行。如果"\n\n"分隔之后，每个len(document)>500，则用"\n"，以此类推
    length_function = len   #chunk_overlap允许重复，由于需要正确统计字符数，就用默认的统计字符函数len
)
splitted = splitter.split_documents(docs)
print(len(splitted))