md5_path = "./data/md5.txt"

#Chroma
collection_name = "pdf_knowledge"
persist_directory = "./data"
embedding_name = "text-embedding-v4"

#RecursiveCharacterTextSplitter
chunk_size = 1500
chunk_overlap = 150
separators=[
        "\n\n",    # 段落
        "。",      # 句子
        ",","，","."],
max_split_char_number = 1000    #超过1000才分隔

similarity_threshold = 5  #检索返回匹配的文档数量

session_config = {
    "configurable": {
        "session_id": "001",
    }
}


