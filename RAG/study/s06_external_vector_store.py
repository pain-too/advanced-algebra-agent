#文件材料为info.csv
#和内存存储的唯一区别：loader，其他完全一样
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader



loader = CSVLoader(
file_path = "/Users/pc/Documents/GitHub/LLM-study-notes/RAG4.20-/material/info.csv",
source_column = "source"        #指定本条数据的来源（列）
)
documents = loader.load()

vector_store = Chroma(
    collection_name = "test",
    embedding_function = DashScopeEmbeddings(),
    persist_directory = "/Users/pc/Documents/GitHub/LLM-study-notes/RAG4.20-/material"
)


#向量存储的新增
vector_store.add_documents(
    documents = documents,              #被添加的文档，类型是list[document]
    ids = ["id"+str(i) for i in range(1,len(documents)+1)]  #给加进来的文档加序号ids
)

#删除
vector_store.delete(["id1","id2"])

#检索
res = vector_store.similarity_search(
    query = "python是不是简单易学的",
    k = 3,
    filter = {"source": "黑马程序员"}    #过滤。只要source是黑马程序员的数据
)

print(res)