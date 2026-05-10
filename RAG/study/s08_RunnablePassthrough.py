#在上一步的基础上，把检索这一步加入链中
#复制代码，在上一步基础上修改
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



model = ChatTongyi(model = "qwen3-max")
prompt = ChatPromptTemplate.from_messages([
    ("system","以提供的参考资料为例，回答用户的问题。参考资料是{context}"),
    ("user","我的问题是{input}")
])

vector_store = InMemoryVectorStore(
    embedding = DashScopeEmbeddings(model = "text-embedding-v4"))

vector_store.add_texts(["减肥就是少吃多练","减肥期间吃的东西很重要，清淡少油控卡并且运动起来","跑步是很好的运动"])
input_text = "怎么减肥？"

#============================================以上复制，以下新增===========================================
retriever = vector_store.as_retriever(search_kwargs = {"k" : 2})

'''
想象中的链chain =  retriever | prompt | model | StrOutputParser()

retriever:
        输入：用户的提问        str
        输出：向量库检索结果     list[document]
prompt:
        输入：用户提问+检索结果   dict
        输出：完整提示词         PromptValue
两个问题      ①上一个输出是下一个输入，类型对不上
             ②retriever输出向量检索结果，不包含用户提问，会把prompt信息丢失
解决：想要把input同时给retriever和prompt
'''


def format_func(docs:list[Document]):
    formatted_str = "["
    for doc in docs:
        formatted_str += doc.page_content
    formatted_str += "]"
    return formatted_str

"""
分隔的规则
    1、先按 分隔符（句号、逗号、换行） 天然断开
    2、再把断开后的小段拼起来
    3、拼到快要接近 chunk_size 就停止
"""

chain = (
    {"input": RunnablePassthrough(),"context": retriever | format_func} | prompt | model | StrOutputParser()
)   #想要一分为二，①用链的嵌套，retriever是真正的“Ker”②RunnablePassthrough()可以截流，也分一份input
















res = chain.invoke(input_text)
print(res)

