#构建提示词，并在向量库中检索，把检索结果+问题->问模型（RAG在线流程）
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
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

#检索向量库
res = vector_store.similarity_search(input_text,k = 2)

reference_text = "["
for doc in res:
    reference_text += doc.page_content
reference_text += "]"

#chain
def print_prompt(prompt):
    print(prompt.to_string())
    print("="*20)
    return prompt

chain = prompt | print_prompt | model | StrOutputParser()

result = chain.invoke({"input":input_text,"context":reference_text})
print(result)