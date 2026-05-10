# 内置库
import os
# 第三方库
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
# 自定义模块
from file_history_store import get_history
from vector_stores import VectorStoreService
from KnowledgeBaseService import KnowledgeBaseService



class RagService(object):
    def __init__(self):
        self.chat_model = ChatTongyi(model="qwen3-max")
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "请你根据参考资料回答问题不要胡说八道。参考资料：{context}\n\n用户的历史对话记录如下："),
                MessagesPlaceholder("chat_history"),
                ("human", "用户问题是{input}，优先在资料中寻找答案，如果没有的话，先打印【非资料中的答案如下】，再显示网络上搜索的答案")
            ]
        )
        self.vector_service = VectorStoreService(embedding=DashScopeEmbeddings(model="text-embedding-v4"))
        self.kb_service = KnowledgeBaseService()
        self.vector_service.vector_store = self.kb_service.chroma
        self.conversation_chain = self.get_chain()

    def pdf_upload_folder_with_md5(self, folder_path):
        if not os.path.exists(folder_path):
            print("【Error】未找到文件夹")
            return

        for file in os.listdir(folder_path):
            if file.endswith(".pdf"):
                file_path = os.path.join(folder_path, file)
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                full_text = "\n".join([doc.page_content for doc in docs])   #把文档内容合并成一个大字符串
                result = self.kb_service.upload_by_str(full_text, file)
                print(f"文件 {file} 处理结果：{result}")



    def get_chain(self):
        retriever = self.vector_service.get_retriever()

        def format_func(docs: list[Document]):
            return "\n".join([doc.page_content for doc in docs])

        chain = (
                {
                    "input": lambda x: x["input"],  # 从输入获取
                    "context": lambda x: format_func(retriever.invoke(x["input"])),  # 从检索器获取
                    "chat_history": lambda x: x.get("chat_history", [])  # 从输入获取历史
                }
                | self.prompt_template
                | self.chat_model
                | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",  # 用户输入用这个键
            history_messages_key="chat_history"  # 历史消息用这个键
        )

        return conversation_chain





if __name__ == "__main__":
    session_config = {
        "configurable": {
            "session_id": "001",
        }
    }

    rag = RagService()

    rag.pdf_upload_folder_with_md5("./data")

    res = rag.conversation_chain.invoke(
        {"input": "栈的基本概念是什么"},
        config = session_config
    )
    print(res)