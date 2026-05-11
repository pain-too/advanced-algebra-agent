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
                ("system", """请你根据参考资料回答问题，并在每处引用后标注来源。
参考资料格式如下：
【文件名 第X页】
内容...

回答要求：
1. 优先根据参考资料回答。如果在资料中有相关内容先返回“这是资料中的内容”。如果没有参考资料，先返回“非参考资料的内容”
2. 在回答之前，标注【文件名 第X页】
3. 示例：
'''这是资料中的内容==>详见【数据结构.pdf 第3页】
模型回复...'''

参考资料：{context}

用户的历史对话记录如下："""),
                MessagesPlaceholder("chat_history"),
                ("human", "用户问题是{input}，请回答")
            ]
        )
        self.vector_service = VectorStoreService(embedding=DashScopeEmbeddings(model="text-embedding-v4"))
        self.kb_service = KnowledgeBaseService()
        self.vector_service.vector_store = self.kb_service.chroma
        self.conversation_chain = self.get_chain()
        # 自动加载 PDF 数据
        self.pdf_upload_folder_with_md5("./data")

    def pdf_upload_folder_with_md5(self, folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".pdf"):
                file_path = os.path.join(folder_path, file)
                # 整文件上传，内部自动按页处理 + 页码保留
                result = self.kb_service.upload_entire_pdf(file_path, file)
                print(f"文件 {file} 处理结果：{result}")





    def get_chain(self):
        retriever = self.vector_service.get_retriever()

        # final_rag_with_chat_history.py 中 format_func 函数修改
        def format_func(docs: list[Document]):
            formatted_docs = []
            for doc in docs:
                source = doc.metadata.get("source", "未知文件")

                # ==============================================
                # 🔥 终极修复：强制读取所有可能的页码字段
                # 先读你存的 page_num，没有就读 page，再没有就强制给 1
                # ==============================================
                page = doc.metadata.get("page_num") or doc.metadata.get("page") or 1

                # 强制转数字，永远 >= 1，绝对不会出现 0！
                page = max(int(page), 1)

                formatted_docs.append(f"【{source} 第{page}页】\n{doc.page_content}")

            return "\n\n".join(formatted_docs)



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
    #rag.kb_service.clear_all_data()  # 第一次跑打开，以后注释掉
    res = rag.conversation_chain.invoke(
        {"input": "败者树是什么"},
        config = session_config
    )
    print(res)