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
# 自定义模块（完整路径）
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
                ("human",
                 "用户问题是{input}，优先在资料中寻找答案，如果没有的话，先打印【非资料中的答案如下】，再显示网络上搜索的答案")
            ]
        )

        # 使用相同的 embedding 实例
        self.embeddings = DashScopeEmbeddings(model="text-embedding-v4")

        # 初始化知识库服务（它会创建 chroma）
        self.kb_service = KnowledgeBaseService()

        # 【关键修改】让 vector_service 直接使用 kb_service 的 chroma 实例
        # 而不是重新创建一个新的
        self.vector_service = VectorStoreService(embedding=self.embeddings)
        self.vector_service.vector_store = self.kb_service.chroma  # 共用同一个实例

        self.conversation_chain = self.get_chain()

    def pdf_upload_folder_with_md5(self, folder_path):
        """上传文件夹中的所有PDF文件到向量库"""
        if not os.path.exists(folder_path):
            print(f"【Error】未找到文件夹: {folder_path}")
            return

        pdf_count = 0
        for file in os.listdir(folder_path):
            if file.endswith(".pdf"):
                file_path = os.path.join(folder_path, file)
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    full_text = "\n".join([doc.page_content for doc in docs])

                    # 上传到知识库（会自动处理md5去重和分割）
                    result = self.kb_service.upload_by_str(full_text, file)
                    print(f"文件 {file} 处理结果：{result}")
                    pdf_count += 1
                except Exception as e:
                    print(f"处理文件 {file} 时出错：{str(e)}")

        print(f"\n共处理了 {pdf_count} 个PDF文件")

        # 【重要】上传后重新获取 chain，确保检索器能看到新数据
        # 由于 vector_service 和 kb_service 共用同一个 chroma，所以不需要重建
        # 但为了确保检索器是最新的，重新创建 chain
        self.conversation_chain = self.get_chain()

    def get_chain(self):
        """创建对话链"""
        retriever = self.vector_service.get_retriever()

        def format_func(docs: list[Document]):
            return "\n".join([doc.page_content for doc in docs])

        chain = (
                {
                    "input": lambda x: x["input"],
                    "context": lambda x: format_func(retriever.invoke(x["input"])),
                    "chat_history": lambda x: x.get("chat_history", [])
                }
                | self.prompt_template
                | self.chat_model
                | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

        return conversation_chain


if __name__ == "__main__":
    session_config = {
        "configurable": {
            "session_id": "001",
        }
    }

    rag = RagService()

    # 上传PDF文件
    rag.pdf_upload_folder_with_md5("./data")

    # 测试问答
    questions = [
        "二叉树的c语言代码实现",
        "什么是线性表？",
        "双链表和单链表的区别是什么？"
    ]

    for question in questions:
        print(f"\n{'=' * 50}")
        print(f"问题: {question}")
        print(f"{'=' * 50}")
        res = rag.conversation_chain.invoke(
            {"input": question},
            config=session_config
        )
        print(f"回答: {res}\n")