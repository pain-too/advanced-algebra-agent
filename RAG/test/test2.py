import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from final_rag_with_chat_history import RagService

print("="*60)
print("🔥 强制上传PDF + 检索测试")
print("="*60)

# 1. 创建RAG对象
rag = RagService()

# 2. 【关键！】手动执行PDF上传（你之前从来没执行过！）
print("\n📤 开始上传PDF到向量库...")
rag.pdf_upload_folder_with_md5("./data")

# 3. 查看向量库数量
count = rag.kb_service.chroma._collection.count()
print(f"\n✅ 向量库总片段数：{count}")

# 4. 测试检索
print("\n🔍 测试检索：矩阵的秩")
retriever = rag.vector_service.get_retriever()
docs = retriever.invoke("矩阵的秩的定义")

if len(docs) > 0:
    print("🎉 检索成功！从教材中找到内容：")
    print("="*60)
    print(docs[0].page_content)
    print("="*60)
else:
    print("❌ 仍然失败...")

# 5. 完整问答
print("\n💬 完整问答测试：")
session_config = {"configurable": {"session_id": "test"}}
res = rag.conversation_chain.invoke({"input": "矩阵的秩是什么"}, config=session_config)
print(res)