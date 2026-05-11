import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入你所有核心模块
from final_rag_with_chat_history import RagService
from KnowledgeBaseService import KnowledgeBaseService
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config

print("="*60)
print("🔍 开始逐环节测试 RAG 系统")
print("="*60)

# ====================== 测试1：检查配置文件 ======================
print("\n【测试1】读取配置文件...")
try:
    print(f"persist_directory: {config.persist_directory}")
    print(f"chunk_size: {config.chunk_size}")
    print(f"separators 长度: {len(config.separators)}")
    print("✅ 配置文件正常")
except Exception as e:
    print(f"❌ 配置文件错误: {e}")

# ====================== 测试2：测试分割器是否能运行 ======================
print("\n【测试2】测试文本分割器...")
try:
    kb = KnowledgeBaseService()
    test_text = "第1章 行列式\n§1.1 二阶与三阶行列式\n矩阵的秩定义：..."
    chunks = kb.splitter.split_text(test_text)
    print(f"✅ 分割器正常，分割成 {len(chunks)} 段")
except Exception as e:
    print(f"❌ 分割器错误: {e}")

# ====================== 测试3：测试向量库是否能连接 ======================
print("\n【测试3】测试向量库连接...")
try:
    vs = VectorStoreService(embedding=DashScopeEmbeddings(model="text-embedding-v4"))
    count = vs.vector_store._collection.count()
    print(f"✅ 向量库正常，当前存储片段数：{count}")
except Exception as e:
    print(f"❌ 向量库错误: {e}")

# ====================== 测试4：测试PDF自动读取 ======================
print("\n【测试4】测试PDF读取...")
data_path = "./data"
pdf_files = [f for f in os.listdir(data_path) if f.endswith(".pdf")]
print(f"📂 data 目录下 PDF 数量: {len(pdf_files)}")
for pdf in pdf_files:
    print(f" - {pdf}")
if len(pdf_files) == 0:
    print("❌ 没有找到任何PDF文件！")
else:
    print("✅ PDF 文件存在")

# ====================== 测试5：测试检索功能（最关键） ======================
print("\n【测试5】测试检索【矩阵的秩】...")
try:
    rag = RagService()
    retriever = rag.vector_service.get_retriever()
    docs = retriever.invoke("矩阵的秩的定义")
    if len(docs) == 0:
        print("❌ 检索失败：没有找到任何相关内容！")
    else:
        print(f"✅ 检索成功！找到 {len(docs)} 条相关内容")
        print("="*50)
        print(docs[0].page_content[:300] + "...")
        print("="*50)
except Exception as e:
    print(f"❌ 检索错误: {e}")

# ====================== 测试6：测试完整问答 ======================
print("\n【测试6】测试完整问答...")
try:
    rag = RagService()
    session_config = {"configurable": {"session_id": "test"}}
    res = rag.conversation_chain.invoke(
        {"input": "矩阵的秩是什么"},
        config=session_config
    )
    print("✅ 问答链运行成功！")
    print("\n===== 回答结果 =====")
    print(res)
except Exception as e:
    print(f"❌ 问答链错误: {e}")

print("\n" + "="*60)
print("🏁 测试完成！根据上面结果定位问题")
print("="*60)
