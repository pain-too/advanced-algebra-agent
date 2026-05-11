from KnowledgeBaseService import KnowledgeBaseService

kb = KnowledgeBaseService()

# 直接看向量库里的数据
docs = kb.chroma.get(limit=5)
print("=== 查看向量库元数据 ===")
print(docs["metadatas"])