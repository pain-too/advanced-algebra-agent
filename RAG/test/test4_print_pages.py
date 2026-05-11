from KnowledgeBaseService import KnowledgeBaseService

kb = KnowledgeBaseService()

# 直接检索败者树，看返回的文档 page_num 是多少！
docs = kb.chroma.similarity_search("败者树", k=3)

print("="*50)
for i, doc in enumerate(docs):
    print(f"文档 {i+1}")
    print(f"内容: {doc.page_content[:50]}...")
    print(f"元数据: {doc.metadata}")
    print(f"page_num = {doc.metadata.get('page_num')}")
    print("-"*50)