from langchain_core.tools import tool
from rag.ds_rag_service import get_rag_service

# 包含页码溯源、MD5去重
rag_service = get_rag_service()



@tool
def ds_knowledge_search(query: str) -> str:
    """
    王道408数据结构 —— 知识点检索工具
    适用场景：查询概念、定义、性质、原理、结构特点、课本原文
    参数：
        query: 用户的知识点问题，例如 什么是败者树、哈希表冲突解决方法
    """
    result = rag_service.search(query)
    if not result:
        return "未在王道408数据结构资料库检索到相关内容，请换一种提问方式。"
    return result





@tool
def ds_concept_compare(query: str) -> str:
    """
    王道408数据结构 —— 易混概念对比辨析工具
    适用场景：对比两个易混淆概念，如顺序表vs链表、B树vsB+树、栈vs队列
    参数：
        query: 需要对比的两个概念，例如 顺序表和链表、堆排序和快速排序
    """
    # 检索相关资料
    context = rag_service.search(query)

    # 结构化输出（你的亮点：细粒度格式要求）
    ans = f"""
# 408数据结构易混概念对比分析
## 对比对象：{query}

## 对比维度（考研标准）
1. 定义与结构特点
2. 时间/空间复杂度
3. 适用场景
4. 优点与缺点
5. 408 高频考点

## 王道教材原文参考
{context}
"""
    return ans







@tool
def ds_chapter_summary(chapter_name: str) -> str:
    """
    王道408数据结构 —— 章节知识点归纳工具
    适用场景：按章节归纳考点、算法、易错点、重难点
    参数：
        chapter_name: 章节名称，例如 二叉树、图、排序、查找
    """
    # 检索整章资料
    context = rag_service.search(chapter_name)

    # 结构化归纳（你的亮点：细粒度归纳）
    ans = f"""
# 408数据结构「{chapter_name}」章节归纳
## 归纳维度（适配考研）
1. 核心概念
2. 必考算法
3. 时间/空间复杂度
4. 易错点 & 常考题型
5. 真题关联考点

## 王道教材原文参考
{context}
"""
    return ans