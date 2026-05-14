import streamlit as st
from rag.ds_rag_service import DSRagService
from react_agent import ReactAgent

# 页面配置
st.set_page_config(page_title="408答疑助手", page_icon="📚")
st.title("📚 王道408数据结构智能答疑助手")

# 初始化会话
if "messages" not in st.session_state:
    st.session_state.messages = []

# 只初始化一次 RAG + Agent
if "rag" not in st.session_state:
    st.session_state.rag = DSRagService()
if "agent" not in st.session_state:
    st.session_state.agent = ReactAgent()

# 展示历史聊天
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
prompt = st.chat_input("请输入你的问题...")

if prompt:
    # 显示用户问题
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 回答
    with st.chat_message("assistant"):
        # ===================== 【关键修复：先显示 RAG 定位】 =====================
        ref_text = st.session_state.rag.search(prompt, mode="location_only")


        # 状态提示 + 流式输出
        with st.spinner("🤖 正在生成回答..."):
            full_answer = ""
            placeholder = st.empty()

            for chunk in st.session_state.agent.execute_stream(prompt):
                full_answer += chunk
                placeholder.markdown(full_answer)

    # 保存到历史（把定位 + 回答一起存，历史记录才完整）
    full_display = ref_text + "\n\n" + full_answer
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_display
    })