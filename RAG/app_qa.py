import streamlit as st
import config_data as config
from final_rag_with_chat_history import RagService

st.title("408《数据结构》知识问答系统")
st.divider()

# 初始化历史消息
if "message" not in st.session_state:
    st.session_state["message"] = [{"role":"assistant","content":"你好，有什么需要帮助的"}]

# 只初始化一次 RagService（关键修复）
if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()
    # 初始化时自动加载 PDF 数据（新增：确保向量库有数据）
    st.session_state["rag"].pdf_upload_folder_with_md5("./data")
# 直接使用持久化的实例
rag_service = st.session_state["rag"]

# 渲染历史消息
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入栏
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user","content":prompt})

    with st.spinner("AI thinking..."):
        # 调用持久化实例的 chain（关键）
        res_stream = rag_service.conversation_chain.stream(
            {"input":prompt},
            config.session_config
        )
        # 修复：write_stream 返回 None，需手动拼接结果
        full_res = ""
        for chunk in res_stream:
            full_res += chunk
            st.chat_message("assistant").write(chunk, unsafe_allow_html=True)
        # 存储完整结果到历史（关键修复）
        st.session_state["message"].append({"role": "assistant", "content": full_res})