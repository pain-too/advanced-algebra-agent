"""
基于Streamlit完成WEB网页上传服务
streamlit的机制：只要网页内容变化，代码就重新执行一遍
==>重跑一遍导致状态丢失（每上传一个文件，就忘记之前的过程）
==>解决方案：st.session_state（会话状态记录器，是一个字典）
"""
import streamlit as st



st.title("知识库更新服务")

#上传文件
uploaded_file = st.file_uploader(
    label = "请上传TXT文件",
    type = "txt",
    accept_multiple_files = False
)


#使用字典st.session_state
if "counter" not in st.session_state:
    st.session_state["counter"] = 0


#输出基本信息
if uploaded_file is not None:
    file_name = uploaded_file.name
    file_size = uploaded_file.size / 1024
    file_type = uploaded_file.type

    st.subheader(f"文件名：{file_name}")
    st.write(f"文件类型：{file_type} |文件大小：{file_size:.2f}KB")

#获取具体内容（通过解码器转为字符串）
    text = uploaded_file.getvalue().decode("utf-8")
    st.write(text)

    st.session_state["counter"] += 1


print(f"上传了{st.session_state["counter"]}个文件")