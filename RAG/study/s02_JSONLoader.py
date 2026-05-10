#针对stu.json（标准json）
from langchain_community.document_loaders import JSONLoader



loader = JSONLoader(
    file_path ="/RAG4.20-/material/stu_josn_lines.json.json",
    jq_schema = ".",
    text_content = False    #jq_schema是点一个点，代表要获取整个文件。需要加入这一句，告诉JSONLoader获取的不是字符串，就可以正常输出
)
document = loader.load()
print(document)





#针对stus.json（列表套json）
from langchain_community.document_loaders import JSONLoader
loader = JSONLoader(
    file_path ="/RAG4.20-/material/stus.json",
    jq_schema = ".[].name",     #.[]代表抽取整个数组。如果是.[0]代表选第一个。.name是选取某个对象
    text_content = False,
)
document = loader.load()
print(document)


#对于每
#
#
# 一行都是一个json的stu_json_lines，每一行都是正确的jso，但是多行放在一起就不对了
from langchain_community.document_loaders import JSONLoader
loader = JSONLoader(
    file_path ="/RAG4.20-/material/stus.json",
    jq_schema = ".[].name",     #.[]代表抽取整个数组。如果是.[0]代表选第一个。.name是选取某个对象
    text_content = False,
    json_lines = True,
)
document = loader.load()
print(document)