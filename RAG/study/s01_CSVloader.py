from langchain_community.document_loaders import CSVLoader



loader = CSVLoader(
    file_path ="/RAG4.20-/material/stu.csv",
    csv_args = {
        "delimiter":",",        #自定义分隔符，遇到这个逗号就分开
        "quotechar":"'",        #如果字段被单引号包围，则不分隔，当做整个字段处理
        "fieldnames":['a','b','c']#强制给一个表头。注意：
                                            #①如果已经有表头就不要写了，会把原来的表头当成数据
                                            #②把第一列叫做a，第二列b...
    }
)

#=======================批量加载======================
documents = loader.load()
for document in documents:
    print(document,type(document))

#========================懒加载=======================
'''for document in loader.lazy_load():
    print(document)'''


