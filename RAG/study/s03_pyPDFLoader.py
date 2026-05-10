from langchain_community.document_loaders import PyPDFLoader



loader = PyPDFLoader(
    file_path = '/Users/pc/Desktop/数据结构/3.1 栈.pdf',
    mode = "single"   #不写的话就是默认的page模式，一页是一个document。
                      #single把整个pdf -> 一个document对象
)

i = 0
for doc in loader.load():
    i += 1
    print(doc)
    print(i)