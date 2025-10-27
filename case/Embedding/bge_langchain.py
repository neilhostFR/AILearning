import os
#pip install langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

def get_documents(path):
	loader = TextLoader(path, encoding="utf-8")
	documents = loader.load()
	return documents

def get_embedding_function(normalize_embeddings,device):
	model_kwargs = {'device': device}  # 如果有GPU，可以设置为 'cuda'
	encode_kwargs = {'normalize_embeddings': normalize_embeddings}  # 归一化向量，便于相似度计算

	embedding_function = HuggingFaceEmbeddings(
	    model_name="BAAI/bge-large-zh-v1.5",  # 模型名称
	    model_kwargs=model_kwargs,
	    encode_kwargs=encode_kwargs
	)

	return embedding_function


if __name__=="__main__":
	documents=get_documents('./source/three_kingdoms.txt')
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
	split_docs = text_splitter.split_documents(documents)
	print(split_docs)
	embedding_function=get_embedding_function(normalize_embeddings=True,device="mps")
	texts = [doc.page_content for doc in split_docs]
	embeddings=embedding_function.embed_documents(texts)
	
	instruction = "为这个句子生成表示以用于检索相关文章："
	query = "曹操"
	embedded_query=embedding_function.embed_query(instruction+query)
	print(sorted(embedded_query))