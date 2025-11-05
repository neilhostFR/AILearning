#pip install chromadb langchain-chroma
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
import shutil
sys.path.append("..") 
from Embedding.bge_langchain import get_documents,get_embedding_function

def save_to_chroma(documents, embedding_function, persist_directory):
	if os.path.exists(persist_directory):
		shutil.rmtree(persist_directory)

	vectorstore=Chroma.form_documents(
		documents=documents,
		embedding=embedding_function,
		persist_directory=persist_directory
	)

	vectorstore.persist()

	return vectorstore

def load_form_chroma(embedding_function,persist_directory):
	vectorstore=Chroma(
		embedding=embedding_function,
		persist_directory=persist_directory
	)

	return vectorstore

def search_similar_documents(vectorstore,query,k=3):
	"""检索相似文档"""
	result=vectorstore.similarity_search(query,k=k)
	for i, doc in enumerate(results):
		print(f"{i+1}. 内容: {doc.page_content}")
		print(f"   元数据: {doc.metadata}")
		print(f"   长度: {len(doc.page_content)} 字符")
	return result

def search_with_scores(vectorstore,query,k):
	"""检索相似文档并返回相似度分数"""
	results = vectorstore.similarity_search_with_score(query, k=k)
	for i,(doc,score) in enumerate(results):
		print(f"{i+1}. 相似度: {score:.4f}")
		print(f"   内容: {doc.page_content[:100]}")
		print(f"   元数据: {doc.metadata}")

def get_vectorstore_info(vectorstore):
	collection=vectorstore._collection
	count=vectorstore.count()

	if count>0:
		sample=collection.get(limit=1)
		if "documents" in sample and len(sample["documents"])>0:
			print(f"样本文档长度:{len(sample["documents"][0])} 字符")
		if "metadata" in sample and len(sample["metadata"])>0:
			print(f"样本元数据:{sample["metadata"][0]}")
	return count

def add_documents_to_chroma(vectorstore,documents):
	vectorstore.add_documents(documents)
	print("添加成功")


if __name__=="__main__":
	documents=get_documents('../Embedding/source/three_kingdoms.txt')
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100,separators=["\n\n", "\n", "。", "！", "？", "，", "、", ""])
	split_docs = text_splitter.split_documents(documents)

	embedding_function=get_embedding_function(normalize_embeddings=True,device="mps")

	embeddings=embedding_function.embed_documents(texts)

	persist_directory="./chroma_db"

	vectorstore = save_to_chroma(split_docs, embedding_function, persist_directory)

	get_vectorstore_info(vectorstore)

	test_queries = ["刘备","关羽和张飞","曹操的谋士","诸葛亮出山"]
	for query in test_queries:
		search_similar_documents(vectorstore, query, k=2)
		search_with_scores(vectorstore, query, k=2)

	loaded_vectorstore = load_from_chroma(embedding_function, persist_directory)

	add_documents_to_chroma(vectorstore,split_docs)
