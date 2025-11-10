import os
import pandas as pd
from typing import List,Any
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
import uuid
import json

class BulidKnowledge:
	def __init__(
			self,
			knowledge_dir:str="./backend/knowledges_base",
			vectordb_path:str="./backend/chroma/knowledge_db",
			json_path:str="./backend/json_data.json"
		):
		self.knowledge_dir=knowledge_dir
		self.embeddings=HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
		self.vectordb=Chroma(
			persist_directory=vectordb_path,
			embedding_function=self.embeddings,
			collection_name="video_content"
		)
		self.vectordb_path=vectordb_path
		self.json_path=json_path

		# 只在需要时才读取Excel文件
		# self.read_xlsx()

	def read_xlsx(self):
		"""读取Excel文件并构建知识库"""
		documents=[]
		doc_map=self.get_json_data()
		ids=[item["id"] for item in doc_map]

		# 检查知识库目录是否存在
		if not os.path.exists(self.knowledge_dir):
			print(f"知识库目录 {self.knowledge_dir} 不存在")
			return

		for filename in os.listdir(self.knowledge_dir):
			file_path=os.path.join(self.knowledge_dir,filename)

			if filename.startswith(".") or os.path.isdir(file_path):
				continue

			if filename.endswith(".xlsx") or filename.endswith(".xls"):
				try:
					df=pd.read_excel(file_path)
					print(f"读取文件: {filename}, 共 {len(df)} 条记录")

					# 获取视频内容文字说明
					content_datas=df["视频内容"]
					for i,content in enumerate(content_datas):
						# 检查是否已存在相同的视频地址
						if df["视频地址"][i] in [item["video_address"] for item in doc_map]:
							continue
						document_id=f"video_content_{uuid.uuid4().hex[:8]}"
						metadata={
							"id":document_id,
							"documents":content,
							"video_address":df["视频地址"][i],
							"page":1,
							"source":df["源路径"][i]
						}
						ids.append(document_id)
						doc_map.append(metadata)
						documents.append(
							Document(page_content=content,metadata=metadata)
						)
				except Exception as e:
					print(f"读取文件 {filename} 时出错: {e}")
					continue
					
		# 无论是否有新文档，都重新保存整个知识库
		print(f"知识库中现有 {len(doc_map)} 条记录")
		if documents:
			print(f"新增 {len(documents)} 条记录到知识库")
		self.save_to_chroma(documents=documents,ids=ids)
		with open(self.json_path,'w',encoding="utf-8") as f:
			json.dump(doc_map,f,ensure_ascii=False,indent=2)
		print("知识库更新完成")

	def embedding_content(self,documents):
		"""对文档进行向量化"""
		# embeddings=HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
		return self.embeddings.embed_documents([documents])

	def get_json_data(self):
		"""获取JSON数据"""
		if not os.path.exists(self.json_path):
			return []

		with open(self.json_path,"r",encoding="utf-8") as f:
			doc_map=json.load(f)
		return doc_map

	def save_to_chroma(self,documents:List[Document],ids:List[str]):
		"""保存文档到Chroma向量数据库"""
		if documents:
			self.vectordb.add_documents(documents=documents,ids=ids)
		else:
			print("没有文档需要添加到向量数据库")


if __name__=="__main__":
	knowledge=BulidKnowledge()
	# help(HuggingFaceEmbeddings)
	# 手动触发读取Excel文件
	# knowledge.read_xlsx()

	result=knowledge.vectordb.similarity_search("codex",k=1)

	print(result)

