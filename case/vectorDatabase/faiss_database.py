import os
import sys
import json
#pip install langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
sys.path.append("..") 
from Embedding.bge_langchain import get_documents,get_embedding_function

def add_documents_to_faiss(embeddings,documents,save_path,embedding_function):
	'''
	将embeddings保存到faiss向量数据库faiss_index.index
	并将原始文档内容保存到documents.txt
	faiss创建方法说明
		1、IndexFlatL2 / IndexFlatIP
		原理：暴力搜索，计算查询向量与数据库中所有向量的L2距离或内积。
		优点：精度100%，无误差。
		缺点：速度慢，不适合大规模数据。
		使用场景：小规模数据集（例如少于1万条），或者作为其他索引的基准测试。

		2、IndexIVFFlat (Inverted File System with Flat storage)
		原理：通过聚类将向量空间划分为多个 Voronoi 单元，搜索时只搜索部分单元内的向量。
		优点：搜索速度快，适合大规模数据。
		缺点：需要训练，有误差（可能不是精确最近邻）。
		使用场景：大规模数据集（百万级），可以容忍少量误差。

		3、IndexIVFPQ (Inverted File System with Product Quantization)
		原理：在IVF的基础上，对向量进行乘积量化，进一步压缩向量，减少内存占用。
		优点：内存占用小，适合海量数据。
		缺点：有损压缩，误差相对IVFFlat更大。
		使用场景：海量数据（千万级以上），内存有限，对精度要求不是极高的场景。

		4、IndexHNSW (Hierarchical Navigable Small World)
		原理：基于图结构的索引，通过构建多层图实现快速近似最近邻搜索。
		优点：搜索速度快，精度较高，无需训练。
		缺点：内存占用大，构建索引较慢。
		使用场景：中等规模数据（百万级以内），对查询速度要求高，内存充足。

		5、IndexLSH (Locality-Sensitive Hashing)
		原理：使用局部敏感哈希将相似向量映射到同一个桶中。
		优点：内存占用小，查询速度快。
		缺点：精度较低，需要调整参数。
		使用场景：对内存要求严格，对精度要求不高的场景。

		6、IndexScalarQuantizer (SQ)
		原理：使用标量量化对向量进行压缩。
		优点：压缩向量，减少内存占用。
		缺点：有损压缩。
		使用场景：内存有限，需要压缩向量的场景。

		7、IndexPQ (Product Quantization)
		原理：将高维向量分解为多个低维子空间，并对每个子空间进行量化。
		优点：高压缩比，内存占用极小。
		缺点：有损压缩，误差较大。
		使用场景：海量数据，内存极度受限。

		8、IndexIVFScalarQuantizer (IVF with Scalar Quantization)
		原理：结合IVF和标量量化。
		优点：在IVF的基础上进一步压缩。
		缺点：有损压缩。
		使用场景：大规模数据，内存受限。

		9、IndexPreTransform
		原理：在索引之前对向量进行线性或非线性变换（如PCA、旋转等）。
		优点：可以降低维度或调整向量分布，提高索引效率。
		缺点：增加计算开销。
		使用场景：需要先对向量进行预处理的情况。

		10、IndexIDMap
		原理：用于为索引中的向量分配自定义ID。
		优点：可以管理自定义ID。
		缺点：无压缩或加速。
		使用场景：需要自定义向量ID的情况。
	'''
	index_path=f"{save_path}/faiss_index.index"
	json_path=f"{save_path}/documents.json"

	if os.path.exists(index_path) and os.path.exists(json_path):
		# 添加
		index=faiss.read_index(index_path)
		with open(json_path,"r",encoding="utf-8") as f:
			doc_map=json.load(f)
		start_id=len(doc_map)
		print(f"现有索引包含{start_id}个文档")
	else:
		# 新增
		sample_embedding=embedding_function.embed_documents([new_docuemnts[0].page_content])
		dimension=len(sample_embedding[0])
		index=faiss.IndexFlatIP(dimension)
		doc_map={}
		start_id=0

	new_tests=[doc.page_content for doc in new_docuemnts]
	new_embeddings=embedding_function.embed_documents(new_tests)
	new_embeddings_array=np.array(new_embedding).astype("float32")

	index.add(new_embeddings_array)

	for i,doc in enumerate(new_docuemnts):
		doc_id=start_id+1
		doc_map[str(doc_id)]={
			"content":doc.page_content,
			"metadata":doc.metadata,
			"length":len(doc.page_content)
		}
	os.makedirs(save_path,exist_ok=True)
	faiss.write_index(index,index_path)
	with open(json_path,'w',encoding="utf-8") as f:
		json.dump(doc_map,f,ensure_ascii=False,indent=2)

def load_index(load_path):
	'''
	加载faiss数据库
	加载文档映射
	'''
	index=faiss.read_index(f"{load_path}/faiss_index.index")
	doc_map={}
	with open(f"{load_path}/documents.json", "r", encoding="utf-8") as f:
		doc_data = json.load(f)
		for idx,data in doc_data.items():
			doc_map[int(idx)] = data["content"]
	print(f"从 JSON 格式加载了 {len(doc_map)} 个文档")
	return index,doc_map

def save_with_langchain_faiss(embeddings,documents,embedding_function,save_path):
	vectorstore=FAISS.form_embedding(
		text_embedding=list(zip([doc.page_content for doc in documents],embeddings)),
		embedding=embedding_function,
		metadatas=[doc.metadata for doc in documents] #保存元数据
	)

	vectorstore.save_local(save_path)


if __name__=="__main__":
	# documents=get_documents('../Embedding/source/three_kingdoms.txt')
	# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
	# split_docs = text_splitter.split_documents(documents)

	# embedding_function=get_embedding_function(normalize_embeddings=True,device="mps")
	# texts = [doc.page_content for doc in split_docs]
	# embeddings=embedding_function.embed_documents(texts)

	# 保存数据
	# add_documents_to_faiss(embeddings,split_docs,"./faiss",embedding_function)
	
	# 读取数据
	# index,doc_map=load_index("./faiss")

	# # 查询
	# query="周瑜火烧赤壁"
	# query_embedding=embedding_function.embed_query(query)
	# query_vector=np.array([query_embedding]).astype("float32")
	# distances, indices=index.search(query_vector,k=3)
	# print(f"查询：{query}")
	# for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
	# 	print(f"{i+1}. 相似度: {distance:.4f}")
	# 	print(f"   内容: {doc_map[idx]}")
	# 	print()


	# 在langchain中使用faiss
	# 保存
	save_path="./faiss_langchain"
	save_with_langchain_faiss(embeddings,split_docs,embedding_function,save_path)
	# 加载
	loaded_vectorstore=FAISS.load_local(
		save_path,
		embedding_function,
		allow_dangerous_deserialization=True
	)

	query="关羽"
	result=loaded_vectorstore.similarity_search(query,k=3)
	for i,doc in enumerate(result):
		print(f"{i+1}，内容：{doc.page_content}")
		print(f"	元数据：{doc.metadata}")



