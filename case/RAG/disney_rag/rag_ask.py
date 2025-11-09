import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from openai import OpenAI
from build_knowledge_base import BuildKnowledge
from transformers import CLIPProcessor,CLIPModel
import torch
import numpy as np

class RagAsk:
	def __init__(self,knowledge:BuildKnowledge):
		metadata_store,text_index,image_index=knowledge.build_knowledge_base()
		self.metadata_store=metadata_store
		self.text_index=text_index
		self.image_index=image_index
		self.get_text_embedding=knowledge.get_text_embedding
		self.client=knowledge.client
		self.clip_model=knowledge.clip_model
		self.clip_processor=knowledge.clip_processor

	def do_ask(self,query:str,k=3):
		print(f"---收到用户提问:{query}---")

		print(f"将用户问题向量化并进行检索")
		retrieved_context=[]

		# 文本检索
		query_embedding=self.get_text_embedding(query)
		query_text_vec=np.array([query_embedding]).astype("float32")
		distances,text_ids=self.text_index.search(query_text_vec,k)
		for i,doc_id in enumerate(text_ids[0]):
			if doc_id!=-1:
				match=next((item for item in self.metadata_store if item["id"]==doc_id),None)
				if match:
					retrieved_context.append(match)
					print(f"-文本检索命中（ID:{doc_id}）,距离:{distances[0][i]:4f}")
		# 图片检索
		if any(keyword in query.lower() for keyword in ["海报", "图片", "长什么样", "看看", "万圣节", "聚在一起"]):
			print("存在图片检索关键字，进行图片检索...")
			# optimized_query = self.optimize_query_for_image_search(query)
			# print(f"optimized_query:{optimized_query}")
			query_clip_vec=np.array([self.get_clip_text_embedding(query)]).astype("float32")
			distances,image_ids=self.image_index.search(query_clip_vec,10)
			initial_image_results = []
			for i,doc_id in enumerate(image_ids[0]):
				if doc_id!=-1:
					match=next((item for item in self.metadata_store if item["id"]==doc_id),None)
					if match:
						context_text=f"找到一张相关图片，图片路径: {match['path']}。图片上的文字是: '{match['ocr']}'"
						result_item = {"type": "image_content", "content": context_text, "metadata": match}
						initial_image_results.append(result_item)
						# retrieved_context.append({"type":"image_content","content":context_text,"metadata": match})
						print(f"图片检索命中（ID:{doc_id}）,距离:{distances[0][i]:4f}")
			# rerank
			reranked_results=self.rerank_images_by_ocr(query,initial_image_results,self.metadata_store,top_k=1)

			for result in reranked_results:
				retrieved_context.append(result)


		print(f"-开始构建prompt")

		context_str=""
		for i,item in enumerate(retrieved_context):
			content=item.get("content","")
			source=item.get("metadata",{}).get("source",item.get("source","未知来源"))
			context_str+=f"背景知识{i+1}(来源{source}):\n{content}\n\n"
		prompt=f"""你是一个迪士尼客服助手，请根据以下背景知识，用友好和专业的语气回答用户的问题，请只使用背景知识中的信息，不要自行发挥。
[背景知识]
{context_str}
[用户问题]
{query}
"""
		print("---prompt内容 [开始]---")
		print(prompt)
		print("---prompt内容 [结束]---")

		try:
			completion=self.client.chat.completions.create(
				model="qwen3:30b",
				messages=[
					{"role":"system","content":"你是一个迪士尼客服助手"},
					{"role":"user","content":prompt}
				]
			)
			final_answer=completion.choices[0].message.content

			# image_path_found=None
			# for item in retrieved_context:
			# 	if item.get("type")=="image_content":
			# 		image_path_found=item.get("metadata",{}).get("path")
			# 		break
			# if image_path_found:
			# 	final_answer+=f"\n\n(同时，我为您找到了相关图片，图片路径为：{image_path_found})"
		except Exception as e:
			final_answer=f"调用LLM时出错:{e}"

		print("\n--- 最终输出 ---")
		print(final_answer)
		return final_answer	


	def get_clip_text_embedding(self,text:str):
		inputs=self.clip_processor(text=text, return_tensors="pt")
		with torch.no_grad():
			text_features=self.clip_model.get_text_features(**inputs)
		# 归一化
		# text_features = text_features / text_features.norm(dim=-1, keepdim=True)
		return text_features[0].numpy()

	# 在图片检索后添加重排序
	def rerank_images_by_ocr(self,query, image_results, metadata_store, top_k=3):
		"""基于OCR文本内容对图片结果进行重排序"""
		from sklearn.feature_extraction.text import TfidfVectorizer
		from sklearn.metrics.pairwise import cosine_similarity
		import jieba
	    
		# 提取OCR文本
		ocr_texts = []
		valid_results = []
	    
		for result in image_results:
			doc_id = result["metadata"]["id"]
			match = next((item for item in metadata_store if item["id"] == doc_id), None)
			if match and match.get("ocr"):
				ocr_texts.append(match["ocr"])
				valid_results.append(result)
	    
		if not ocr_texts:
			return image_results[:top_k]
	    
		# 使用TF-IDF计算文本相似度
		all_texts = [query] + ocr_texts
	    # 简单中文分词
		tokenized_texts = [" ".join(jieba.cut(text)) for text in all_texts]
	    
		vectorizer = TfidfVectorizer()
		try:
			tfidf_matrix = vectorizer.fit_transform(tokenized_texts)
			similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
	        
			# 按相似度排序
			scored_results = list(zip(valid_results, similarities))
			scored_results.sort(key=lambda x: x[1], reverse=True)
	        
			return [result for result, score in scored_results[:top_k]]
		except:
			return image_results[:top_k]

if __name__=="__main__":
	knowledge=BuildKnowledge()
	rag_ask=RagAsk(knowledge)

	# rag_ask.do_ask("我想了解一下迪士尼门票的退款流程")

	print("\n----------------------------------")
	rag_ask.do_ask("最近万圣的活动海报是什么")

	print("\n----------------------------------")
	# rag_ask.do_ask("迪士尼年卡有什么优惠")