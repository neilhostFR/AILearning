
def tfidf_rerank(query,documents,top_k=3):
	"""
	TF-IDF/BM25 重排序
	适用场景：
		1、关键词匹配要求高的场景
		2、短文本检索
		3、需要精确匹配特定术语的情况
	pip install scikit-learn
	"""
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.metrics.pairwise import cosine_similarity

	vectorizer=TfidfVectorizer()
	tfidf_matrix=vectorizer.fit_transform([query]+documents)
	similarities=cosine_similarity(tfidf_matrix[0:1],tfidf_matrix[1:])

	scored_docs=list(zip(documents,similarities[0]))
	scored_docs.sort(key=lambda x:x[1],reverse=True)

	return [doc for doc in scored_docs[:top_k]]

def cross_encoder_rerank(query,documents,model_name="BAAI/bge-reranker-large"):
	"""
	交叉编码器
	适用场景：
		1、需要深度语义理解
		2、长文档和复杂查询
		3、高精度要求的场景
	"""
	from transformers import AutoModelForSequenceClassification,AutoTokenizer
	import torch

	tokenizer=AutoTokenizer.from_pretrained(model_name)
	model=AutoModelForSequenceClassification.from_pretrained(model_name)

	scores=[]

	for doc in documents:
		inputs=tokenizer(query,documents,return_tensors="pt",truncation=True)
		with torch.no_grad():
			outputs=model(**inputs)
			score=torch.softmax(outputs.logits,dim=1)[0][1].item()
			scores.append(score)

	return sorted(zip(documents,scores),key=lambda x:x[1],reverse=True)

def _calculate_text_similarity(query,ocr_text):
	"""使用TF-IDF计算查询与OCR文本的相似度"""
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.metrics.pairwise import cosine_similarity
	import jieba

	if not query or not ocr_text:
		return 0.0

	try:
		query_cut=" ".join(jieba.cut(query))
		ocr_cut=" ".join(jieba.cut(ocr_text))

		vectorizer=TfidfVectorizer()
		tfidf_matrix = vectorizer.fit_transform([query_cut, ocr_cut])

		# 计算余弦相似度
		similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

		return similarity
	except Exception as e:
		return 0.0



def multimodal_rerank(query,image_results,text_weight=0.7,visual_weight=0.3):
	"""
	结合文本和视觉特征的重排序
	适用场景
		1、多模态检索系统
		2、图片、视频等富媒体内容
		3、需要平衡文本和视觉信息的场景
	"""
	rerank_results=[]

	for result in image_results:
		#文本相似度分数
		text_similarity=_calculate_text_similarity(query,result["ocr"])

		# 视觉相似度分数（使用原始CLIP距离转换）
		visual_similarity = 1 / (1 + result['clip_distance'])

		# 加权综合分数
		combined_score = text_weight * text_similarity + visual_weight * visual_similarity

		reranked_results.append({
			**result,
			'combined_score': combined_score,
			'text_score': text_similarity,
			'visual_score': visual_similarity
		})

	return sorted(reranked_results, key=lambda x: x['combined_score'], reverse=True)

def rule_base_rerank(query,results,rules):
	"""
	基于业务规则的重排序
	适用场景
		1、企业知识库
		2、需要结合业务逻辑的场景
		3、多维度评估需求
	"""
	scored_results=[]
	for result in results:
		score=0

		# 时间新鲜度规则
		if "timestamp" in result:
			days_old=(datetime.now()-result["timestamp"]).days
			freshness_score=max(0,1-days_old/365) #一年内衰减

			score+=rules.get('freshness_weight', 0.2) * freshness_score

		# 来源权威性规则
		if "source" in result:
			authority_scores={
				"官方文档":1.0,
				"用户生成":0.3,
				"第三方":0.6
			}
			authority_score = authority_scores.get(result['source'], 0.5)
			score += rules.get('authority_weight', 0.3) * authority_score

		# 内容类型规则
		if "type" in result:
			type_scores={
				"教程":0.9,
				"参考":0.8,
				"示例":0.7,
				"讨论":0.5
			}
			type_score = type_scores.get(result['type'], 0.6)
			score += rules.get('type_weight', 0.2) * type_score

		# 长度适宜性规则
		if "content_length" in result:
			ideal_length = rules.get('ideal_length', 500)
			length_penalty = 1 - abs(result['content_length'] - ideal_length) / ideal_length

			score += rules.get('length_weight', 0.1) * max(0, length_penalty)

		scored_results.append({**result, 'rule_score': score})

		return sorted(scored_results, key=lambda x: x['rule_score'], reverse=True)

def personalized_rerank(user_id,query,results,user_preferences):
	"""
	基于用户历史行为的重排序
	适用场景:
		1、个性化推荐系统
		2、有用户行为数据的场景
		3、需要长期优化的系统
	"""

	scored_results=[]
	for result in results:
		score=0
		# 基于用户点击历史
		if user_id in user_preferences.get("click_history",{}):
			similar_clicks = find_similar_clicks(query, result, user_preferences['click_history'][user_id])
			score += 0.4 * len(similar_clicks)

		# 基于用户停留时间
		avg_dwell_time = user_preferences.get('avg_dwell_time', {}).get(user_id, 0)
		if avg_dwell_time > 30:  # 长时间停留用户偏好深度内容
			score += 0.3 * min(1, len(result.get('content', '')) / 1000)

		# 基于用户搜索历史
		search_history = user_preferences.get('search_history', {}).get(user_id, [])
		query_similarity = max([calculate_query_similarity(query, past_query) for past_query in search_history])
		score += 0.3 * query_similarity

		# 基于用户偏好主题
		preferred_topics = user_preferences.get('preferred_topics', {}).get(user_id, {})
		topic_match_score = calculate_topic_match(result, preferred_topics)
		score += 0.2 * topic_match_score

		scored_results.append({**result, 'personalized_score': score})

	return sorted(scored_results, key=lambda x: x['personalized_score'], reverse=True)


def temporal_rerank(query,results,time_decay_factor=0.1):
	"""
	考虑时间因素的重排序
	适用场景
		1、新闻、社交媒体检索
		2、时效性要求高的内容
		3、时间敏感查询（如"最新"、"最近"）
	"""
	current_time = datetime.now()
	scored_results = []

	for result in results:
		base_score = result.get('similarity_score', 0.5)
        
		# 时间衰减
		if 'timestamp' in result:
			time_diff = (current_time - result['timestamp']).days
			time_penalty = math.exp(-time_decay_factor * time_diff)
			final_score = base_score * time_penalty
		else:
			final_score = base_score

		scored_results.append({**result, 'temporal_score': final_score})
    
    return sorted(scored_results, key=lambda x: x['temporal_score'], reverse=True)