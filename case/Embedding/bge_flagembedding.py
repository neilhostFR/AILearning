import os
from FlagEmbedding import FlagModel
import numpy as np

def get_file_content(path):
	content=[]
	with open(path) as f:
		content = f.read()
		content = content.split()
		
		# lines=f.readlines()
		# for line in lines:
		# 	# content+=line.split()
		# 	content.extend(line.split())
	# content=list(set(content))
	return content

def find_similar_words(query_word, corpus, model, top_k=10, exclude_query=True):
    
    corpus_embeddings=model.encode(corpus)
    # 编码查询词
    query_embedding=model.encode([query_word])
    
    # 计算相似度
    similarities = (query_embedding @ corpus_embeddings.T).flatten()
    
    # 创建结果列表
    results = []
    for i, score in enumerate(similarities):
        word = corpus[i]
        
        # 如果需要排除查询词本身
        if exclude_query and word == query_word:
            continue
            
        results.append({
            'word': word,
            'score': float(score),
            'index': i
        })
    
    # 按相似度排序，取前top_k个
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return results[:top_k]

if __name__=="__main__":
	content=get_file_content('./segment/segment_0.txt')
	model=FlagModel('BAAI/bge-large-zh-v1.5', 
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True)


	results = find_similar_words('曹操', content, model)
	
	print(results)