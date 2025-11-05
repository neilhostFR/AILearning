import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import ollama
import re
from document_processor import DocumentProcessor

class RAGEngine:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.keywords = set()  # 动态关键字集合
        
    def build_index(self, documents):
        """构建向量索引"""
        self.documents = documents
        if not documents:
            return
            
        # 生成文档向量
        doc_texts = [doc.content for doc in documents]
        embeddings = self.model.encode(doc_texts)
        
        # 创建Faiss索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings, dtype=np.float32))
        
        # 动态提取关键字
        self._extract_keywords()
        
    def _extract_keywords(self):
        """从文档中动态提取关键字"""
        self.keywords = set()
        # 收集所有文档内容
        all_content = " ".join([doc.content for doc in self.documents])
        
        # 提取可能的关键字（这里可以根据需要扩展）
        # 例如提取所有英文单词
        english_words = re.findall(r'\b[A-Za-z]+\b', all_content)
        for word in english_words:
            # 只保留长度大于2的单词作为关键字
            if len(word) > 2:
                self.keywords.add(word.lower())
        
        # 可以添加更多关键字提取逻辑
        # 例如特定的技术术语、产品名称等
        tech_terms = ['codex', 'cursor', 'copilot', 'ai', 'programming', 'tool', 'github']
        for term in tech_terms:
            self.keywords.add(term)
            
        print(f"提取到关键字: {self.keywords}")
        
    def update_index(self):
        """更新索引（当有新文档时）"""
        processor = DocumentProcessor()
        processor.process_all_documents("videos")
        self.build_index(processor.documents)
        
    def optimize_query(self, query):
        """使用大模型优化用户查询"""
        print(f"开始优化查询: '{query}'")
        # 如果查询为空或很短，直接返回
        if not query or len(query.strip()) < 2:
            print("查询为空或太短，直接返回")
            return query
            
        # 对于简单的单个英文词查询，尝试进行大小写优化
        # 使用正则表达式分割查询，支持中英文混合
        # 分离英文单词和中文字符
        english_words = re.findall(r'[A-Za-z]+', query)
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', query)
        query_words = english_words + chinese_chars
        print(f"查询分词结果: {query_words}, 词数: {len(query_words)}")
        
        # 只有当确实是单个英文单词时，才进行大小写优化
        if len(english_words) == 1 and len(chinese_chars) == 0 and len(query_words) == 1:
            print("处理单个英文词查询")
            # 收集所有文档中的关键词
            all_content = " ".join([doc.content for doc in self.documents])
            # 如果查询词在内容中以不同大小写形式存在，则使用内容中的形式
            if query.lower() in all_content.lower():
                # 直接在所有内容中查找精确匹配（忽略大小写）
                # 使用正则表达式进行不区分大小写的搜索
                pattern = re.compile(re.escape(query), re.IGNORECASE)
                match = pattern.search(all_content)
                if match:
                    optimized = match.group()
                    print(f"查询优化: {query} -> {optimized}")
                    return optimized
        # 对于更复杂的查询，使用大模型进行优化
        # 打印调试信息
        word_count = len(query_words)
        # 检查查询中是否包含任何关键字
        contains_keyword = any(keyword in query.lower() for keyword in self.keywords)
        
        print(f"查询分析 - 词数: {word_count}, 包含关键字: {contains_keyword}")
        
        # 修改条件：对于包含关键字的查询，使用大模型优化
        condition1 = word_count >= 1
        condition2 = contains_keyword
        print(f"条件1 (word_count >= 1): {condition1}")
        print(f"条件2 (包含关键字): {condition2}")
        
        # 添加更详细的调试信息
        print(f"完整条件判断: condition1({condition1}) and condition2({condition2}) = {condition1 and condition2}")
        
        if condition1 and condition2:
            print(f"使用大模型优化查询: {query}")
            optimization_prompt = f"""你是一个智能查询优化助手。请优化以下用户查询，使其更适合在文档中搜索相关信息。

原始查询："{query}"

请根据以下规则优化：
1. 保持查询的核心意图
2. 添加相关的术语或上下文词汇
3. 纠正可能的拼写错误
4. 使查询更具体和准确
5. 只返回优化后的查询，不要添加其他内容

优化后的查询："""

            try:
                response = ollama.generate(
                    model='qwen3:30b',
                    prompt=optimization_prompt,
                    options={
                        'temperature': 0.3,
                        'top_p': 0.9,
                    }
                )
                optimized_query = response['response'].strip()
                print(f"查询优化: {query} -> {optimized_query}")
                return optimized_query
            except Exception as e:
                print(f"查询优化失败: {e}")
                # 如果大模型优化失败，返回原始查询
                return query
        else:
            print("不满足使用大模型优化的条件")
        
        return query
        
    def search(self, query, k=3, threshold=0.02):
        """搜索相关文档"""
        if not self.index or not self.documents:
            return []
            
        # 优化查询
        optimized_query = self.optimize_query(query)
        print(f"原始查询: {query}, 优化后查询: {optimized_query}")
        
        # 生成查询向量
        query_vector = self.model.encode([optimized_query])
        
        # 搜索相似文档
        k_value = min(k, len(self.documents))
        distances, indices = self.index.search(np.array(query_vector, dtype=np.float32), k_value)
        
        # 过滤低相似度的结果
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            # 转换距离为相似度分数（L2距离越小越相似）
            similarity = 1 / (1 + distance)
            if similarity >= threshold and idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    'content': doc.content,
                    'video_url': doc.video_url,
                    'similarity': float(similarity),
                    'filename': os.path.basename(doc.file_path)
                })
        
        return results
    
    def generate_summary(self, query, search_results):
        """使用大模型生成总结"""
        # 如果没有找到相关结果，返回预设的回复
        if not search_results:
            return "抱歉，未找到与您的查询相关的内容。"
        
        # 检查相似度是否足够高
        max_similarity = max(result['similarity'] for result in search_results)
        if max_similarity < 0.02:  # 设置一个合理的相似度阈值
            return "抱歉，未找到与您的查询相关的内容。"
        
        # 构建提示词
        context = ""
        for i, result in enumerate(search_results, 1):
            context += f"内容：{result['content']}\n视频地址：{result['video_url']}\n\n"
        
        prompt = f"""你是视频内容助手，请根据以下内容回答用户的问题："{query}"

{context}
请严格按照以下规则回答：
1. 如果上述内容中有与用户问题相关的信息，请直接总结相关内容并提供视频地址，不要提及这是第几个结果
2. 如果上述内容中没有与用户问题相关的信息，请直接回复：抱歉，未找到与您的查询相关的内容。
3. 不要编造或推测与内容无关的信息
4. 回答要简洁明了，重点突出
5. 不要使用"与XX相关的视频是相关视频内容N"这类机械式表述
6. 直接以核心信息开始回答，不要添加"根据您的查询"等前缀
7. 如果有多个相关内容，请综合总结，不要分别列出
8. 当用户查询特定工具时，即使查询中包含通用词，也请提供该特定工具的相关信息"""

        try:
            # 调用Ollama大模型
            response = ollama.generate(
                model='qwen3:30b',
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                }
            )
            
            # 提取生成的文本
            summary = response['response'].strip()
            
            # 清理可能的多余内容
            # 移除可能的"根据您的查询"前缀
            summary = re.sub(r'^根据您的查询[^\n]*\n*', '', summary)
            
            return summary
            
        except Exception as e:
            print(f"大模型调用出错: {e}")
            return "抱歉，处理您的请求时出现错误。"