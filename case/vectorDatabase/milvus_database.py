# pip install pymilvus langchain-milvus
# Milvus文档 https://milvus.io/docs/zh
# 当前代码使用 Milvus Distributed部署版本
import os
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_milvus import Milvus
import numpy as np
from typing import List, Dict, Any, Optional

class MilvusVectorManager:
	def __init__(self,host:str="localhost",port:str="19530",collection_name:str="document_vectors"):
		self.host=host
		self.port=port
		self.collection_name=collection_name
		self.collection=None
		self.embedding_function=None
	def connect(self):
		try:
			connections.connect(alias="default",host=self.host,port=self.port)
			print(f"连接成功 Milvus:{self.host}:{self.port}")
			return True
		except:
			return False
	def initialize_embedding_function(self,model_name:str="BAAI/bge-large-zh-v1.5",device:str="mps",normalize_embeddings:bool=True):
		model_kwargs={"device":device}
		encode_kwargs={'normalize_embeddings': normalize_embeddings}

		self.embedding_function=HuggingFaceEmbeddings(
			model_name=model_name,
			model_kwargs=model_kwargs,
			encode_kwargs=encode_kwargs
		)
        
        # 测试嵌入维度
        test_embedding = self.embedding_function.embed_documents(["test"])
        self.dimension = len(test_embedding[0])
        print(f"嵌入模型初始化完成，维度: {self.dimension}")
        
        return self.dimension
    
    def create_collection(self, overwrite: bool = False):
        """创建集合"""
        if not self.connect():
            return False
            
        # 检查集合是否已存在
        if utility.has_collection(self.collection_name):
            if overwrite:
                utility.drop_collection(self.collection_name)
                print(f"已删除现有集合: {self.collection_name}")
            else:
                print(f"集合已存在: {self.collection_name}")
                self.collection = Collection(self.collection_name)
                return True
        
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="created_time", dtype=DataType.INT64),
        ]
        
        # 创建集合架构
        schema = CollectionSchema(fields, description="文档向量存储")
        
        # 创建集合
        self.collection = Collection(name=self.collection_name, schema=schema)
        
        # 创建索引
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        
        self.collection.create_index("embedding", index_params)
        print(f"创建集合成功: {self.collection_name}")
        return True
    
    def insert_documents(self, documents: List[Any], batch_size: int = 100):
        """插入文档"""
        if not self.collection:
            print("请先创建集合")
            return False
        
        total_docs = len(documents)
        inserted_count = 0
        
        for i in range(0, total_docs, batch_size):
            batch_docs = documents[i:i + batch_size]
            texts = [doc.page_content for doc in batch_docs]
            
            # 生成嵌入向量
            embeddings = self.embedding_function.embed_documents(texts)
            
            # 准备插入数据
            entities = []
            for j, doc in enumerate(batch_docs):
                entities.append([
                    doc.page_content,  # text
                    embeddings[j],     # embedding
                    str(doc.metadata), # metadata
                    doc.metadata.get('source', 'unknown'),  # source
                    int(os.times().elapsed)  # created_time
                ])
            
            # 插入数据
            insert_result = self.collection.insert(entities)
            inserted_count += len(batch_docs)
            print(f"已插入 {inserted_count}/{total_docs} 个文档")
        
        # 刷新数据
        self.collection.flush()
        print(f"成功插入 {inserted_count} 个文档")
        return inserted_count
    
    def search_similar(self, 
                      query: str, 
                      k: int = 5,
                      filters: Optional[Dict] = None) -> List[Dict]:
        """搜索相似文档"""
        if not self.collection:
            print("请先创建集合")
            return []
        
        # 加载集合到内存
        self.collection.load()
        
        # 生成查询向量
        query_embedding = self.embedding_function.embed_query(query)
        
        # 构建搜索参数
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        # 构建过滤表达式
        expr = None
        if filters:
            filter_parts = []
            for key, value in filters.items():
                if isinstance(value, str):
                    filter_parts.append(f'{key} == "{value}"')
                else:
                    filter_parts.append(f'{key} == {value}')
            expr = " and ".join(filter_parts)
        
        # 执行搜索
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=k,
            expr=expr,
            output_fields=["id", "text", "metadata", "source"]
        )
        
        # 格式化结果
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.id,
                    "text": hit.entity.get("text"),
                    "metadata": eval(hit.entity.get("metadata", "{}")),
                    "source": hit.entity.get("source"),
                    "score": hit.score,
                    "distance": hit.distance
                })
        
        return formatted_results
    
    def update_document(self, doc_id: int, new_text: str, new_metadata: Dict = None):
        """更新文档"""
        if not self.collection:
            print("请先创建集合")
            return False
        
        # 生成新的嵌入向量
        new_embedding = self.embedding_function.embed_documents([new_text])[0]
        
        # 准备更新数据
        update_data = {
            "text": new_text,
            "embedding": new_embedding,
            "metadata": str(new_metadata) if new_metadata else "{}"
        }
        
        try:
            # Milvus 2.x 使用 upsert 进行更新
            entities = [[
                new_text,
                new_embedding,
                str(new_metadata) if new_metadata else "{}",
                "updated",  # source
                int(os.times().elapsed)  # created_time
            ]]
            
            # 先删除旧文档，再插入新文档（模拟更新）
            self.delete_documents([doc_id])
            self.collection.insert(entities)
            self.collection.flush()
            
            print(f"成功更新文档 ID: {doc_id}")
            return True
            
        except Exception as e:
            print(f"更新文档失败: {e}")
            return False
    
    def delete_documents(self, doc_ids: List[int]):
        """删除文档"""
        if not self.collection:
            print("请先创建集合")
            return False
        
        try:
            # 构建删除表达式
            ids_str = ", ".join(map(str, doc_ids))
            expr = f"id in [{ids_str}]"
            
            # 执行删除
            result = self.collection.delete(expr)
            self.collection.flush()
            
            print(f"成功删除 {len(doc_ids)} 个文档")
            return True
            
        except Exception as e:
            print(f"删除文档失败: {e}")
            return False
    
    def get_collection_stats(self):
        """获取集合统计信息"""
        if not self.collection:
            print("请先创建集合")
            return None
        
        stats = {
            "collection_name": self.collection_name,
            "num_entities": self.collection.num_entities,
            "is_empty": self.collection.is_empty
        }
        
        print(f"集合统计:")
        print(f"  - 名称: {stats['collection_name']}")
        print(f"  - 文档数量: {stats['num_entities']}")
        print(f"  - 是否为空: {stats['is_empty']}")
        
        return stats
    
    def list_collections(self):
        """列出所有集合"""
        if not self.connect():
            return []
        
        collections = utility.list_collections()
        print("所有集合:")
        for col in collections:
            print(f"  - {col}")
        
        return collections
    
    def drop_collection(self):
        """删除集合"""
        if not self.connect():
            return False
        
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"成功删除集合: {self.collection_name}")
            return True
        else:
            print(f"集合不存在: {self.collection_name}")
            return False
    
    def backup_collection(self, backup_path: str):
        """备份集合数据"""
        if not self.collection:
            print("请先创建集合")
            return False
        
        try:
            # 查询所有数据
            self.collection.load()
            results = self.collection.query(
                expr="id >= 0",
                output_fields=["id", "text", "metadata", "source"]
            )
            
            # 保存到文件
            import json
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"备份成功，保存到: {backup_path}")
            print(f"备份了 {len(results)} 个文档")
            return True
            
        except Exception as e:
            print(f"备份失败: {e}")
            return False

# 使用 LangChain 的 Milvus 集成
class LangChainMilvusManager:
    def __init__(self, 
                 connection_args: Dict = None,
                 collection_name: str = "langchain_docs"):
        self.connection_args = connection_args or {
            "host": "localhost",
            "port": "19530"
        }
        self.collection_name = collection_name
        self.vectorstore = None
        
    def initialize(self, embedding_function, documents: List[Any] = None):
        """初始化 LangChain Milvus"""
        if documents:
            # 创建新的向量存储
            self.vectorstore = Milvus.from_documents(
                documents=documents,
                embedding=embedding_function,
                collection_name=self.collection_name,
                connection_args=self.connection_args
            )
            print(f"创建新的 Milvus 向量存储，包含 {len(documents)} 个文档")
        else:
            # 加载现有向量存储
            self.vectorstore = Milvus(
                embedding_function=embedding_function,
                collection_name=self.collection_name,
                connection_args=self.connection_args
            )
            print("加载现有 Milvus 向量存储")
        
        return self.vectorstore
    
    def add_documents(self, documents: List[Any]):
        """添加文档"""
        if not self.vectorstore:
            print("请先初始化向量存储")
            return False
        
        self.vectorstore.add_documents(documents)
        print(f"成功添加 {len(documents)} 个文档")
        return True
    
    def similarity_search(self, query: str, k: int = 5, **kwargs):
        """相似度搜索"""
        if not self.vectorstore:
            print("请先初始化向量存储")
            return []
        
        results = self.vectorstore.similarity_search(query, k=k, **kwargs)
        return results
    
    def delete_documents(self, ids: List[str]):
        """删除文档（通过 ID）"""
        # 注意：LangChain Milvus 的删除功能可能有限
        # 通常需要直接使用 pymilvus
        print("LangChain Milvus 删除功能有限，建议使用原生 pymilvus")
        return False


def main():
    """Milvus 完整功能演示"""
    
    # 1. 初始化管理器
    print("=== 1. 初始化 Milvus 管理器 ===")
    manager = MilvusVectorManager(
        host="localhost",  # 根据你的 Milvus 配置修改
        port="19530",
        collection_name="three_kingdoms"
    )
    
    # 2. 初始化嵌入函数
    print("\n=== 2. 初始化嵌入函数 ===")
    dimension = manager.initialize_embedding_function(
        model_name="BAAI/bge-large-zh-v1.5",
        device="cpu",  # 如果有 GPU 可以改为 "cuda"
        normalize_embeddings=True
    )
    
    # 3. 创建集合
    print("\n=== 3. 创建集合 ===")
    manager.create_collection(overwrite=True)  # 覆盖现有集合
    
    # 4. 加载和处理文档
    print("\n=== 4. 加载和处理文档 ===")
    try:
        documents = get_documents('../Embedding/source/three_kingdoms.txt')
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"加载了 {len(split_docs)} 个文档块")
    except Exception as e:
        print(f"文档加载失败: {e}")
        # 使用示例文档
        from langchain_core.documents import Document
        split_docs = [
            Document(page_content="刘备字玄德，中山靖王刘胜之后", metadata={"source": "sample"}),
            Document(page_content="关羽字云长，河东解良人也", metadata={"source": "sample"}),
            Document(page_content="张飞字翼德，涿郡人也", metadata={"source": "sample"})
        ]
        print(f"使用示例文档: {len(split_docs)} 个")
    
    # 5. 插入文档
    print("\n=== 5. 插入文档 ===")
    inserted_count = manager.insert_documents(split_docs, batch_size=50)
    
    # 6. 获取统计信息
    print("\n=== 6. 集合统计 ===")
    stats = manager.get_collection_stats()
    
    # 7. 搜索演示
    print("\n=== 7. 搜索演示 ===")
    test_queries = ["刘备", "关羽", "曹操", "诸葛亮"]
    
    for query in test_queries:
        print(f"\n🔍 搜索: '{query}'")
        results = manager.search_similar(query, k=3)
        
        for i, result in enumerate(results):
            print(f"  {i+1}. ID: {result['id']}, 分数: {result['score']:.4f}")
            print(f"     内容: {result['text'][:80]}...")
            print(f"     来源: {result['source']}")
    
    # 8. 更新文档演示
    print("\n=== 8. 更新文档演示 ===")
    if results:
        first_doc_id = results[0]['id']
        print(f"🔄 更新文档 ID: {first_doc_id}")
        manager.update_document(
            first_doc_id, 
            "刘备字玄德，中山靖王刘胜之后，汉景帝阁下玄孙",  # 新内容
            {"source": "updated", "version": "2.0"}
        )
    
    # 9. 删除文档演示
    print("\n=== 9. 删除文档演示 ===")
    if len(results) >= 2:
        delete_ids = [results[1]['id']]
        print(f"🗑️ 删除文档 ID: {delete_ids}")
        manager.delete_documents(delete_ids)
    
    # 10. 最终统计
    print("\n=== 10. 最终统计 ===")
    manager.get_collection_stats()
    
    # 11. 备份演示
    print("\n=== 11. 备份数据 ===")
    manager.backup_collection("./milvus_backup.json")
    
    # 12. 使用 LangChain 集成版本
    print("\n=== 12. LangChain Milvus 集成演示 ===")
    lc_manager = LangChainMilvusManager(collection_name="langchain_docs")
    lc_vectorstore = lc_manager.initialize(manager.embedding_function, split_docs[:10])
    
    # LangChain 搜索
    lc_results = lc_manager.similarity_search("三国英雄", k=2)
    for i, doc in enumerate(lc_results):
        print(f"  {i+1}. {doc.page_content[:80]}...")
        print(f"     元数据: {doc.metadata}")

if __name__ == "__main__":
    main()
