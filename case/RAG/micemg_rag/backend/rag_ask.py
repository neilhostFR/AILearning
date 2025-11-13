# from langchain_community.chains import create_retrieval_chain
import langchain
from langchain_classic.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM
from knowledge_base import BulidKnowledge
from langchain_community.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def initialize_rag_system():
	"""初始化RAG系统"""
	print("正在初始化RAG系统...")
	knowledge=BulidKnowledge()

	ollama_llm=OllamaLLM(
		model="qwen3:30b"
	)

	# 使用ConversationBufferMemory来管理对话历史
	memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
	# 创建基础检索器
	base_retriever = knowledge.vectordb.as_retriever(search_kwargs={"k": 10})
	
	# 创建重排序器
	compressor = CrossEncoderReranker(model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3"), top_n=1)
	
	# 创建带重排序的检索器
	retriever = ContextualCompressionRetriever(
		base_compressor=compressor, 
		base_retriever=base_retriever
	)
	
	qa=ConversationalRetrievalChain.from_llm(
		llm=ollama_llm, 
		retriever=retriever,
		return_source_documents=True,
		memory=memory
	)
	
	return knowledge, qa

def query_knowledge_base(qa, query):
	"""查询知识库"""
	try:
		result=qa.invoke({"question":query})
		return result
	except Exception as e:
		print(f"处理问题时出错: {e}")
		return None

if __name__=="__main__":
	# 初始化系统
	knowledge, qa = initialize_rag_system()

	# 添加欢迎信息
	print("=" * 50)
	print("欢迎使用RAG问答系统！")
	print("输入'bye'、'exit'或'退出'结束对话。")
	print("=" * 50)
	
	while True:
		query=input("\n请输入你要查询的问题：")
		# 如果用户输入为空，继续下一次循环
		if not query.strip():
			continue

		if query.lower() in ['bye', 'exit', '退出']:
			print("感谢使用，再见！")
			break
		
		# 查询知识库
		result = query_knowledge_base(qa, query)
		if result:
			print("\n回答:", result["answer"])

			source=[]
			if "source_documents" in result:
				for i,doc in enumerate(result["source_documents"]):
					if "source" in doc.metadata:
						source.append(doc.metadata['source'])
			if source:
				print("\n相关文件路径:", ", ".join(source))