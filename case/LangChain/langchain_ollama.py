import langchain
print(langchain.__version__)
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage,SystemMessage
from typing import List

def use_chatollama(message:List,model_name:str="qwen3:30b",temperature:float=0.7):
	if not message:
		return "缺少提示词"

	llm=ChatOllama(
		model=model_name,
		base_url="http://localhost:11434",
		temperature=temperature
	)

	response=llm.invoke(message) #message=[HumanMessage, SystemMessage, AIMessage]

	return response.content

def use_ollamallm(message:str,model_name:str="qwen3:30b",temperature:float=0.7):
	if not message:
		return "缺少提示词"

	llm=Ollama(
		model=model_name,
		temperature=temperature
	)

	response=llm.invoke(message)

	return response



if __name__=="__main__":
	chatollama_message=[
		SystemMessage('你是一个有帮助的助手'),
		HumanMessage(content="请解释机器学习")
	]
	responseContent=use_chatollama(message=chatollama_message)
	print(responseContent)


	response=use_ollamallm('请解释机器学习')
	print(response)