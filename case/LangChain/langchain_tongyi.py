import os
from langchain_community.chat_models import ChatTongyi
from langchain_community.llms import Tongyi
from langchain_core.messages import HumanMessage,SystemMessage
from typing import List
#pip install dashscope

dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")

def use_chattongyi(message:List,mode_name:str="deepseek-v3"):
	if not message:
		return '缺少提示词'
	llm=ChatTongyi(
		model_name=mode_name,
		dashscope_api_key=dashscope_api_key
	)

	response=llm.invoke(message) #message=[HumanMessage, SystemMessage, AIMessage]

	return response.content

def use_tongyi(message:str,mode_name:str="deepseek-v3"):
	if not message:
		return '缺少提示词'

	llm=Tongyi(
		model_name=mode_name,
		dashscope_api_key=dashscope_api_key
	)

	response=llm.invoke("请解释机器学习")

	return response


if __name__=="__main__":
	chattongyi_message=[
		SystemMessage('你是一个有帮助的助手'),
		HumanMessage(content="请解释机器学习")
	]
	responseContent=use_chattongyi(message=chattongyi_message)
	print(responseContent)


	response=use_tongyi('请解释机器学习')
	print(response)