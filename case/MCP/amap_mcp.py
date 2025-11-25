"""高德mcp"""
import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

async def init_amap_tools():
	amap_key=os.getenv("AMAP_MAPS_API_KEY")
	mcp_config = {
        "amap-maps-sse": {
            "url": f"https://mcp.amap.com/sse?key={amap_key}", 
            "transport": "sse"
        }
    }

	# amap_server_env={
	# 	"AMAP_MAPS_API_KEY":os.getenv("AMAP_MAPS_API_KEY")
	# }

	client=MultiServerMCPClient(mcp_config)
	tools=await client.get_tools()
	return tools

async def main(agent,question):

	async for step in agent.astream(
		{'messages': question},
		stream_mode="values"
	):
		step["messages"][-1].pretty_print()

if __name__ =="__main__":

	tools=asyncio.run(init_amap_tools())

	for tool in tools:
		print(f"工具名称:{tool.name}")
		print(f"工具描述:{tool.description}\n")

	llm=ChatOllama(model="qwen3:30b")

	agent=create_agent(model=llm,tools=tools,system_prompt="你是一个智能助手")

	# response=asyncio.run(
	# 	agent.ainvoke({
	# 		"messages":[
	# 			{
	# 				"role":"user",
	# 				"content":"帮我规划一个从北京天安门到上海外滩的3日游路线。"
	# 			}
	# 		]
	# 	})
	# )
	# print(response)

	asyncio.run(main(agent,"上海今天天气怎么样？"))