from openai import OpenAI

class QueryOptimize:
	"""查询优化助手，优化用户查询"""
	def __init__(self,model_name="qwen3:30b"):
		self.model_name=model_name

	def do_optimize(self,query)->str:
		"""对单个提问的优化"""
		prompt=f"""你是一个专业的迪士尼知识库查询助手，负责优化用户问题以匹配迪士尼知识库。
## 你的任务
对用户输入的问题进行优化和扩展，生成更适合迪士尼知识库检索的查询语句。

## 迪士尼知识库特点
- 内容包含：迪士尼门票规则，老人票定价规定，游玩攻略清单，迪士尼乐园酒店会员制度，活动海报

## 优化规则
1.**保持原意**：不改变用户问题的核心意图
2.**扩展同义词**：添加相关的技术术语、场景词汇
3.**多角度覆盖**：从不同角度重新表述问题
4.**具体化**：将模糊问题转化为具体可检索的查询

## 优化策略
- 如果是操作类问题，考虑添加步骤、方法、工具等关键词
- 如果是概念类问题，考虑添加定义、原理、应用场景等
- 如果是问题解决类，考虑添加原因、解决方案、排查步骤等

## 输出格式
请严格按照以下JSON格式输出：
{{
	"original_question": "原始问题",
	"optimized_queries": [
		{{
			"optimized_query":"优化后提示词",
			"confidence_level":"置信度[0-1]"
		}}
		...
	],
	"reasoning": "优化思路说明"
}}

现在请优化一下问题：
用户问题：{query}
"""
		client=OpenAI(
			base_url="http://localhost:11434/v1",
			api_key="ollama"
		)

		completion=client.chat.completions.create(
			model=self.model_name,
			messages=[
				{"role":"system","content":"你是一个查询优化助手"},
				{"role":"user","content":prompt}
			]
		)

		return completion.choices[0].message.content

	def do_optimize_history(self,messages):
		"""对会话历史的优化"""

		history_message=["\n".join([f"{message["role"]}:{message["content"]}" for message in messages if message["content"]])]
		user_query=messages[-1]["content"]
		prompt=f"""你是一个专业的迪士尼知识库检索助手，负责结合历史对话优化用户当前问题，以更好的匹配迪士尼知识库。
## 你的任务
分析用户当前的问题和历史对话上下文，生成更合适视频内容检索的优化查询语句

## 可用信息
- **当前问题**：{user_query}
- **历史对话**：
{history_message}

## 迪士尼知识库特点
- 内容包含：迪士尼门票规则，老人票定价规定，游玩攻略清单，迪士尼乐园酒店会员制度，活动海报

## 优化原则
1. **上下文理解**: 基于历史对话理解用户的真实需求和上下文
2. **指代消解**: 解析代词（它、这个、那个）和省略内容
3. **意图延续**: 识别对话的延续性和主题发展
4. **多角度覆盖**: 从不同角度重新表述问题
5. **具体化**: 将模糊问题转化为具体可检索的查询

## 优化策略
- 如果是对话的延续，补充前文提到的关键信息
- 解析代词和省略内容，还原完整问题
- 识别用户可能感兴趣的相关方向
- 考虑视频内容的结构特点

## 输出格式
请严格按照以下JSON格式输出：
{{
	"original_question":"原始问题",
	"context_understanding":"对上下文的理解分析",
	"optimized_queries": [
		{{
			"optimized_query":"优化后提示词",
			"confidence_level":"置信度[0-1]"
		}}
		...
	],
	"reasoning": "基于上下文的优化思路说明"
}}
"""

if __name__=="__main__":
	queryOptimize=QueryOptimize()

	new_query=queryOptimize.do_optimize("万圣节")

	print(new_query)