import os
from openai import OpenAI
client=OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
response=client.chat.completions.create(
	model="qwen-flash",
	messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    # stream=False
    extra_body={"enable_thinking": False}
)

print(response.choices[0].message.content)