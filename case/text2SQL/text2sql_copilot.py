import os
import pymysql
from urllib.parse import quote_plus
from openai import OpenAI


def get_database_schema(conn_dict):
	"""通过sql链接，获取对应数据库的建表语句"""
	conn=pymysql.connect(
		host=conn_dict["host"],
		port=conn_dict["port"],
		user=conn_dict["user"],
		passwd=conn_dict["password"],
		db=conn_dict["db"],
		charset=conn_dict["charset"]
	)
	cursor=conn.cursor()

	sql_tables=f"SHOW TABLES"
	cursor.execute(sql_tables)
	table_list=[table[0] for table in cursor.fetchall()]

	database_schemas={}
	for table_name in table_list:
		sql_table=f"SHOW CREATE TABLE {table_name}"
		cursor.execute(sql_table)
		table_detail=cursor.fetchone()
		database_schemas[table_name]=table_detail[1]

	cursor.close()
	conn.close()
	return database_schemas

def get_response(messages):
	client=OpenAI(
		base_url="http://localhost:11434/v1",
		api_key="ollama"
	)

	response=client.chat.completions.create(
		model="qwen3:30b",
		messages=messages,
		temperature=0.01
	)

	return response

def get_response_dashscope(messages):
	client=OpenAI(
		api_key=os.getenv("DASHSCOPE_API_KEY"),
		base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
	)
	response=client.chat.completions.create(
		model='qwen-turbo-latest',
		messages=messages,
		temperature=0.01
	)

	return response

def get_sql(query,table_description):
	sys_prompt="""我正在编写SQL，以下是数据库中的数据表和字段，请思考：哪些数据表和字段是该SQL需要的，然后编写对应的SQL，如果有多个查询语句，请尝试合并为一个。如果在查询中用到新的表头，请用中文表示。编写SQL请采用```sql"""
	user_prompt=f"""数据表和字段如下
=====
{table_description}
=====
我要写的SQL是：{query}
请思考：哪些数据表和字段是该SQL需要的，然后编写对应的SQL
"""
	messages = [
		{"role": "system", "content": sys_prompt},
		{"role": "user", "content": user_prompt}
    ]
	print(user_prompt)
	response = get_response_dashscope(messages)
	return response

if __name__=="__main__":
	conn_dict = {'host': 'localhost', 'port': 3306, 'user': os.getenv("LOCAL_MYSQL_USER"), 'password': os.getenv("LOCAL_MYSQL_PWD"), 'db': 'mes_system', 'charset': 'utf8'}
	database_schemas=get_database_schema(conn_dict)

	table_descriptions=[]
	for table_name in database_schemas:
		table_descriptions.append(database_schemas[table_name])

	# print("\n".join(table_descriptions))

	response=get_sql("统计每个产品的合格率","\n".join(table_descriptions))

	print(response.choices[0].message.content)


