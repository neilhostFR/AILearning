import os
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from urllib.parse import quote_plus

# 构建mysql数据库链接
db_user = os.getenv("LOCAL_MYSQL_USER")
db_password = os.getenv("LOCAL_MYSQL_PWD")
db_host = "localhost:3306"
db_name = "mes_system"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{quote_plus(db_password)}@{db_host}/{db_name}")

# 构建llm
api_key=os.getenv("DASHSCOPE_API_KEY")
llm = ChatOpenAI(
    temperature=0.01,
    model="deepseek-v3",
    openai_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key  = api_key
)

# 构建工具类
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# 构建智能体
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

# tasks
# agent_executor.run("描述与生产工单表相关的表及其关系")
agent_executor.run("统计每个产品的合格率")