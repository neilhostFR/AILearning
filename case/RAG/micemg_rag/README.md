# RAG问答系统

基于检索增强生成（Retrieval-Augmented Generation, RAG）技术的智能问答系统，能够根据知识库内容回答用户问题。

## 项目结构

```
micemg_rag/
├── backend/                 # 后端服务
│   ├── app.py              # Flask API服务
│   ├── knowledge_base.py   # 知识库管理模块
│   ├── rag_ask.py          # RAG问答核心模块
│   ├── knowledges_base/    # 知识库原始数据（Excel格式）
│   ├── chroma/             # 向量数据库
│   └── json_data.json      # 知识库JSON数据
├── frontend/               # 前端页面
│   └── index.html          # 问答界面
└── README.md               # 项目说明文档
```

## 技术栈

- **后端**: Python + Flask
- **RAG框架**: LangChain
- **向量数据库**: Chroma
- **嵌入模型**: HuggingFace embeddings (Qwen/Qwen3-Embedding-0.6B)
- **大语言模型**: Ollama (qwen3:30b)
- **前端**: HTML + CSS + JavaScript

## 功能特点

1. **智能问答**: 基于知识库内容回答用户问题
2. **前后端分离**: RESTful API架构，支持Web界面交互
3. **向量检索**: 使用Chroma向量数据库实现高效相似度检索
4. **多轮对话**: 支持对话历史记录管理
5. **来源追踪**: 显示答案相关的知识库文件路径

## 环境要求

- Python 3.11+
- Ollama服务（需安装qwen3:30b模型）
- 相关Python依赖包

## 安装部署

### 1. 安装依赖

```bash
# 安装Python依赖
pip install -r requirements.txt

# 启动Ollama服务并拉取模型
ollama run qwen3:30b
```

### 2. 启动服务

```bash
# 启动后端API服务
cd backend
python app.py

# 启动前端页面服务
cd frontend
python -m http.server 8000
```

### 3. 访问应用

- 前端页面: http://localhost:8000
- 后端API: http://localhost:5001

## API接口

### 问答接口

```
POST /api/query
Content-Type: application/json

{
  "question": "您的问题"
}

响应:
{
  "answer": "回答内容",
  "sources": ["相关文件路径"]
}
```

### 健康检查

```
GET /api/health

响应:
{
  "status": "healthy",
  "message": "RAG问答系统运行正常"
}
```

## 使用说明

1. 在前端页面输入框中输入问题
2. 点击"发送"按钮或按回车键提交
3. 系统将显示基于知识库内容的回答
4. 可以进行多轮对话交互

## 知识库管理

知识库数据存储在`backend/knowledges_base/`目录下的Excel文件中，系统会自动读取并构建向量数据库。

## 注意事项

1. 首次运行需要下载嵌入模型和大语言模型，可能需要较长时间
2. 确保Ollama服务正常运行
3. 如需更新知识库，重启后端服务即可自动重新加载