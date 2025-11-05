from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, Depends, Cookie
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import shutil
from typing import List, Optional
import aiofiles
import ollama
import asyncio
import uuid
from datetime import datetime, timedelta, timezone
import mysql.connector
import bcrypt

from document_processor import DocumentProcessor
from rag_engine import RAGEngine
from database import DatabaseManager

app = FastAPI(title="Video RAG System")

# 初始化文档处理器和RAG引擎
doc_processor = DocumentProcessor()
rag_engine = RAGEngine()
db_manager = DatabaseManager()

# 初始化模板引擎
templates = Jinja2Templates(directory="templates")

# 确保videos目录存在
VIDEO_DIR = "videos"
if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)

# 初始化数据库
@app.on_event("startup")
async def startup_event():
    # 初始化数据库表
    db_manager.init_database()
    
    # 启动时处理所有现有文档
    doc_processor.process_all_documents(VIDEO_DIR)
    rag_engine.build_index(doc_processor.documents)

# 依赖项：检查管理员会话
async def get_current_admin(session_token: Optional[str] = Cookie(None)):
    if not session_token:
        raise HTTPException(status_code=401, detail="未登录")
    
    admin_id = db_manager.verify_session(session_token)
    if not admin_id:
        raise HTTPException(status_code=401, detail="会话已过期或无效")
    
    return admin_id

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 管理员登录页面
@app.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page(request: Request):
    return templates.TemplateResponse("admin_login.html", {"request": request})

# 管理员登录处理
@app.post("/admin/login")
async def admin_login(request: Request):
    from fastapi.responses import JSONResponse
    from datetime import timezone
    
    data = await request.json()
    username = data.get("username")
    password = data.get("password")
    
    if not username or not password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")
    
    # 验证管理员
    admin_id = db_manager.verify_admin(username, password)
    if not admin_id:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    
    # 创建会话
    session_token = str(uuid.uuid4())
    expires_at = datetime.now(timezone.utc) + timedelta(hours=24)  # 24小时过期
    
    if not db_manager.create_session(admin_id, session_token, expires_at):
        raise HTTPException(status_code=500, detail="创建会话失败")
    
    # 设置会话cookie
    response = JSONResponse({"message": "登录成功"})
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        max_age=24*60*60,  # 24小时
        expires=expires_at
    )
    
    return response

# 管理员后台页面
@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(request: Request, admin_id: int = Depends(get_current_admin)):
    return templates.TemplateResponse("admin_dashboard.html", {"request": request})

# 管理员退出登录
@app.post("/admin/logout")
async def admin_logout(session_token: Optional[str] = Cookie(None)):
    from fastapi.responses import JSONResponse
    
    if session_token:
        # 这里可以添加从数据库删除会话的逻辑
        pass
    
    response = JSONResponse({"message": "退出登录成功"})
    response.delete_cookie("session_token")
    return response

# 获取文档列表
@app.get("/admin/documents")
async def get_documents(admin_id: int = Depends(get_current_admin)):
    """获取所有文档列表"""
    documents = []
    if os.path.exists(VIDEO_DIR):
        for filename in os.listdir(VIDEO_DIR):
            if filename.endswith('.txt'):
                file_path = os.path.join(VIDEO_DIR, filename)
                documents.append({
                    "name": filename,
                    "path": file_path,
                    "size": os.path.getsize(file_path)
                })
    return documents

# 管理员上传文件
@app.post("/admin/upload")
async def admin_upload_file(file: UploadFile = File(...), admin_id: int = Depends(get_current_admin)):
    # 检查文件扩展名
    if not file.filename or not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="只允许上传.txt文件")
    
    # 保存文件到videos目录
    file_path = os.path.join(VIDEO_DIR, file.filename) if file.filename else os.path.join(VIDEO_DIR, "unnamed.txt")
    
    # 异步保存文件
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    
    # 处理新上传的文档
    doc_processor.process_document(file_path)
    rag_engine.update_index()
    
    return {"filename": file.filename, "message": "文件上传并处理成功"}

# 管理员修改密码
@app.post("/admin/change-password")
async def change_password(request: Request, admin_id: int = Depends(get_current_admin)):
    data = await request.json()
    current_password = data.get("currentPassword")
    new_password = data.get("newPassword")
    
    if not current_password or not new_password:
        raise HTTPException(status_code=400, detail="当前密码和新密码不能为空")
    
    # 验证当前密码是否正确
    # 我们需要直接查询数据库来验证当前密码
    connection = db_manager.get_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="数据库连接失败")
    
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT password FROM admins WHERE id = %s",
            (admin_id,)
        )
        result = cursor.fetchone()
        
        if not result:
            cursor.close()
            connection.close()
            raise HTTPException(status_code=404, detail="管理员不存在")
        
        hashed_password = result[0]
        
        # 验证当前密码
        # 确保hashed_password是bytes类型
        if isinstance(hashed_password, str):
            hashed_password_bytes = hashed_password.encode('utf-8')
        else:
            hashed_password_bytes = hashed_password
            
        # 验证密码
        if not bcrypt.checkpw(current_password.encode('utf-8'), hashed_password_bytes):
            cursor.close()
            connection.close()
            raise HTTPException(status_code=400, detail="当前密码错误")
        
        # 加密新密码
        new_hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        
        # 更新密码
        cursor.execute(
            "UPDATE admins SET password = %s WHERE id = %s",
            (new_hashed_password, admin_id)
        )
        
        connection.commit()
        cursor.close()
        connection.close()
        
        return {"message": "密码修改成功"}
    except HTTPException:
        # 重新抛出已知的HTTP异常
        raise
    except Exception as e:
        print(f"密码修改失败: {e}")
        if connection:
            connection.rollback()
            connection.close()
        raise HTTPException(status_code=500, detail="密码修改失败")

@app.post("/query")
async def query_documents(query: dict):
    user_query = query.get("query", "")
    if not user_query:
        return {"results": []}
    
    # 使用RAG引擎搜索相关文档
    results = rag_engine.search(user_query)
    
    # 使用大模型生成总结
    summary = rag_engine.generate_summary(user_query, results)
    
    return {"summary": summary, "results": results}

@app.post("/query-stream")
async def query_documents_stream(query: dict):
    from fastapi.responses import StreamingResponse
    import asyncio
    
    user_query = query.get("query", "")
    if not user_query:
        async def empty_stream():
            yield "data: 抱歉，未找到与您的查询相关的内容。\n\n"
        return StreamingResponse(empty_stream(), media_type="text/event-stream")
    
    # 使用RAG引擎搜索相关文档
    results = rag_engine.search(user_query)
    
    # 如果没有找到相关结果，返回预设的回复
    if not results:
        async def no_results_stream():
            yield "data: 抱歉，未找到与您的查询相关的内容。\n\n"
        return StreamingResponse(no_results_stream(), media_type="text/event-stream")
    
    # 检查相似度是否足够高
    max_similarity = max(result['similarity'] for result in results)
    if max_similarity < 0.02:  # 设置一个合理的相似度阈值
        async def low_similarity_stream():
            yield "data: 抱歉，未找到与您的查询相关的内容。\n\n"
        return StreamingResponse(low_similarity_stream(), media_type="text/event-stream")
    
    # 构建提示词
    context = ""
    for i, result in enumerate(results, 1):
        context += f"内容：{result['content']}\n视频地址：{result['video_url']}\n\n"
    
    prompt = f"""你是视频内容助手，请根据以下内容回答用户的问题："{user_query}"

{context}
请严格按照以下规则回答：
1. 如果上述内容中有与用户问题相关的信息，请直接总结相关内容并提供视频地址，不要提及这是第几个结果
2. 如果上述内容中没有与用户问题相关的信息，请直接回复：抱歉，未找到与您的查询相关的内容。
3. 不要编造或推测与内容无关的信息
4. 回答要简洁明了，重点突出
5. 不要使用"与XX相关的视频是相关视频内容N"这类机械式表述
6. 直接以核心信息开始回答，不要添加"根据您的查询"等前缀
7. 如果有多个相关内容，请综合总结，不要分别列出
8. 当用户查询特定工具时，即使查询中包含通用词，也请提供该特定工具的相关信息"""

    async def generate_summary():
        try:
            # 调用Ollama大模型进行流式输出
            response = ollama.generate(
                model='qwen3:30b',
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                },
                stream=True  # 启用流式输出
            )
            
            # 流式返回生成的文本
            for chunk in response:
                if 'response' in chunk:
                    content = chunk['response']
                    yield f"data: {content}\n\n"
                    # 添加短暂延迟以模拟流式效果
                    await asyncio.sleep(0.01)
        except Exception as e:
            print(f"大模型调用出错: {e}")
            yield "data: 抱歉，处理您的请求时出现错误。\n\n"
    
    return StreamingResponse(generate_summary(), media_type="text/event-stream")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # 检查文件扩展名
    if not file.filename or not file.filename.endswith('.txt'):
        return {"error": "只允许上传.txt文件"}
    
    # 保存文件到videos目录
    file_path = os.path.join(VIDEO_DIR, file.filename) if file.filename else os.path.join(VIDEO_DIR, "unnamed.txt")
    
    # 异步保存文件
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    
    # 处理新上传的文档
    doc_processor.process_document(file_path)
    rag_engine.update_index()
    
    return {"filename": file.filename, "message": "文件上传并处理成功"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)