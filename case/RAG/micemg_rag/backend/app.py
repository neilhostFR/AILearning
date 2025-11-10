from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

from knowledge_base import BulidKnowledge
from rag_ask import initialize_rag_system, query_knowledge_base

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 初始化RAG系统
knowledge, qa = initialize_rag_system()

@app.route('/api/query', methods=['POST'])
def query():
    """处理用户查询请求"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': '问题不能为空'}), 400
            
        # 查询知识库
        result = query_knowledge_base(qa, question)
        
        if result:
            # 提取相关文件路径
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    if "source" in doc.metadata:
                        sources.append(doc.metadata['source'])
            
            return jsonify({
                'answer': result["answer"],
                'sources': sources
            })
        else:
            return jsonify({'error': '未找到相关答案'}), 404
            
    except Exception as e:
        return jsonify({'error': f'处理问题时出错: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({'status': 'healthy', 'message': 'RAG问答系统运行正常'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)