import os
import re

class Document:
    def __init__(self, file_path, content, video_url):
        self.file_path = file_path
        self.content = content
        self.video_url = video_url

class DocumentProcessor:
    def __init__(self):
        self.documents = []
    
    def process_all_documents(self, directory):
        """处理目录中的所有文档"""
        self.documents = []
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory, filename)
                self.process_document(file_path)
    
    def process_document(self, file_path):
        """处理单个文档"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                
                # 提取视频URL（假设URL在最后一行，以"http"开头）
                lines = content.split('\n')
                video_url = ""
                main_content = content
                
                if lines:
                    last_line = lines[-1]
                    # 查找可能的URL模式
                    url_match = re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', last_line)
                    if url_match:
                        video_url = url_match.group()
                        # 移除最后一行中的URL，保留主要内容
                        main_content = '\n'.join(lines[:-1])
                
                doc = Document(file_path, main_content, video_url)
                # 如果是新文档，添加到列表中；如果是更新的文档，替换它
                self._add_or_update_document(doc)
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    def _add_or_update_document(self, new_doc):
        """添加或更新文档"""
        # 检查是否已存在相同路径的文档
        for i, doc in enumerate(self.documents):
            if doc.file_path == new_doc.file_path:
                self.documents[i] = new_doc
                return
        # 如果不存在，则添加新文档
        self.documents.append(new_doc)