import io
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path

# PDF处理相关
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF

# 图像处理相关
from PIL import Image
import pytesseract

# 嵌入模型相关
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import openai

# 向量数据库
import chromadb
from chromadb.config import Settings

class PDFMultiModalProcessor:
    """
    多模态PDF文档处理类
    支持文本、表格、图片内容的提取和embedding生成
    """
    
    def __init__(self, 
                 text_model_name: str = 'all-MiniLM-L6-v2',
                 clip_model_name: str = 'openai/clip-vit-base-patch32',
                 chroma_persist_dir: str = "./chroma_db",
                 ocr_languages: str = 'chi_sim+eng'):
        """
        初始化处理器
        
        Args:
            text_model_name: 文本embedding模型名称
            clip_model_name: 多模态CLIP模型名称  
            chroma_persist_dir: 向量数据库存储目录
            ocr_languages: OCR识别语言
        """
        self.text_model_name = text_model_name
        self.clip_model_name = clip_model_name
        self.chroma_persist_dir = chroma_persist_dir
        self.ocr_languages = ocr_languages
        
        # 初始化组件
        self._initialize_components()
        
        # 设置日志
        self._setup_logging()
    
    def _initialize_components(self):
        """初始化各个处理组件"""
        try:
            # 文本嵌入模型
            self.text_model = SentenceTransformer(self.text_model_name)
            self.logger.info(f"文本嵌入模型 {self.text_model_name} 加载成功")
        except Exception as e:
            self.logger.error(f"文本嵌入模型加载失败: {e}")
            raise
        
        try:
            # 多模态CLIP模型
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self.logger.info(f"CLIP模型 {self.clip_model_name} 加载成功")
        except Exception as e:
            self.logger.error(f"CLIP模型加载失败: {e}")
            raise
        
        # 初始化向量数据库客户端
        self._setup_vector_db()
    
    def _setup_logging(self):
        """设置日志"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _setup_vector_db(self):
        """设置向量数据库"""
        try:
            self.chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.chroma_persist_dir
            ))
            self.collection = self.chroma_client.create_collection("pdf_documents")
            self.logger.info("向量数据库初始化成功")
        except Exception as e:
            self.logger.error(f"向量数据库初始化失败: {e}")
            raise
    
    def extract_content(self, pdf_path: str) -> Dict[str, List]:
        """
        从PDF中提取所有内容
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            包含文本、表格、图片内容的字典
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        self.logger.info(f"开始提取PDF内容: {pdf_path}")
        
        content = {
            'text': [],
            'tables': [], 
            'images': []
        }
        
        try:
            # 提取文本内容
            content['text'] = self._extract_text_content(pdf_path)
            self.logger.info(f"提取到 {len(content['text'])} 个文本块")
            
            # 提取表格内容
            content['tables'] = self._extract_table_content(pdf_path)
            self.logger.info(f"提取到 {len(content['tables'])} 个表格")
            
            # 提取图片内容
            content['images'] = self._extract_image_content(pdf_path)
            self.logger.info(f"提取到 {len(content['images'])} 张图片")
            
        except Exception as e:
            self.logger.error(f"PDF内容提取失败: {e}")
            raise
        
        return content
    
    def _extract_text_content(self, pdf_path: Path) -> List[Dict]:
        """提取文本内容"""
        text_content = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    text_content.append({
                        'page': page_num + 1,
                        'type': 'text',
                        'content': text.strip(),
                        'bbox': page.bbox
                    })
        
        return text_content
    
    def _extract_table_content(self, pdf_path: Path) -> List[Dict]:
        """提取表格内容"""
        table_content = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                
                for table_num, table in enumerate(tables):
                    if table and any(any(cell for cell in row) for row in table):
                        try:
                            # 清理表格数据
                            cleaned_table = []
                            for row in table:
                                cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                                cleaned_table.append(cleaned_row)
                            
                            # 转换为DataFrame
                            headers = cleaned_table[0]
                            data = cleaned_table[1:]
                            df = pd.DataFrame(data, columns=headers)
                            
                            # 生成表格描述文本
                            table_text = self._table_to_text(df, headers)
                            
                            table_content.append({
                                'page': page_num + 1,
                                'type': 'table',
                                'content': table_text,
                                'table_data': df.to_dict('records'),
                                'table_num': table_num + 1
                            })
                        except Exception as e:
                            self.logger.warning(f"表格处理失败 第{page_num+1}页表格{table_num+1}: {e}")
                            continue
        
        return table_content
    
    def _table_to_text(self, df: pd.DataFrame, headers: List) -> str:
        """将表格转换为描述性文本"""
        try:
            # 基本表格信息
            table_text = f"表格包含 {len(df)} 行 {len(headers)} 列。列名: {', '.join(headers)}。"
            
            # 添加数值列的基本统计
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                for col in numeric_columns[:3]:  # 只处理前3个数值列
                    try:
                        col_data = pd.to_numeric(df[col], errors='coerce').dropna()
                        if len(col_data) > 0:
                            table_text += f" {col}列: 平均值{col_data.mean():.2f}, 范围[{col_data.min():.2f}-{col_data.max():.2f}]。"
                    except:
                        continue
            
            return table_text
        except Exception as e:
            self.logger.warning(f"表格文本转换失败: {e}")
            return df.to_markdown(index=False)
    
    def _extract_image_content(self, pdf_path: Path) -> List[Dict]:
        """提取图片内容"""
        image_content = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # 非透明图片
                        img_data = pix.tobytes("png")
                        image = Image.open(io.BytesIO(img_data))
                        
                        # OCR文字识别
                        ocr_text = pytesseract.image_to_string(
                            image, 
                            lang=self.ocr_languages
                        ).strip()
                        
                        image_content.append({
                            'page': page_num + 1,
                            'type': 'image',
                            'image_data': img_data,
                            'image_size': image.size,
                            'ocr_text': ocr_text,
                            'image_index': img_index + 1
                        })
                    
                    pix = None  # 释放内存
                    
                except Exception as e:
                    self.logger.warning(f"图片提取失败 第{page_num+1}页图片{img_index+1}: {e}")
                    continue
        
        doc.close()
        return image_content
    
    def generate_embeddings(self, content: Dict[str, List]) -> List[Dict]:
        """
        为所有内容生成embedding
        
        Args:
            content: 提取的内容字典
            
        Returns:
            包含embedding的内容列表
        """
        self.logger.info("开始生成embedding...")
        
        all_embeddings = []
        
        # 处理文本和表格
        text_items = content['text'] + content['tables']
        for item in text_items:
            try:
                embedding = self.text_model.encode(item['content'])
                all_embeddings.append({
                    **item,
                    'embedding': embedding,
                    'embedding_type': 'text',
                    'content_hash': self._generate_content_hash(item['content'])
                })
            except Exception as e:
                self.logger.warning(f"文本embedding生成失败: {e}")
                continue
        
        # 处理图片
        for item in content['images']:
            try:
                image_embedding, text_embedding = self._generate_image_embedding(item)
                
                # 使用图片embedding作为主要embedding，如果有OCR文本则融合
                if text_embedding is not None and len(item['ocr_text']) > 10:
                    # 简单融合策略
                    final_embedding = 0.3 * text_embedding + 0.7 * image_embedding
                else:
                    final_embedding = image_embedding
                
                all_embeddings.append({
                    **item,
                    'embedding': final_embedding,
                    'image_embedding': image_embedding,
                    'text_embedding': text_embedding,
                    'embedding_type': 'image',
                    'content_hash': self._generate_content_hash(
                        item['ocr_text'] + str(item['image_size'])
                    )
                })
            except Exception as e:
                self.logger.warning(f"图片embedding生成失败: {e}")
                continue
        
        self.logger.info(f"成功生成 {len(all_embeddings)} 个embedding")
        return all_embeddings
    
    def _generate_image_embedding(self, image_item: Dict) -> tuple:
        """生成图片embedding"""
        image = Image.open(io.BytesIO(image_item['image_data']))
        
        # 图片embedding
        inputs = self.clip_processor(images=image, return_tensors="pt")
        image_features = self.clip_model.get_image_features(**inputs)
        image_embedding = image_features.detach().numpy()[0]
        
        # 文本embedding (如果有OCR文本)
        text_embedding = None
        if image_item['ocr_text']:
            text_inputs = self.clip_processor(
                text=image_item['ocr_text'], 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=77
            )
            text_features = self.clip_model.get_text_features(**text_inputs)
            text_embedding = text_features.detach().numpy()[0]
        
        return image_embedding, text_embedding
    
    def _generate_content_hash(self, content: str) -> str:
        """生成内容哈希值"""
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()
    
    def store_in_vector_db(self, embeddings: List[Dict], collection_name: str = None):
        """
        将embedding存储到向量数据库
        
        Args:
            embeddings: 包含embedding的内容列表
            collection_name: 集合名称，如果为None则使用默认集合
        """
        if collection_name:
            collection = self.chroma_client.create_collection(collection_name)
        else:
            collection = self.collection
        
        documents = []
        embeddings_list = []
        metadatas = []
        ids = []
        
        for i, item in enumerate(embeddings):
            # 准备文档内容
            if item['type'] == 'text':
                content = item['content']
            elif item['type'] == 'table':
                content = f"表格: {item['content']}"
            else:  # image
                content = item['ocr_text'] or "图片内容"
                if not content.strip():
                    content = f"图片位于第{item['page']}页"
            
            documents.append(content[:1000])  # 限制长度
            embeddings_list.append(item['embedding'].tolist())
            metadatas.append({
                'type': item['type'],
                'page': item['page'],
                'embedding_type': item['embedding_type'],
                'content_hash': item.get('content_hash', '')
            })
            ids.append(f"doc_{i}_{item['type']}_{item['page']}")
        
        # 批量添加到向量数据库
        collection.add(
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        self.logger.info(f"成功存储 {len(embeddings)} 个embedding到向量数据库")
    
    def search_similar(self, 
                      query: str, 
                      n_results: int = 5,
                      filter_types: List[str] = None,
                      collection_name: str = None) -> List[Dict]:
        """
        搜索相似内容
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            filter_types: 过滤类型 ['text', 'table', 'image']
            collection_name: 集合名称
            
        Returns:
            相似内容列表
        """
        if collection_name:
            collection = self.chroma_client.get_collection(collection_name)
        else:
            collection = self.collection
        
        # 生成查询embedding
        query_embedding = self.text_model.encode(query).tolist()
        
        # 构建过滤器
        where_filter = None
        if filter_types:
            where_filter = {"type": {"$in": filter_types}}
        
        # 执行查询
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=['metadatas', 'documents', 'distances']
        )
        
        # 格式化结果
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return formatted_results
    
    def process_pdf(self, 
                   pdf_path: str, 
                   store_in_db: bool = True,
                   collection_name: str = None) -> Dict[str, Any]:
        """
        完整的PDF处理流程
        
        Args:
            pdf_path: PDF文件路径
            store_in_db: 是否存储到向量数据库
            collection_name: 集合名称
            
        Returns:
            处理结果字典
        """
        self.logger.info(f"开始处理PDF: {pdf_path}")
        
        try:
            # 1. 提取内容
            content = self.extract_content(pdf_path)
            
            # 2. 生成embedding
            embeddings = self.generate_embeddings(content)
            
            # 3. 存储到向量数据库
            if store_in_db:
                self.store_in_vector_db(embeddings, collection_name)
            
            result = {
                'status': 'success',
                'content': content,
                'embeddings': embeddings,
                'statistics': {
                    'total_chunks': len(embeddings),
                    'text_chunks': len(content['text']),
                    'table_chunks': len(content['tables']),
                    'image_chunks': len(content['images'])
                }
            }
            
            self.logger.info(f"PDF处理完成: {result['statistics']}")
            return result
            
        except Exception as e:
            self.logger.error(f"PDF处理失败: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_collection_info(self, collection_name: str = None) -> Dict:
        """获取集合信息"""
        try:
            if collection_name:
                collection = self.chroma_client.get_collection(collection_name)
            else:
                collection = self.collection
            
            count = collection.count()
            
            # 获取一些样本查看类型分布
            sample_results = collection.get(limit=min(100, count))
            type_distribution = {}
            if sample_results['metadatas']:
                for metadata in sample_results['metadatas']:
                    doc_type = metadata.get('type', 'unknown')
                    type_distribution[doc_type] = type_distribution.get(doc_type, 0) + 1
            
            return {
                'collection_count': count,
                'type_distribution': type_distribution,
                'sample_ids': sample_results['ids'][:5] if sample_results['ids'] else []
            }
        except Exception as e:
            self.logger.error(f"获取集合信息失败: {e}")
            return {}

if __name__ == "__main__":
    # 初始化处理器
    processor = PDFMultiModalProcessor()
    
    # 处理PDF
    result = processor.process_pdf("./source/产品介绍.pdf")
    
    if result['status'] == 'success':
        print(f"处理成功: {result['statistics']}")
        
        # 搜索示例
        similar_items = processor.search_similar("查询文本", n_results=3)
        print("相似内容:", similar_items)
        
        # 查看集合信息
        collection_info = processor.get_collection_info()
        print("集合信息:", collection_info)