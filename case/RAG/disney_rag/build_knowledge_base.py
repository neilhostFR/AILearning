import os
from docx import Document as DocxDocument
from openai import OpenAI
from transformers import CLIPProcessor,CLIPModel
from typing import List,Any
from PIL import Image
import torch
from paddleocr import PaddleOCR
import faiss
import numpy as np
import pytesseract



class BuildKnowledge:
	"""构建知识库"""
	def __init__(self,docs_dir:str="./disney_knowledge_base",images_dir:str="./disney_knowledge_base/images"):
		self.docs_dir=docs_dir
		self.images_dir=images_dir
		# 这里使用“双塔模型” 文本和图片的向量存储在不同的向量空间中，检索时分别检索这2个向量空间
		self.text_embedding_dim=1024
		self.image_embedding_dim=512
		self.text_embedding_model="qwen3-embedding"
		# 初始化ollama客户端
		self.client=OpenAI(
			base_url="http://localhost:11434/v1",
			api_key="ollama"
		)
		
		print("开始加载clip模型")
		try:
			self.clip_model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
			self.clip_processor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
			print("clip模型加载成功")
		except Exception as e:
			print(f"加载clip模型失败：{e}")

		self.ocr=PaddleOCR(
			use_doc_orientation_classify=False,
			use_doc_unwarping=False,
			use_textline_orientation=False
		)

	def build_knowledge_base(self):
		"""读取文件夹中的docx和image，构建知识库"""
		# 源数据列表
		metadata_store=[]
		# 文本向量列表
		text_vectors=[]
		# 图片向量列表
		images_vectors=[]

		# 文件计数器
		doc_id_counter=0

		# 处理docx文档
		for filename in os.listdir(self.docs_dir):
			file_path=os.path.join(self.docs_dir,filename)
			if filename.startswith(".") or os.path.isdir(file_path):
				continue
			if filename.endswith(".docx"):
				print(f"正在处理:{filename}")
				chunks=self._parse_docx(file_path)

				for chunk in chunks:
					metadata={
						"id":doc_id_counter,
						"source":filename,
						"page":1
					}

					if chunk["type"]=="text" or chunk["type"]=="table":
						text=chunk["content"]
						if not text.strip():
							continue
						metadata["type"]="text"
						metadata["content"]=text

						vector=self.get_text_embedding(text)
						text_vectors.append(vector)
						metadata_store.append(metadata)
						doc_id_counter+=1

		# 处理图片
		for img_filename in os.listdir(self.images_dir):
			if img_filename.lower().endswith((".png",".jpg",".jpeg",".gif",".bmp")):
				img_path=os.path.join(self.images_dir,img_filename)
				img_text_info=self._image_to_text(img_path)
				# img_text_info=self._image_to_text_pytesseract(img_path)
				metadata={
					"id":doc_id_counter,
					"source":f"独立图片:{img_filename}",
					"type":"image",
					"path":img_path,
					"ocr":img_text_info["ocr"],
					"page":1
				}

				vector=self._get_image_embedding(img_path)
				images_vectors.append(vector)
				metadata_store.append(metadata)
				doc_id_counter+=1

		# 创建FAISS索引
		# 文本索引
		text_index=faiss.IndexFlatL2(self.text_embedding_dim)
		text_index_map=faiss.IndexIDMap(text_index)
		text_ids=[m["id"] for m in metadata_store if m["type"]=="text"]
		if text_vectors:
			text_index_map.add_with_ids(np.array(text_vectors).astype("float32"),np.array(text_ids))

		# 图片索引
		image_index=faiss.IndexFlatL2(self.image_embedding_dim)
		image_index_map=faiss.IndexIDMap(image_index)
		image_ids=[m["id"] for m in metadata_store if m["type"]=="image"]
		if images_vectors:
			image_index_map.add_with_ids(np.array(images_vectors).astype("float32"),np.array(image_ids))

		print(f"构建索引完成，共索引 {len(text_vectors)} 个文本片段和 {len(images_vectors)} 张图片")

		return metadata_store,text_index_map,image_index_map
	
	def _parse_docx(self,file_path:str)->List[Any]:
		"""解析docx文档，提取文本和表格"""
		doc=DocxDocument(file_path)
		content_chunks=[]

		for element in doc.element.body:
			# 段落
			if element.tag.endswith("p"):
				paragraph_text=""
				# 完整的命名空间
				namespaces = {
					'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
					'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
					'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships', 
					'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
					'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture'
				}
				for run in element.findall(".//w:t",namespaces):
					paragraph_text=run.text if run.text else ""
				if paragraph_text.strip():
					content_chunks.append({"type":"text","content":paragraph_text.strip()})
			# 表格
			if element.tag.endswith("tbl"):
				md_table=[]
				table=[t for t in doc.tables if t._element is element][0]

				if table.rows:
					# 添加表头
					header=[cell.text.strip() for cell in table.rows[0].cells]
					md_table.append("| "+" | ".join(header)+" |")
					md_table.append("|"+"---|"*len(header))

					# 添加数据行
					for row in table.rows[1:]:
						row_data=[cell.text.strip() for cell in row.cells]
						md_table.append("| "+" | ".join(row_data)+" |")
					table_content="\n".join(md_table)
					if table_content.strip():
						content_chunks.append({"type":"table","content":table_content})

		return content_chunks

	def get_text_embedding(self,text:str):
		"""获取文本向量"""
		response=self.client.embeddings.create(
			model=self.text_embedding_model,
			input=text,
			dimensions=self.text_embedding_dim
		)

		return response.data[0].embedding

	def _get_image_embedding(self,image_path:str):
		"""获取图片向量"""
		image=Image.open(image_path)
		inputs=self.clip_processor(images=image,return_tensors="pt")
		with torch.no_grad():
			image_features=self.clip_model.get_image_features(**inputs)
		# 归一化
		# image_features = image_features / image_features.norm(dim=-1, keepdim=True)
		return image_features[0].numpy()

	def _image_to_text_pytesseract(self,image_path:str):
		try:
			image = Image.open(image_path)

			ocr_text = pytesseract.image_to_string(image, lang='chi_sim+eng').strip()
			return {"ocr": ocr_text}
		except Exception as e:
			print(f"处理图片失败 {image_path}: {e}")
			return {"ocr": ""}

	def _image_to_text(self,image_path:str):
		try:
			# OCR
			print(f"正在对图片:{image_path}进行OCR识别")
			result=self.ocr.predict(image_path)
			return {"ocr":"\n".join(result[0]["rec_texts"])}
		except Exception as e:
			print(f"处理图片{image_path}失败:{e}")
			return {"ocr":""}



if __name__=="__main__":
	print("开始处理文档")
	knowledge=BuildKnowledge()
	metadata_store,text_index_map,image_index_map=knowledge.build_knowledge_base()
	print(metadata_store)
	print("-"*100)
	print(text_index_map)
	print("-"*100)
	print(image_index_map)