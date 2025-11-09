import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# 初始化模型
model = SentenceTransformer('all-MiniLM-L6-v2')

def chunked_embedding_csv(file_path, chunk_size=1000, text_column='text'):
    """
    分块处理CSV文件并生成嵌入向量
    """
    all_embeddings = []
    
    # 分块读取CSV
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # 提取文本列
        texts = chunk[text_column].tolist()
        
        # 生成嵌入向量
        embeddings = model.encode(texts)
        
        # 保存结果
        chunk['embeddings'] = list(embeddings)
        all_embeddings.append(chunk)
        
        print(f"处理了 {len(texts)} 条记录")
    
    # 合并所有结果
    return pd.concat(all_embeddings, ignore_index=True)


if __name__=="__main__":
    # 使用示例
    result = chunked_embedding_csv('./source/员工数据.csv', chunk_size=500)
    result.to_csv('embedded_file.csv', index=False)