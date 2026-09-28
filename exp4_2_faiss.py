import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 读取切分后的 chunks
with open("chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"共 {len(chunks)} 个片段，开始向量化...")

# 加载中文 Embedding 模型
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# 批量编码，normalize_embeddings=True 使向量模长为 1，此时内积等价于余弦相似度
embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=True)
print(f"向量维度：{embeddings.shape}")

# 创建 FAISS 精确内积索引
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype(np.float32))
print(f"✅ 索引建立完成，共 {index.ntotal} 个向量")

# 验证：第 3 个向量与自身做点积，应接近 1.0
self_sim = float(np.dot(embeddings[2], embeddings[2]))
print(f"自相似度验证（应≈1.0）：{self_sim:.4f}")

# 保存索引
faiss.write_index(index, "faiss.index")
print("✅ 已保存 faiss.index")

# 测试检索：随便问一个问题
test_query = "奖学金怎么申请？"
q_emb = model.encode([test_query], normalize_embeddings=True)
scores, ids = index.search(q_emb.astype(np.float32), k=3)

print(f"\n测试查询：{test_query}")
for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), 1):
    print(f"\nTop{rank} 相似度={score:.4f}")
    print(f"  片段：{chunks[idx][:100]}...")