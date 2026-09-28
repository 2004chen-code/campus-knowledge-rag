import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import numpy as np
import faiss
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# 初始化
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

with open("chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
index = faiss.read_index("faiss.index")
print(f"✅ 知识库加载完成，共 {len(chunks)} 个片段\n")

def rag_answer(question, top_k=3):
    # 1. 问题向量化
    q_emb = model.encode([question], normalize_embeddings=True)
    # 2. 检索
    scores, ids = index.search(q_emb.astype(np.float32), top_k)
    context = "\n\n".join(chunks[i] for i in ids[0])
    # 3. 构造提示词
    prompt = f"""请仅根据以下资料回答问题，若资料不足请如实说明"资料中未提及"，不要编造。

【资料】
{context}

【问题】{question}

【回答】"""
    # 4. 调用大模型，低温保证 grounding
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=800
    )
    return resp.choices[0].message.content, [chunks[i] for i in ids[0]], scores[0]

if __name__ == "__main__":
    # 测试 1：库内问题
    print("=" * 50)
    print("【测试 1】库内问题：奖学金怎么申请？")
    answer, sources, scores = rag_answer("奖学金怎么申请？")
    print(f"\n回答：{answer}")
    print(f"\n引用来源（Top3）：")
    for i, (s, sc) in enumerate(zip(sources, scores), 1):
        print(f"  [{i}] 相似度={sc:.4f} | {s[:80]}...")

    # 测试 2：库外问题
    print("\n" + "=" * 50)
    print("【测试 2】库外问题：今天天气怎么样？")
    answer, sources, scores = rag_answer("今天天气怎么样？")
    print(f"\n回答：{answer}")
    print("（应该回答'资料中未提及'，而不是编造天气）")