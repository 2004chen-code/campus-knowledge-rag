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
client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
with open("chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
index = faiss.read_index("faiss.index")

def ask_llm(question):
    """纯大模型回答（不带知识库）"""
    prompt = f"请回答以下问题：{question}"
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=300
    )
    return resp.choices[0].message.content

def ask_rag(question, top_k=3):
    """RAG 回答（带知识库）"""
    q_emb = model.encode([question], normalize_embeddings=True)
    scores, ids = index.search(q_emb.astype(np.float32), top_k)
    context = "\n\n".join(chunks[i] for i in ids[0])
    prompt = f"""请仅根据以下资料回答问题，若资料不足请如实说明"资料中未提及"，不要编造。

【资料】
{context}

【问题】{question}

【回答】"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500
    )
    return resp.choices[0].message.content

# 10个测试问题
questions = [
    "奖学金怎么申请？",
    "图书馆开放时间？",
    "宿舍违规怎么处理？",
    "怎么转专业？",
    "校园卡丢了怎么办？",
    "国家奖学金多少钱？",
    "今天天气怎么样？",
    "红烧肉怎么做？",
    "深度学习与机器学习的区别？",
    "学校有几个校区？"
]

print("=" * 60)
print("开始批量测试...")
results = []
for i, q in enumerate(questions, 1):
    print(f"\n--- 测试 {i}: {q} ---")
    ans_llm = ask_llm(q)
    print(f"【纯大模型】{ans_llm[:100]}...")
    
    ans_rag = ask_rag(q)
    print(f"【RAG】{ans_rag[:100]}...")
    
    results.append({
        "id": i,
        "question": q,
        "llm_answer": ans_llm,
        "rag_answer": ans_rag
    })

with open("test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n✅ 测试完成，结果已保存到 test_results.json")