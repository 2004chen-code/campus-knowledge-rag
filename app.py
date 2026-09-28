import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from llm import call_llm
from prompts import RAG_PROMPT_TEMPLATE

st.set_page_config(page_title="校园知识问答助手", page_icon="🎓", layout="wide")

@st.cache_resource
def load_resources():
    with open("chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    index = faiss.read_index("faiss.index")
    return chunks, model, index

chunks, model, index = load_resources()

def rag_answer(question, top_k=3):
    q_emb = model.encode([question], normalize_embeddings=True)
    scores, ids = index.search(q_emb.astype(np.float32), top_k)
    context = "\n\n".join(chunks[i] for i in ids[0])
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    resp = call_llm(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=800
    )
    return resp.choices[0].message.content, [chunks[i] for i in ids[0]], scores[0]

with st.sidebar:
    st.title("🎓 校园知识助手")
    st.markdown(f"**知识库**：{len(chunks)} 个片段")
    st.markdown("---")
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("**示例问题：**")
    st.markdown("- 奖学金怎么申请？")
    st.markdown("- 图书馆开放时间？")

st.title("🎓 校园知识问答助手（RAG）")
st.caption("基于私有知识库的检索增强问答，答案附带引用来源")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and "sources" in m:
            with st.expander("📚 查看引用来源"):
                for i, s in enumerate(m["sources"], 1):
                    st.markdown(f"**片段 {i}**：{s[:300]}...")

if question := st.chat_input("请输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库..."):
            answer, sources, scores = rag_answer(question)
        st.markdown(answer)
        with st.expander("📚 查看引用来源"):
            for i, (s, sc) in enumerate(zip(sources, scores), 1):
                st.markdown(f"**片段 {i}**（相似度 {sc:.4f}）：{s[:300]}...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })