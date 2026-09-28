import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("knowledge.txt", encoding="utf-8") as f:
    text = f.read()

print(f"原文长度：{len(text)} 字")
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_text(text)
print(f"共切分为 {len(chunks)} 个片段\n")

for i, c in enumerate(chunks[:3]):
    print(f"--- 片段 {i} ---")
    print(c[:100] + "...\n")

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)
print(f"✅ 已保存 chunks.json，共 {len(chunks)} 个片段")