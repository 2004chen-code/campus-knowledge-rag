RAG_PROMPT_TEMPLATE = """请仅根据以下资料回答问题，若资料不足请如实说明"资料中未提及"，不要编造。

【资料】
{context}

【问题】{question}

【回答】"""