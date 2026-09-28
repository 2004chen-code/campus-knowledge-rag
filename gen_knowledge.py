import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)

prompt = """《陕西理工大学校园知识手册》，包含以下章节：
1. 学校概况（历史、校区、学院）
2. 学籍管理（入学、注册、休学、复学、转专业）
3. 课程与考试（选课、考试、补考、重修）
4. 奖学金与助学金（国家奖学金、励志奖学金、助学金申请条件）
5. 宿舍管理（入住、退宿、违规处理）
6. 图书馆使用（借阅规则、开放时间、电子资源）
7. 校园卡与网络（办理、充值、挂失、校园网）
8. 社团与活动（社团种类、活动申请、经费）
9. 就业与实习（就业指导、实习申请、三方协议）
10. 常见问题答疑（20 条 Q&A）"""

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
    max_tokens=8000
)

with open("knowledge.txt", "w", encoding="utf-8") as f:
    f.write(resp.choices[0].message.content)

print(f"✅ 已生成 knowledge.txt，共 {len(resp.choices[0].message.content)} 字")