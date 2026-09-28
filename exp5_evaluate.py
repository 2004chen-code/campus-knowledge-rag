import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from dotenv import load_dotenv
load_dotenv()
import json
from llm import call_llm
from exp4_3_rag import rag_answer # 复用你实验4-3的问答函数

# 准备测试用例（这里的答案可以粗略写，重点是让模型评分）
test_cases = [
    {"q": "奖学金怎么申请？", "ref": "学生向所在学院提出申请，学院初评后报学生工作部（处）审核，公示5个工作日后报上级主管部门备案。"},
    {"q": "学校有几个校区？", "ref": "学校现有南北两个校区，占地面积约2000余亩。"},
    {"q": "图书馆开放时间？", "ref": "依据知识库内容"},
    {"q": "宿舍违规怎么处理？", "ref": "依据知识库内容"},
    {"q": "怎么转专业？", "ref": "依据知识库内容"},
    {"q": "校园卡丢了怎么办？", "ref": "依据知识库内容"},
    {"q": "国家奖学金多少钱？", "ref": "每人每年8000元"},
    {"q": "今天天气怎么样？", "ref": "资料中未提及"},
    {"q": "红烧肉怎么做？", "ref": "资料中未提及"},
    {"q": "深度学习与机器学习的区别？", "ref": "资料中未提及"},
]

def llm_judge(question, answer, reference):
    prompt = f"""【问题】{question}
【参考答案】{reference}
【待评答案】{answer}
请从准确性、完整性、相关性三方面1-5分评分并说明理由，返回JSON：
{{"score": 数字, "reason": "理由"}}"""
    resp = call_llm(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    return json.loads(resp.choices[0].message.content)

results = []
print("开始评估...")
for tc in test_cases:
    answer, _, _ = rag_answer(tc["q"])
    score_data = llm_judge(tc["q"], answer, tc["ref"])
    results.append({
        "question": tc["q"],
        "answer": answer[:100] + "...",
        "score": score_data.get("score", 0),
        "reason": score_data.get("reason", "")
    })
    print(f"问题：{tc['q']} -> 得分：{score_data.get('score')}")

avg_score = sum(r["score"] for r in results) / len(results)
print(f"\n=== 评估完成 ===")
print(f"平均得分：{avg_score:.2f} / 5.00")

with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)