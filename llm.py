import os
import time
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def call_llm(messages, retries=3, **kwargs):
    """带指数退避重试的大模型调用"""
    for i in range(retries):
        try:
            return client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                **kwargs
            )
        except Exception as e:
            if i == retries - 1:
                raise
            print(f"调用失败，第{i+1}次重试... 错误：{e}")
            time.sleep(2 ** i)