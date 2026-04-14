import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 加载 .env 文件中的环境变量
load_dotenv()

print(os.getenv("GEMINI_API_KEY"))


def main():
    # 初始化客户端，默认会从环境变量 GEMINI_API_KEY 中读取 API Key
    # 请确保在运行前设置了环境变量：export GEMINI_API_KEY="your_api_key"
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # 创建一个对话会话 (使用当前推荐的默认模型 gemini-2.5-flash)
    print("正在初始化 Gemini 对话...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=["你好，我是来学习gemini api"],
        config=types.GenerateContentConfig(
            temperature=0.8,
            max_output_tokens=1000,
        ),
    )
    print(response.text)


if __name__ == "__main__":
    if "GEMINI_API_KEY" not in os.environ:
        print("警告：未检测到 GEMINI_API_KEY 环境变量。请先设置您的 API 密钥。")
        print("例如运行: export GEMINI_API_KEY='your_api_key'")
    else:
        main()
