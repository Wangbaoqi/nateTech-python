import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_embedding(text: str, dim: int = 1536) -> list[float]:
    """获取文本的embedding"""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="SEMANTIC_SIMILARITY",
            output_dimensionality=dim,
        ),
    )
    return result.embeddings


def cus_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """计算两个向量的余弦相似度"""
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def custom_similarity_main():
    """自定义计算相似度"""
    texts = [
        "Hello world",
        "Transformer",
        "人工智能",
        "大语言模型",
        "中文比英文消耗更多Token吗？",
    ]
    query = "人工智能"

    embeddings = {text: get_embedding(text).pop().values for text in texts}

    # print(f"embeddings: {embeddings}")
    similarities = [
        (text, cus_cosine_similarity(embeddings[text], embeddings[query]))
        for text in texts
        if text != query
    ]

    print(f"similarities: {similarities}")
    # 按相似度从高到低排序
    similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"\n查询词: '{query}'")
    print("-" * 30)
    print("相似度排名:")
    for i, (text, score) in enumerate(similarities, 1):
        print(f"{i}. {text}: {score:.4f}")


def get_df_table():
    """获取文本的embedding并计算相似度"""
    texts = [
        "Hello world",
        "Transformer",
        "人工智能",
        "大语言模型",
        "中文比英文消耗更多Token吗？",
    ]

    result_emb = get_embedding(texts)

    embeddings = [e.values for e in result_emb]
    # print(f"embeddings: {embeddings}")

    df = pd.DataFrame(
        cosine_similarity(embeddings),
        columns=texts,
        index=texts,
    )
    print(df)


def get_mult_dimension_vec(imput: str, dim: str):
    """获取多维度的embedding"""

    result_vec = get_embedding(imput, dim)
    embeddings = result_vec.pop().values
    print(len(embeddings), "embeddings")
    print(embeddings)


def main():

    # custom_similarity_main()
    # get_df_table()
    get_mult_dimension_vec("苹果", 1536)


if __name__ == "__main__":
    main()
