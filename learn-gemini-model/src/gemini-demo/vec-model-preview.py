import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def prepare_query(query):
    """准备查询"""
    return f"task: search result | query:{query}"


def prepare_document(content, title=None):
    """准备文档"""
    if title is None:
        title = "none"
    return f"title:{title} | content:{content}"
