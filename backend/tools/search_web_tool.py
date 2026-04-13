from tavily import TavilyClient
import os
from langchain.tools import tool

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str):
    """
    Tavily 联网检索公开网页，最多 3 条内容摘要（list，可能含噪声）。用于较新或需在线核对的客观事实；参数 query 为自然语言检索式
    """
    response = tavily.search(query=query, max_results=3)
    print("web_search===========response \n", response, "\n")
    return [r["content"] for r in response["results"]]