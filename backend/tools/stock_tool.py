from langchain.tools import tool
from pydantic import BaseModel,Field
from typing import Literal

class stockInput(BaseModel):
    stock_name:Literal["苹果","腾讯","特斯拉"]=Field(default="苹果",description="股票名称")
    stock_time:str=Field(description="股票时间",default="今天")

@tool(args_schema=stockInput)
def get_stock_price(stock_name: str,stock_time:str) -> str:
    """查询示例股票价格（演示数据）。仅当用户明确要问股票价格时调用；其他话题不要调用。"""
    stock={
        "特斯拉": "180美元",
        "苹果": "280美元",
        "腾讯": "320港币"
    }
    response=f"{stock_name}在{stock_time}的价格是：{stock[stock_name]}"
    return response
