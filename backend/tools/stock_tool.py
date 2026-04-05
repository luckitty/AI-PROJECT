from langchain.tools import tool
from pydantic import BaseModel,Field
from typing import Literal

class stockInput(BaseModel):
    stock_name:Literal["苹果","腾讯","特斯拉"]=Field(default="苹果",description="股票名称")
    stock_time:str=Field(description="股票时间",default="今天")

@tool(args_schema=stockInput)
def get_stock_price(stock_name: str,stock_time:str) -> str:
    """【强制调用】当你需要获取某个时间的股票价格时调用此工具"""
    stock={
        "特斯拉": "180美元",
        "苹果": "280美元",
        "腾讯": "320港币"
    }
    response=f"{stock_name}在{stock_time}的价格是：{stock[stock_name]}"
    return response
