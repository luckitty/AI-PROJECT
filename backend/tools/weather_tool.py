import os
import requests
from langchain.tools import tool
from core.config import AMAP_KEY

# AMAP_KEY="87a4c3fdc2dd16cfdee09073968c9781"
# AMAP_KEY = os.getenv("AMAP_KEY")

@tool
def get_weather(city:str) -> str:
  """"【强制调用】当你需要获取任何地点、任何日期的实时天气、温度或气候信息时，必须使用此工具。"""
  url="https://restapi.amap.com/v3/weather/weatherInfo"
  params={
    "key":AMAP_KEY,
    "city":city,
    "extensions":"base"
  }
  response=requests.get(url,params=params)

  data=response.json()

  if data["status"]!="1":
    return "天气查询失败"

  lives=data["lives"][0]
  city=lives["city"]
  weather=lives["weather"]
  temperature=lives["temperature"]
  winddirection=lives["winddirection"]
  print('高德天气--------------：',f"{city}今天的天气是{weather},温度为{temperature}℃,风向为{winddirection}")

  return f"{city}今天的天气是{weather},温度为{temperature}℃,风向为{winddirection}"









