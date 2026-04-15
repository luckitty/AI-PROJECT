# 小红书旅游路线规划模块

这个模块用于：

1. 抓取小红书旅游帖子（标题/正文等）
2. 解析与清洗帖子数据
3. 生成旅游路线草案（按天输出）

## 运行前准备

1. 安装 Python 依赖（含 playwright）
2. 安装浏览器驱动
3. 默认建议先非 headless 运行，便于观察页面状态

```bash
cd backend
pip3 install -r requirements.txt
python3 -m playwright install chromium
```

## 快速运行

在项目根目录执行：

```bash
cd backend
python3 -m crawler.runner \
  --keyword "杭州 三日游 攻略" \
  --destination "杭州" \
  --days 3 \
  --max-scroll-rounds 8 \
  --max-note-count 20
```

## 使用你抓包的 HTTP 接口（不走页面）

你已经提供了 `search/notes` 和 `feed` 接口后，可用接口模式：

```bash
cd backend
python3 -m crawler.runner --keyword "香港 一日游 攻略" --use-http-api --force-refresh
```

注意：接口模式需要你在 `crawler/config.py` 里填好 `http_cookie`，并把抓包里的关键请求头放进 `http_common_headers`。
另外建议把抓包里的 `search_id` 放到 `http_search_id`，便于尽量贴近真实请求形态。

默认开启本地缓存：同一个关键词会优先读取 `data/cache/` 下历史抓取结果，减少重复抓取频次。
如果你要重新抓最新数据，可追加：

```bash
python3 -m crawler.runner --keyword "杭州 三日游 攻略" --force-refresh
```

## 复用你已打开的 Chrome 登录态（CDP）

普通启动的 Chrome 无法被 Playwright 直接接管。要复用你已经登录的小红书，需要先用远程调试端口启动 Chrome：

```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --remote-debugging-port=9222 \
  --user-data-dir="/tmp/xhs-cdp-profile"
```

在这个 Chrome 里登录小红书后，运行：

```bash
cd backend
python3 -m crawler.runner --keyword "杭州 三日游 攻略" --use-cdp --cdp-url "http://127.0.0.1:9222"
```

## 输出文件

- 帖子明细：`data/xhs_notes.json`
- 路线草案：`data/xhs_route_plan.json`

## 路线数据结构

`xhs_route_plan.json` 主要字段：

- `destination`: 目的地
- `days`: 行程天数
- `top_spots`: 高频景点词
- `route`: 每日路线（景点列表 + 备注）

## 风险说明

Playwright 只能降低被风控概率，无法保证账号绝对安全。建议使用小号、低频抓取并限制每日任务量。
