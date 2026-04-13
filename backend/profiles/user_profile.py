"""
用户画像存储（Redis）：

- 从用户输入提取结构化画像字段；
- 仅在命中“长期特征”时更新；
- 采用 JSON 字符串落 Redis，避免依赖 RedisJSON 模块。
"""
import json
import threading
from datetime import datetime, timezone

from memory.redis_config import REDIS_URL

# 包内相对导入：前面的「.」表示当前包（profiles），避免与标准库 profile 同名冲突。
from .decision import should_update_profile
from .extractor import extract_profile

USER_PROFILE_KEY_PREFIX = "user_profile:"
# 画像属于中长期信息，这里给 180 天 TTL，长期不用会自动过期。
USER_PROFILE_TTL_SECONDS = 15552000

# 进程内单例：整个应用共用同一个 Redis 连接，避免每个请求都新建 TCP 连接。
profileRedisClient = None
# 多线程下同时第一次调用 get 时可能并发，用锁保证只创建一次客户端（双重检查惯用法）。
profileRedisClientLock = threading.Lock()


def get_profile_redis_client():
    """
    懒加载 Redis 客户端，复用连接减少频繁建连开销。
    """
    # global：要修改模块级变量 profileRedisClient，必须声明，否则 Python 会当成局部变量。
    global profileRedisClient
    with profileRedisClientLock:
        if profileRedisClient is None:
            # 延迟导入，避免模块加载阶段因 Redis 环境问题直接中断应用启动。
            import redis
            # from_url：用 redis://... 这种 URL 一次配好地址和库号；decode_responses=True 表示从 Redis 读出的 bytes 自动转成 str。
            profileRedisClient = redis.from_url(REDIS_URL, decode_responses=True)
        return profileRedisClient


def build_user_profile_key(user_id: str) -> str:
    """
    统一用户画像 key 规范，便于后续排查和批量管理。
    """
    return f"{USER_PROFILE_KEY_PREFIX}{user_id}"


def get_user_profile(user_id: str) -> dict:
    """
    读取用户画像；不存在或异常时返回空 dict。
    """
    if not user_id:
        return {}
    try:
        key = build_user_profile_key(user_id)
        # .get(key)：Redis 的 GET 命令；没有该 key 时返回 None，空字符串在 Python 里也是 falsy。
        raw = get_profile_redis_client().get(key)
        if not raw:
            return {}
        data = json.loads(raw)
        # JSON 根可能是 []、"字符串" 等，loads 不一定是 dict；非 dict 时降级为空，避免后面当字典用报错。
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print("get_user_profile===========error \n", str(exc), "\n")
        return {}


def merge_user_profile(old_profile: dict, new_profile: dict) -> dict:
    """
    画像合并策略：
    - 新值为空则保留旧值；
    - 列表字段做去重追加；
    - 标量字段用新值覆盖旧值。
    """
    # old_profile or {}：若传入 None，用空字典代替，避免对 None 调 .items() 报错（短路求值常见写法）。
    merged = dict(old_profile or {})
    for key, value in (new_profile or {}).items():
        if value in (None, "", [], {}):
            continue

        old_value = merged.get(key)
        if isinstance(old_value, list) and isinstance(value, list):
            # 用 set 做“是否出现过”的判断，平均 O(1)；列表里 in 全表扫描会慢。
            seen = set()
            combined = []
            # 两个列表相加：生成新列表，元素顺序为先旧后新（不是数学加法）。
            for item in old_value + value:
                token = str(item).strip()
                if not token or token in seen:
                    continue
                seen.add(token)
                combined.append(token)
            merged[key] = combined
            continue

        merged[key] = value
    return merged


def save_user_profile(user_id: str, profile: dict) -> dict:
    """
    写入用户画像（合并后覆盖存储）。
    """
    if not user_id or not isinstance(profile, dict):
        return {}
    try:
        key = build_user_profile_key(user_id)
        current_profile = get_user_profile(user_id)
        print("save_user_profile===========current_profile \n", current_profile, "\n")
        merged_profile = merge_user_profile(current_profile, profile)
        print("save_user_profile===========merged_profile \n", merged_profile, "\n")
        # timezone.utc：明确用 UTC，避免服务器本地时区不同导致时间混乱；isoformat() 得到 ISO8601 字符串。
        merged_profile["updated_at"] = datetime.now(timezone.utc).isoformat()

        # ensure_ascii=False：中文等字符直接输出为 Unicode，不转成 \uXXXX 转义，便于人读和前端展示。
        payload = json.dumps(merged_profile, ensure_ascii=False)
        # setex：SET + 过期时间一条命令完成；秒数到期后 Redis 自动删 key（与 USER_PROFILE_TTL_SECONDS 一致）。
        get_profile_redis_client().setex(key, USER_PROFILE_TTL_SECONDS, payload)
        print("save_user_profile===========payload \n", payload, "\n")
        return merged_profile
    except Exception as exc:
        print("save_user_profile===========error \n", str(exc), "\n")
        return {}


def update_user_profile_from_text(user_id: str, user_input: str) -> dict:
    """
    从用户输入提取画像并按规则更新。
    """
    # (user_input or "")：若 user_input 为 None，先当成空字符串再 strip，避免对 None 调 strip 报错。
    if not user_id or not (user_input or "").strip():
        return {}

    extracted = extract_profile(user_input)
    if not should_update_profile(extracted):
        return {}

    # 只持久化 schema 字段；updated_at 在 save_user_profile 里写入。
    return save_user_profile(user_id, extracted.model_dump())
