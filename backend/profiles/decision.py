from .schema import UserProfile


def should_update_profile(profile: UserProfile) -> bool:
    # 有长期特征才更新（与业务上对「值得写入 Redis」的判定一致）
    if profile.interests or profile.personality:
        return True
    return False