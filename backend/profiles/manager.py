def merge_profile(old: dict, new: dict):
    for k, v in new.items():
        if isinstance(v, list):
            old[k] = list(set(old.get(k, []) + v))
        elif v:
            old[k] = v
    return old