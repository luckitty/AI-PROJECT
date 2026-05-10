"""langchain-milvus 通过 ORM Collection(using=MilvusClient._using) 读 schema；
pymilvus 2.6 的 MilvusClient 不会在 connections 里注册该 alias，需在首次创建 client 后补注册。"""

_PATCHED = False


def ensure_milvus_orm_for_langchain() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from pymilvus import MilvusClient, connections

    _orig = MilvusClient.__init__

    def _wrapped(self, *args, **kwargs):
        _orig(self, *args, **kwargs)
        alias = self._using
        if connections.has_connection(alias):
            return
        cfg = self._config
        extra = dict(cfg.get_handler_kwargs())
        connections.connect(alias=alias, uri=cfg.uri, **extra)

    MilvusClient.__init__ = _wrapped  # type: ignore[method-assign]
    _PATCHED = True


ensure_milvus_orm_for_langchain()
