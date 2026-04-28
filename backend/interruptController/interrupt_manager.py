class InterruptManager:
    def __init__(self):
        self.flags = {}
        self.user_active_session = {}

    def stop(self, session_id: str):
        self.flags[session_id] = True

    def reset(self, session_id: str):
        self.flags[session_id] = False

    def is_stopped(self, session_id: str) -> bool:
        return self.flags.get(session_id, False)

    def register_user_session(self, user_id: str, session_id: str):
        # 同一用户发起新会话时，先中断旧会话，避免刷新后旧请求继续执行。
        last_session_id = self.user_active_session.get(user_id)
        if last_session_id and last_session_id != session_id:
            self.stop(last_session_id)
        self.user_active_session[user_id] = session_id

interrupt_manager = InterruptManager()
