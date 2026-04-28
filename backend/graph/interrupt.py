from interruptController.interrupt_manager import interrupt_manager

def check_interrupt(state):
    # 会话已被 stop 时，写入统一状态，供图路由和调用方识别并尽快结束执行。
    if interrupt_manager.is_stopped(state["session_id"]):
        return {
            "is_interrupted": True,
            "final_answer": "请求已中断",
        }
    return {"is_interrupted": False}