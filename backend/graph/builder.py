from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.router import route_from_planner
from memory.short_memory import get_short_term_checkpointer

from agents.planner_node import planner_node
from agents.memory_node import memory_node
from agents.rag_node import rag_node
from agents.travel_node import travel_node
from agents.tool_node import tool_node
from agents.response_node import response_node
from agents.save_memory_node import save_memory_node


def build_graph():
    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("planner", planner_node)
    graph.add_node("memory", memory_node)
    graph.add_node("rag", rag_node)
    graph.add_node("travel", travel_node)
    graph.add_node("tool", tool_node)
    graph.add_node("response", response_node)
    graph.add_node("save_memory", save_memory_node)
    # 入口
    graph.set_entry_point("planner")

    # 动态路由🔥
    graph.add_conditional_edges(
        "planner",
        route_from_planner,
        {
            "rag": "rag",
            "memory": "memory",
            "tool": "tool",
            "response": "response",
        },
    )
    # rag 结束后固定进入 travel，确保旅游编排链路稳定。
    graph.add_edge("rag", "travel")
    # 汇总流向 response
    graph.add_edge("travel", "response")
    graph.add_edge("memory", "response")
    graph.add_edge("tool", "response")
    graph.add_edge("response", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile(checkpointer=get_short_term_checkpointer())