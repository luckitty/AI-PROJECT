from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.conversation_compress import conversation_compress_node
from graph.router import route_from_planner
from memory.short_memory import get_short_term_checkpointer

from agents.planner_node import planner_node
from agents.memory_node import memory_node
from agents.rag_node import rag_node
from agents.tool_node import tool_node
from agents.response_node import response_node
from agents.save_memory_node import save_memory_node
from graph.interrupt import (
    human_checkpoint_start,
    human_checkpoint_after_planner,
    human_checkpoint_after_rag,
    human_checkpoint_after_capability,
    human_checkpoint_after_response,
)


def route_after_rag(state):
    """
    rag 之后：攻略生成统一在 response 节点调用 search_travel（有召回则带素材，无则空素材），此处不再进入独立 travel 节点。
    """
    if state.get("human_halt"):
        return "halt"
    return "capability"


def route_human_or_next(state, next_label):
    """
    人机选择终止或子节点已协作中断时收口到 END；否则进入指定下游标签。
    """
    if state.get("human_halt"):
        return "halt"
    return next_label


def route_human_or_planner(state):
    """
    planner 后的分发：若本轮已标记终止（人机拒绝或协作 stop），否则按 planner 的 need_* 路由。
    """
    if state.get("human_halt"):
        return "halt"
    return route_from_planner(state)


def add_human_gate(graph, gate_name, gate_fn):
    """注册绑定 ``interrupt()`` 的人机闸门节点。"""
    graph.add_node(gate_name, gate_fn)


def connect_with_human_gate(graph, source_nodes, gate_name, route_fn, route_map):
    """上游先进人机闸门，再按 route_map 分流。"""
    for source_node in source_nodes:
        graph.add_edge(source_node, gate_name)
    graph.add_conditional_edges(gate_name, route_fn, route_map)


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("conversation_compress", conversation_compress_node)
    graph.add_node("planner", planner_node)
    graph.add_node("memory", memory_node)
    graph.add_node("rag", rag_node)
    graph.add_node("tool", tool_node)
    graph.add_node("response", response_node)
    graph.add_node("save_memory", save_memory_node)

    human_gate_specs = [
        ("human_checkpoint_start", human_checkpoint_start),
        ("human_checkpoint_after_planner", human_checkpoint_after_planner),
        ("human_checkpoint_after_rag", human_checkpoint_after_rag),
        ("human_checkpoint_after_capability", human_checkpoint_after_capability),
        ("human_checkpoint_after_response", human_checkpoint_after_response),
    ]
    for gate_name, gate_fn in human_gate_specs:
        add_human_gate(graph, gate_name, gate_fn)

    graph.set_entry_point("human_checkpoint_start")

    graph.add_conditional_edges(
        "human_checkpoint_start",
        lambda s: route_human_or_next(s, "conversation_compress"),
        {
            "halt": END,
            "conversation_compress": "conversation_compress",
        },
    )
    graph.add_edge("conversation_compress", "planner")

    connect_with_human_gate(
        graph,
        ["planner"],
        "human_checkpoint_after_planner",
        route_human_or_planner,
        {
            "halt": END,
            "rag": "rag",
            "memory": "memory",
            "tool": "tool",
            "response": "response",
        },
    )

    connect_with_human_gate(
        graph,
        ["rag"],
        "human_checkpoint_after_rag",
        route_after_rag,
        {
            "halt": END,
            "capability": "human_checkpoint_after_capability",
        },
    )

    connect_with_human_gate(
        graph,
        ["memory", "tool"],
        "human_checkpoint_after_capability",
        lambda s: route_human_or_next(s, "response"),
        {
            "halt": END,
            "response": "response",
        },
    )

    connect_with_human_gate(
        graph,
        ["response"],
        "human_checkpoint_after_response",
        lambda s: route_human_or_next(s, "save_memory"),
        {
            "halt": END,
            "save_memory": "save_memory",
        },
    )
    graph.add_edge("save_memory", END)

    return graph.compile(checkpointer=get_short_term_checkpointer())
