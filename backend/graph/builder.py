from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.router import route_from_planner
from memory.short_memory import get_short_term_checkpointer

from agents.planner_node import planner_node
from agents.memory_node import memory_node
from agents.rag_node import rag_node
from agents.travel_node import travel_node
from agents.amap_node import amap_node
from agents.tool_node import tool_node
from agents.response_node import response_node
from agents.save_memory_node import save_memory_node
from graph.interrupt import check_interrupt


def route_interrupt_or_node(state, next_node):
    """
    通用中断路由：命中中断返回 interrupt，否则返回指定下游节点。
    """
    return "interrupt" if state.get("is_interrupted") else next_node


def route_interrupt_or_planner_target(state):
    """
    planner 后通用路由：命中中断直接结束，否则按 planner 决策分发。
    """
    if state.get("is_interrupted"):
        return "interrupt"
    return route_from_planner(state)


def add_interrupt_gate(graph, gate_name):
    """
    注册一个可复用的中断闸门节点，统一绑定 check_interrupt。
    """
    graph.add_node(gate_name, check_interrupt)


def connect_with_interrupt_gate(graph, source_nodes, gate_name, route_func, route_map):
    """
    让一个或多个上游节点先经过中断闸门，再按 route_map 分流到下游。
    """
    for source_node in source_nodes:
        graph.add_edge(source_node, gate_name)
    graph.add_conditional_edges(gate_name, route_func, route_map)


def build_graph():
    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node("planner", planner_node)
    graph.add_node("memory", memory_node)
    graph.add_node("rag", rag_node)
    graph.add_node("travel", travel_node)
    graph.add_node("amap", amap_node)
    graph.add_node("tool", tool_node)
    graph.add_node("response", response_node)
    graph.add_node("save_memory", save_memory_node)

    # 统一注册所有中断闸门，后续新增链路只需要补 gate 名称和下游映射。
    interrupt_gate_names = [
        "check_interrupt_start",
        "check_interrupt_after_planner",
        "check_interrupt_after_rag",
        "check_interrupt_after_travel",
        "check_interrupt_after_capability",
        "check_interrupt_after_response",
    ]
    for gate_name in interrupt_gate_names:
        add_interrupt_gate(graph, gate_name)

    # 入口
    graph.set_entry_point("check_interrupt_start")

    # 入口先检查一次 stop，避免请求刚进图就继续执行。
    graph.add_conditional_edges(
        "check_interrupt_start",
        lambda state: route_interrupt_or_node(state, "planner"),
        {
            "interrupt": END,
            "planner": "planner",
        },
    )

    # planner 产出 need_* 后先过中断闸门，再进入首跳分发。
    connect_with_interrupt_gate(
        graph,
        ["planner"],
        "check_interrupt_after_planner",
        route_interrupt_or_planner_target,
        {
            "interrupt": END,
            "rag": "rag",
            "memory": "memory",
            "tool": "tool",
            "response": "response",
        },
    )

    # rag -> travel 间加闸门，保证检索阶段可立即停止。
    connect_with_interrupt_gate(
        graph,
        ["rag"],
        "check_interrupt_after_rag",
        lambda state: route_interrupt_or_node(state, "travel"),
        {
            "interrupt": END,
            "travel": "travel",
        },
    )

    # travel -> amap 间加闸门，保证编排阶段可立即停止。
    connect_with_interrupt_gate(
        graph,
        ["travel"],
        "check_interrupt_after_travel",
        lambda state: route_interrupt_or_node(state, "amap"),
        {
            "interrupt": END,
            "amap": "amap",
        },
    )

    # 子能力节点结束后统一先检查 stop，再决定是否进入 response。
    connect_with_interrupt_gate(
        graph,
        ["amap", "memory", "tool"],
        "check_interrupt_after_capability",
        lambda state: route_interrupt_or_node(state, "response"),
        {
            "interrupt": END,
            "response": "response",
        },
    )

    # response 结束后再检查一次 stop，保证最终生成阶段可终止。
    connect_with_interrupt_gate(
        graph,
        ["response"],
        "check_interrupt_after_response",
        lambda state: route_interrupt_or_node(state, "save_memory"),
        {
            "interrupt": END,
            "save_memory": "save_memory",
        },
    )
    graph.add_edge("save_memory", END)

    return graph.compile(checkpointer=get_short_term_checkpointer())