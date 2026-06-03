"""LangGraph builder for the agentic production pipeline."""

from __future__ import annotations

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover - exercised only without dependency installed
    END = START = None
    StateGraph = None
    _LANGGRAPH_IMPORT_ERROR = exc
else:
    _LANGGRAPH_IMPORT_ERROR = None

from .nodes import (
    bgm_agent_node,
    fail_closed_node,
    finalize_artifacts_node,
    grounding_agent_node,
    image_generation_agent_node,
    load_source_node,
    meme_planner_agent_node,
    persist_script_artifacts_node,
    render_agent_node,
    route_after_failure,
    script_writer_agent_node,
    tts_agent_node,
)
from .state import PipelineState


def build_pipeline_graph():
    if StateGraph is None:
        raise RuntimeError(
            "LangGraph is not installed. Install project dependencies with `pip install -e .`."
        ) from _LANGGRAPH_IMPORT_ERROR

    graph = StateGraph(PipelineState)
    graph.add_node("load_source", load_source_node)
    graph.add_node("grounding_agent", grounding_agent_node)
    graph.add_node("script_writer_agent", script_writer_agent_node)
    graph.add_node("persist_script_artifacts", persist_script_artifacts_node)
    graph.add_node("meme_planner_agent", meme_planner_agent_node)
    graph.add_node("image_generation_agent", image_generation_agent_node)
    graph.add_node("tts_agent", tts_agent_node)
    graph.add_node("bgm_agent", bgm_agent_node)
    graph.add_node("render_agent", render_agent_node)
    graph.add_node("finalize_artifacts", finalize_artifacts_node)
    graph.add_node("fail_closed", fail_closed_node)

    graph.add_edge(START, "load_source")
    graph.add_conditional_edges("load_source", route_after_failure, {"continue": "grounding_agent", "fail_closed": "fail_closed"})
    graph.add_conditional_edges("grounding_agent", route_after_failure, {"continue": "script_writer_agent", "fail_closed": "fail_closed"})
    graph.add_conditional_edges("script_writer_agent", route_after_failure, {"continue": "persist_script_artifacts", "fail_closed": "fail_closed"})
    graph.add_conditional_edges("persist_script_artifacts", route_after_failure, {"continue": "meme_planner_agent", "fail_closed": "fail_closed"})
    graph.add_conditional_edges("meme_planner_agent", route_after_failure, {"continue": "image_generation_agent", "fail_closed": "fail_closed"})
    graph.add_conditional_edges("image_generation_agent", route_after_failure, {"continue": "tts_agent", "fail_closed": "fail_closed"})
    graph.add_edge("tts_agent", "bgm_agent")
    graph.add_edge("bgm_agent", "render_agent")
    graph.add_conditional_edges("render_agent", route_after_failure, {"continue": "finalize_artifacts", "fail_closed": "fail_closed"})
    graph.add_edge("finalize_artifacts", END)
    graph.add_edge("fail_closed", END)

    return graph.compile()
