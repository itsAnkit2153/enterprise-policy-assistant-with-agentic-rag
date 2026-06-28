import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from src.config import GROQ_MODEL, GROQ_API_KEY
from src.prompts import system_prompt
from src.tools import query_rewriter, policy_retriever, set_vector_store
from src.query_rewriter import rewrite_query   # direct call for UI metadata

_NOT_ANSWERED = {
    "answer":     "Sorry, I could not find this information in the uploaded policy documents.",
    "sources":    [],
    "confidence": "LOW",
    "reasoning":  "No relevant context was retrieved from the indexed policy documents.",
}


def build_agent(vector_store):
    """
    Build a LangGraph ReAct agent with two tools:
      1. query_rewriter  — rewrites + expands the user question
      2. policy_retriever — multi-query retrieval with deduplication
    """
    set_vector_store(vector_store)

    llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)

    agent = create_react_agent(
        model=llm,
        tools=[query_rewriter, policy_retriever],
        prompt=system_prompt(),
    )
    return agent


def _parse_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON from the agent's final message."""
    text = raw.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "answer":     text,
            "sources":    [],
            "confidence": "LOW",
            "reasoning":  "Response could not be parsed as structured JSON.",
        }


def run_agent(agent, user_query: str) -> tuple[dict, dict]:
    """
    Run the ReAct agent on a user query.

    Returns:
      result      : dict with answer, sources, confidence, reasoning
      rewrite_meta: dict with original_query, rewritten_query, alternatives, reasoning
    """
    # Run rewriter directly so UI can show the metadata regardless of agent internals
    rewrite_meta = rewrite_query(user_query)

    try:
        messages = [HumanMessage(content=user_query)]
        response = agent.invoke({"messages": messages})
        last = response["messages"][-1].content
        result = _parse_response(last)
    except Exception as e:
        result = {**_NOT_ANSWERED, "reasoning": f"Agent error: {str(e)}"}

    return result, rewrite_meta
