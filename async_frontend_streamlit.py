import streamlit as st
import asyncio

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

from backend import get_workflow

st.set_page_config(page_title="LangGraph Chatbot", page_icon="🤖")
st.title("LangGraph Chatbot")

# -----------------------------
# session state
# -----------------------------
if "workflow" not in st.session_state:
    st.session_state.workflow = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "demo-thread-1"

if "frontend_messages" not in st.session_state:
    st.session_state.frontend_messages = []

if "waiting_for_interrupt" not in st.session_state:
    st.session_state.waiting_for_interrupt = False

# -----------------------------
# load workflow once
# -----------------------------
if st.session_state.workflow is None:
    st.session_state.workflow = asyncio.run(get_workflow())

workflow = st.session_state.workflow
CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}

# -----------------------------
# show old messages
# -----------------------------
for msg in st.session_state.frontend_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -----------------------------
# helper
# -----------------------------
def handle_graph_result(result):
    if "__interrupt__" in result:
        interrupt_data = result["__interrupt__"]
        if interrupt_data:
            first_interrupt = interrupt_data[0]
            interrupt_value = getattr(first_interrupt, "value", None)

            if isinstance(interrupt_value, dict):
                message = interrupt_value.get("message", "Approval needed.")
            else:
                message = "Approval needed."

            st.session_state.waiting_for_interrupt = True
            st.session_state.frontend_messages.append(
                {"role": "assistant", "content": message}
            )
        return

    messages = result.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and last_msg.content:
            st.session_state.frontend_messages.append(
                {"role": "assistant", "content": last_msg.content}
            )


# -----------------------------
# input
# -----------------------------
user_input = st.chat_input("Type here...")

if user_input:
    st.session_state.frontend_messages.append({"role": "user", "content": user_input})

    if st.session_state.waiting_for_interrupt:
        result = asyncio.run(
            workflow.ainvoke(Command(resume=user_input), config=CONFIG)
        )
        st.session_state.waiting_for_interrupt = False
        handle_graph_result(result)
        st.rerun()

    else:
        initial_state = {"messages": [HumanMessage(content=user_input)]}

        result = asyncio.run(workflow.ainvoke(initial_state, config=CONFIG))
        handle_graph_result(result)
        st.rerun()
