# LangGraph Chatbot with Tools, MCP Client, and Human-in-the-Loop (HITL)

This project is a simple LangGraph-based chatbot that shows how to:

- Use **LangGraph** to build a graph-based AI workflow
- Connect an **LLM (OpenAI)** with multiple **tools** (web search, stock tools, GitHub MCP, filesystem MCP)
- Use the **Model Context Protocol (MCP)** via `langchain_mcp_adapters` and `MultiServerMCPClient`
- Implement **Human-in-the-Loop (HITL)** approval for sensitive actions (like buying stocks)
- Build an **async Streamlit frontend** that talks to the LangGraph backend and handles interrupts/approvals

The code is split into two main files:

- `backend.py` – defines the LangGraph workflow, tools, and MCP client
- `async_frontend_streamlit.py` – Streamlit UI that interacts with the graph and supports HITL interrupts

---

## 1. Backend: `backend.py`

The backend is responsible for:

1. Defining the **state** used in the LangGraph workflow
2. Configuring the **LLM**
3. Defining **tools** (search, stock actions, MCP tools)
4. Building and compiling the **LangGraph StateGraph**

### 1.1. State definition

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class chat_state(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

The `chat_state` keeps a list of LangChain `BaseMessage` objects. `add_messages` tells LangGraph how to accumulate messages over time.

### 1.2. LLM setup

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5.1", temperature=0.9)
```

The model `gpt-5.1` is used with a slightly creative temperature (0.9).

You must provide your OpenAI API key via environment variables (see **Environment variables** below).

### 1.3. Tools

The backend defines and uses several tools:

1. **Web search tool** (DuckDuckGo)

   ```python
   from langchain_community.tools import DuckDuckGoSearchRun

   search_tools = DuckDuckGoSearchRun(region="us-en")
   ```

2. **Custom stock tools** (with Human-in-the-Loop)

   ```python
   from langgraph.types import interrupt
   from langchain_core.tools import tool
   import requests, os

   @tool
   def purchase_stock(company: str, quantity: int) -> str:
       """Simulates purchasing shares of a company. Use this tool when the user wants to buy stocks."""

       decision = interrupt({
           "type": "approval",
           "message": f"Should I purchase {quantity} shares of {company}?",
       })

       if decision.lower() == "yes":
           return f"I have successfully purchased {quantity} shares of {company}."
       else:
           return f"I decided not to purchase {quantity} shares of {company}."
   ```

   This shows the **Human-in-the-Loop (HITL)** pattern: the tool calls `interrupt(...)` and waits for the human to respond ("yes" / "no"). The frontend is responsible for capturing that answer and resuming the workflow.

   ```python
   @tool
   def get_stock_price(ticker: str) -> dict:
       """Get the latest stock price, including high/low, for a ticker symbol."""
       try:
           url = f"https://api.twelvedata.com/time_series?apikey={os.getenv('STOCK_PRICE_API_KEY')}&symbol={ticker}&interval=1min"
           response = requests.get(url)
           return response.json()
       except Exception as e:
           return {"error": str(e)}
   ```

   This uses the **Twelve Data API** to fetch stock prices.

3. **MCP Client (Model Context Protocol)**

   ```python
   from langchain_mcp_adapters.client import MultiServerMCPClient

   client = MultiServerMCPClient({
       "github": {
           "transport": "http",
           "url": "https://api.githubcopilot.com/mcp/",
           "headers": {"Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}"},
       },
       "filesystem": {
           "transport": "stdio",
           "command": "npx",
           "args": [
               "-y",
               "@modelcontextprotocol/server-filesystem",
               "C:\\PYTHON\\langGraph",  # folder you want to allow
           ],
       },
   })

   github_tool = await client.get_tools()
   all_tools = [search_tools, purchase_stock, get_stock_price, *github_tool]
   llm_with_tools = llm.bind_tools(all_tools)
   ```

   - **`MultiServerMCPClient`** connects to multiple MCP servers:
     - A **GitHub MCP** server (via HTTP) using an auth token
     - A **filesystem MCP** server (via `npx @modelcontextprotocol/server-filesystem`), limited to `C:\PYTHON\langGraph`
   - `client.get_tools()` discovers all tools exposed by these MCP servers.
   - `llm.bind_tools(all_tools)` lets the LLM call all of these tools as needed.

### 1.4. LangGraph workflow

```python
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

async def build_graph():
    graph = StateGraph(chat_state)

    # (MCP client + all_tools setup here)

    async def chat_node(state: chat_state):
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    tools_node = ToolNode(all_tools)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tools_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    checkpointer = InMemorySaver()
    workflow = graph.compile(checkpointer=checkpointer)
    return workflow

async def get_workflow():
    return await build_graph()
```

- `chat_node` is the main LLM node. It receives the message history and returns an AI response.
- `ToolNode` executes tools when the LLM decides to call them.
- `tools_condition` routes between the chat node and the tools node, depending on whether a tool call is present.
- `InMemorySaver` keeps state in memory (per `thread_id`), which the frontend uses.

---

## 2. Frontend: `async_frontend_streamlit.py`

The frontend is an **async Streamlit app** that:

- Initializes the workflow once and keeps it in `st.session_state`
- Stores conversation messages for display
- Handles **HITL interrupts** triggered by the backend (e.g., stock purchase approvals)

### 2.1. Setup and session state

```python
import streamlit as st
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from backend import get_workflow

st.set_page_config(page_title="LangGraph Chatbot", page_icon="🤖")
st.title("LangGraph Chatbot")

if "workflow" not in st.session_state:
    st.session_state.workflow = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "demo-thread-1"

if "frontend_messages" not in st.session_state:
    st.session_state.frontend_messages = []

if "waiting_for_interrupt" not in st.session_state:
    st.session_state.waiting_for_interrupt = False

if st.session_state.workflow is None:
    st.session_state.workflow = asyncio.run(get_workflow())

workflow = st.session_state.workflow
CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}
```

- `thread_id` identifies the conversation thread for LangGraph.
- `frontend_messages` is a simple list used to render the chat.
- `waiting_for_interrupt` tracks whether the graph is waiting for a user approval.

### 2.2. Displaying messages

```python
for msg in st.session_state.frontend_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
```

This replays the conversation history on each rerun.

### 2.3. Handling interrupts (HITL)

```python
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
```

- If the result contains `"__interrupt__"`, the backend is asking for human input (e.g., approval for `purchase_stock`).
- The frontend shows the interrupt message and sets `waiting_for_interrupt = True`.

When the user replies while `waiting_for_interrupt` is `True`, the app **resumes** the workflow:

```python
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
```

- Normal messages call `workflow.ainvoke(initial_state, config=CONFIG)`.
- Approval responses call `workflow.ainvoke(Command(resume=user_input), config=CONFIG)`.

This is how **Human-in-the-Loop** is implemented end-to-end.

---

## 3. Environment variables

You should create a `.env` file (or otherwise set environment variables) with at least:

```bash
OPENAI_API_KEY=your_openai_api_key_here
GITHUB_TOKEN=your_github_mcp_token_here
STOCK_PRICE_API_KEY=your_twelve_data_api_key_here
```

- `OPENAI_API_KEY`: For `ChatOpenAI` (OpenAI LLM)
- `GITHUB_TOKEN`: For the GitHub MCP server (used via `https://api.githubcopilot.com/mcp/`)
- `STOCK_PRICE_API_KEY`: For the Twelve Data stock API

The backend calls `load_dotenv()` to load these from a `.env` file in the working directory.

---

## 4. Requirements

Below is a `requirements.txt` that matches the imports used in these files:

```txt
# Core
python-dotenv
requests

# LangChain / LangGraph / OpenAI
langgraph
langchain-core
langchain-openai
langchain-community
langchain-mcp-adapters

# MCP filesystem server (run via npx)
# Note: this is a Node package, but listed here as a reminder
tqdm  # (optional, if required by your environment)

# Web UI
streamlit
```

> Note: The MCP filesystem server itself (`@modelcontextprotocol/server-filesystem`) is a **Node.js package**, installed via `npm`/`npx`, not via `pip`. Make sure you have Node.js and `npx` installed.

You can copy the block above into a `requirements.txt` file (see next section).

---

## 5. Creating and running the app

1. **Create and activate a virtual environment (optional but recommended)**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # on Windows
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install MCP filesystem server (Node.js)**

   ```bash
   npm install -g @modelcontextprotocol/server-filesystem
   ```

   or let `npx` handle it automatically when the backend runs.

4. **Set up `.env`** with `OPENAI_API_KEY`, `GITHUB_TOKEN`, and `STOCK_PRICE_API_KEY`.

5. **Run the Streamlit app**

   From the `chatbot + Human_in_the_loop` folder:

   ```bash
   streamlit run async_frontend_streamlit.py
   ```

6. Open the URL shown in the terminal (usually `http://localhost:8501`) and start chatting.

---

## 6. Features summary

- **LangGraph-based chatbot**: Uses `StateGraph` with an LLM node and a tool node.
- **Multiple tools**:
  - DuckDuckGo web search
  - `purchase_stock` tool with human approval (HITL)
  - `get_stock_price` tool using the Twelve Data API
  - MCP GitHub tools (via `MultiServerMCPClient`)
  - MCP filesystem tools (read/write within `C:\PYTHON\langGraph`)
- **MCP Client integration**:
  - Uses `langchain_mcp_adapters.client.MultiServerMCPClient`
  - Connects to both HTTP-based and stdio-based MCP servers
- **Human-in-the-Loop (HITL)**:
  - Backend uses `interrupt(...)` in `purchase_stock`
  - Frontend detects `__interrupt__` and asks the user for approval
  - The workflow is resumed with `Command(resume=user_input)`
- **Async Streamlit frontend**:
  - Maintains `thread_id` and conversation state
  - Handles both normal messages and interrupt/approval flows

This README plus the `requirements.txt` should be enough for someone to understand and run your project in simple terms.
