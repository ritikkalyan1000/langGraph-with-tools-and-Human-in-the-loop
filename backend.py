from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from langgraph.graph.message import add_messages
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
from langgraph.types import interrupt
import requests
import asyncio


from typing import TypedDict, Annotated

load_dotenv()

# deine states


class chat_state(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatOpenAI(model="gpt-5.1", temperature=0.9)


# tools
search_tools = DuckDuckGoSearchRun(region="us-en")

from langchain_core.tools import tool


@tool
def purchase_stock(company: str, quantity: int) -> str:
    """
    Simulates purchasing shares of a company.
    Use this tool when the user wants to buy stocks.
    """

    decision = interrupt(
        {
            "type": "approval",
            "message": f"Should I purchase {quantity} shares of {company}?",
        }
    )

    if decision.lower() == "yes":
        return f"I have successfully purchased {quantity} shares of {company}."
    else:
        return f"I decided not to purchase {quantity} shares of {company}."


@tool
def get_stock_price(ticker: str) -> dict:
    """
    Get the latest stock price along with high and low values for a ticker symbol.
    Example symbols: AAPL, TSLA, MSFT

    """
    try:

        url = f"https://api.twelvedata.com/time_series?apikey={os.getenv('STOCK_PRICE_API_KEY')}&symbol={ticker}&interval=1min"
        response = requests.get(url)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# define graph


async def build_graph():

    graph = StateGraph(chat_state)

    client = MultiServerMCPClient(
        {
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
        }
    )

    github_tool = await client.get_tools()
    all_tools = [search_tools, purchase_stock, get_stock_price, *github_tool]

    llm_with_tools = llm.bind_tools(all_tools)

    # node 1 functions
    async def chat_node(state: chat_state):
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # tool node
    tools_node = ToolNode(all_tools)

    # add nodes
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tools_node)

    # add edges
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)

    graph.add_edge("tools", "chat_node")
    # graph.add_edge("chat_node", END)

    checkpointer = InMemorySaver()

    workflow = graph.compile(checkpointer=checkpointer)
    return workflow


async def get_workflow():
    return await build_graph()


async def main():
    workflow = await build_graph()


if __name__ == "__main__":
    asyncio.run(main())
