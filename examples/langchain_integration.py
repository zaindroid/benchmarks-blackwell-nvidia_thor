"""LangChain integration — expose ThorMCP tools to a LangChain agent.

Requires: ``pip install langchain langchain-mcp langchain-openai``.

The MCP toolkit loads tools from the running ``thor-mcp`` server and the
agent can then run benchmarks, compare results and track experiments
from natural language.
"""

# import os
#
# from langchain_mcp import MCPToolkit
# from langchain.agents import create_openai_tools_agent
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
#
# toolkit = MCPToolkit.from_server_command(
#     command="thor-mcp",
#     args=["--stdio"],
# )
#
# llm = ChatOpenAI(model="gpt-4")
# tools = toolkit.get_tools()
#
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are an AI assistant that benchmarks and optimizes "
#                "models on NVIDIA Thor. Always provide detailed analysis."),
#     ("human", "{input}"),
#     ("placeholder", "{agent_scratchpad}"),
# ])
#
# agent = create_openai_tools_agent(llm, tools, prompt)
#
# # Example queries:
# # "What models have been benchmarked on Thor?"
# # "Optimize YOLOv8 for real-time detection on Thor"
# # "Compare power efficiency of different quantization methods"

print(__doc__)
