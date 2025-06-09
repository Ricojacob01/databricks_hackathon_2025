# Databricks notebook source
# MAGIC %md
# MAGIC #Tool-calling Agent
# MAGIC
# MAGIC This is an auto-generated notebook created by an AI Playground export.
# MAGIC
# MAGIC This notebook uses [Mosaic AI Agent Framework](https://docs.databricks.com/generative-ai/agent-framework/build-genai-apps.html) to recreate your agent from the AI Playground. It  demonstrates how to develop, manually test, evaluate, log, and deploy a tool-calling agent in LangGraph.
# MAGIC
# MAGIC The agent code implements [MLflow's ChatAgent](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ChatAgent) interface, a Databricks-recommended open-source standard that simplifies authoring multi-turn conversational agents, and is fully compatible with Mosaic AI agent framework functionality.
# MAGIC
# MAGIC  **_NOTE:_**  This notebook uses LangChain, but AI Agent Framework is compatible with any agent authoring framework, including LlamaIndex or pure Python agents written with the OpenAI SDK.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - Address all `TODO`s in this notebook.

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow langchain langgraph==0.3.4 databricks-langchain pydantic databricks-agents unitycatalog-langchain[databricks] uv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Define the agent in code
# MAGIC Below we define our agent code in a single cell, enabling us to easily write it to a local Python file for subsequent logging and deployment using the `%%writefile` magic command.
# MAGIC
# MAGIC For more examples of tools to add to your agent, see [docs](https://docs.databricks.com/generative-ai/agent-framework/agent-tool.html).

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC from typing import Any, Generator, Optional, Sequence, Union
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks_langchain import (
# MAGIC     ChatDatabricks,
# MAGIC     VectorSearchRetrieverTool,
# MAGIC     DatabricksFunctionClient,
# MAGIC     UCFunctionToolkit,
# MAGIC     set_uc_function_client,
# MAGIC )
# MAGIC from langchain_core.language_models import LanguageModelLike
# MAGIC from langchain_core.runnables import RunnableConfig, RunnableLambda
# MAGIC from langchain_core.tools import BaseTool
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.graph import CompiledGraph
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC from langgraph.prebuilt.tool_node import ToolNode
# MAGIC from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC from mlflow.types.agent import (
# MAGIC     ChatAgentChunk,
# MAGIC     ChatAgentMessage,
# MAGIC     ChatAgentResponse,
# MAGIC     ChatContext,
# MAGIC )
# MAGIC
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC client = DatabricksFunctionClient()
# MAGIC set_uc_function_client(client)
# MAGIC
# MAGIC ############################################
# MAGIC # Define your LLM endpoint and system prompt
# MAGIC ############################################
# MAGIC LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC system_prompt = """your a chat bot that helps users find the specific plan type based on their special needs """
# MAGIC
# MAGIC ###############################################################################
# MAGIC ## Define tools for your agent, enabling it to retrieve data or take actions
# MAGIC ## beyond text generation
# MAGIC ## To create and see usage examples of more tools, see
# MAGIC ## https://docs.databricks.com/generative-ai/agent-framework/agent-tool.html
# MAGIC ###############################################################################
# MAGIC tools = []
# MAGIC
# MAGIC # You can use UDFs in Unity Catalog as agent tools
# MAGIC uc_tool_names = ["workspace.wellness.*"]
# MAGIC uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
# MAGIC tools.extend(uc_toolkit.tools)
# MAGIC
# MAGIC
# MAGIC # # (Optional) Use Databricks vector search indexes as tools
# MAGIC # # See https://docs.databricks.com/generative-ai/agent-framework/unstructured-retrieval-tools.html
# MAGIC # # for details
# MAGIC #
# MAGIC # # TODO: Add vector search indexes as tools or delete this block
# MAGIC # vector_search_tools = [
# MAGIC #         VectorSearchRetrieverTool(
# MAGIC #         index_name="",
# MAGIC #         # filters="..."
# MAGIC #     )
# MAGIC # ]
# MAGIC # tools.extend(vector_search_tools)
# MAGIC
# MAGIC
# MAGIC #####################
# MAGIC ## Define agent logic
# MAGIC #####################
# MAGIC
# MAGIC
# MAGIC def create_tool_calling_agent(
# MAGIC     model: LanguageModelLike,
# MAGIC     tools: Union[Sequence[BaseTool], ToolNode],
# MAGIC     system_prompt: Optional[str] = None,
# MAGIC ) -> CompiledGraph:
# MAGIC     model = model.bind_tools(tools)
# MAGIC
# MAGIC     # Define the function that determines which node to go to
# MAGIC     def should_continue(state: ChatAgentState):
# MAGIC         messages = state["messages"]
# MAGIC         last_message = messages[-1]
# MAGIC         # If there are function calls, continue. else, end
# MAGIC         if last_message.get("tool_calls"):
# MAGIC             return "continue"
# MAGIC         else:
# MAGIC             return "end"
# MAGIC
# MAGIC     if system_prompt:
# MAGIC         preprocessor = RunnableLambda(
# MAGIC             lambda state: [{"role": "system", "content": system_prompt}]
# MAGIC             + state["messages"]
# MAGIC         )
# MAGIC     else:
# MAGIC         preprocessor = RunnableLambda(lambda state: state["messages"])
# MAGIC     model_runnable = preprocessor | model
# MAGIC
# MAGIC     def call_model(
# MAGIC         state: ChatAgentState,
# MAGIC         config: RunnableConfig,
# MAGIC     ):
# MAGIC         response = model_runnable.invoke(state, config)
# MAGIC
# MAGIC         return {"messages": [response]}
# MAGIC
# MAGIC     workflow = StateGraph(ChatAgentState)
# MAGIC
# MAGIC     workflow.add_node("agent", RunnableLambda(call_model))
# MAGIC     workflow.add_node("tools", ChatAgentToolNode(tools))
# MAGIC
# MAGIC     workflow.set_entry_point("agent")
# MAGIC     workflow.add_conditional_edges(
# MAGIC         "agent",
# MAGIC         should_continue,
# MAGIC         {
# MAGIC             "continue": "tools",
# MAGIC             "end": END,
# MAGIC         },
# MAGIC     )
# MAGIC     workflow.add_edge("tools", "agent")
# MAGIC
# MAGIC     return workflow.compile()
# MAGIC
# MAGIC
# MAGIC class LangGraphChatAgent(ChatAgent):
# MAGIC     def __init__(self, agent: CompiledStateGraph):
# MAGIC         self.agent = agent
# MAGIC
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC
# MAGIC         messages = []
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 messages.extend(
# MAGIC                     ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
# MAGIC                 )
# MAGIC         return ChatAgentResponse(messages=messages)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> Generator[ChatAgentChunk, None, None]:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 yield from (
# MAGIC                     ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
# MAGIC                 )
# MAGIC
# MAGIC
# MAGIC # Create the agent object, and specify it as the agent object to use when
# MAGIC # loading the agent back for inference via mlflow.models.set_model()
# MAGIC agent = create_tool_calling_agent(llm, tools, system_prompt)
# MAGIC AGENT = LangGraphChatAgent(agent)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC
# MAGIC Interact with the agent to test its output. Since this notebook called `mlflow.langchain.autolog()` you can view the trace for each step the agent takes.
# MAGIC
# MAGIC Replace this placeholder input with an appropriate domain-specific example for your agent.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from agent import AGENT

AGENT.predict({"messages": [{"role": "user", "content": "Hello!"}]})

# COMMAND ----------

for event in AGENT.predict_stream(
    {"messages": [{"role": "user", "content": "What is 5+5 in python"}]}
):
    print(event, "-----------\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the `agent` as an MLflow model
# MAGIC Determine Databricks resources to specify for automatic auth passthrough at deployment time
# MAGIC - **TODO**: If your Unity Catalog Function queries a [vector search index](https://docs.databricks.com/generative-ai/agent-framework/unstructured-retrieval-tools.html) or leverages [external functions](https://docs.databricks.com/generative-ai/agent-framework/external-connection-tools.html), you need to include the dependent vector search index and UC connection objects, respectively, as resources. See [docs](https://docs.databricks.com/generative-ai/agent-framework/log-agent.html#specify-resources-for-automatic-authentication-passthrough) for more details.
# MAGIC
# MAGIC Log the agent as code from the `agent.py` file. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).

# COMMAND ----------

# Determine Databricks resources to specify for automatic auth passthrough at deployment time
import mlflow
from agent import tools, LLM_ENDPOINT_NAME
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]
for tool in tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        # TODO: If the UC function includes dependencies like external connection or vector search, please include them manually.
        # See the TODO in the markdown above for more information.
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "tell me plans that institutional"
        }
    ]
}

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model="agent.py",
        input_example=input_example,
        resources=resources,
        extra_pip_requirements=[
            "databricks-connect"
        ]
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with [Agent Evaluation](https://docs.databricks.com/generative-ai/agent-evaluation/index.html)
# MAGIC
# MAGIC You can edit the requests or expected responses in your evaluation dataset and run evaluation as you iterate your agent, leveraging mlflow to track the computed quality metrics.
# MAGIC
# MAGIC To evaluate your tool calls, try adding [custom metrics](https://docs.databricks.com/generative-ai/agent-evaluation/custom-metrics.html#evaluating-tool-calls).

# COMMAND ----------

import pandas as pd

eval_examples = [
    {
        "request": {
            "messages": [
                {
                    "role": "system",
                    "content": "your a chat bot that helps users find the specific plan type based on their special needs "
                },
                {
                    "role": "user",
                    "content": "tell me plans that institutional"
                }
            ]
        },
        "expected_response": None
    },
    {
        "request": {
            "messages": [
                {
                    "role": "system",
                    "content": "your a chat bot that helps users find the specific plan type based on their special needs "
                },
                {
                    "role": "user",
                    "content": "tell me plans that institutional"
                },
                {
                    "role": "assistant",
                    "content": "The user is asking for plans that are institutional, so I need to look up the plan type to provide the relevant information.\n\n<uc_function_call>{\"id\":\"12IX3XeI36OhbRh2jJRd\",\"name\":\"workspace.wellness.health_plan_look_up\",\"arguments\":\"{\\\"plan_type\\\":\\\"institutional\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"12IX3XeI36OhbRh2jJRd\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"[]\\\"]]}\"}</uc_function_result>\n\nIt seems that there are no plans available that are specifically institutional. If you're looking for plans that cater to special needs, I can try to help you with that. Could you please provide more information about the type of special needs you're looking for? For example, are you looking for plans that support physical disabilities, mental health, or something else? This will help me provide a more tailored response."
                },
                {
                    "role": "user",
                    "content": "what about Institutional"
                }
            ]
        },
        "expected_response": None
    },
    {
        "request": {
            "messages": [
                {
                    "role": "system",
                    "content": "your a chat bot that helps users find the specific plan type based on their special needs "
                },
                {
                    "role": "user",
                    "content": "tell me plans that institutional"
                },
                {
                    "role": "assistant",
                    "content": "The user is asking for plans that are institutional, so I need to look up the plan type to provide the relevant information.\n\n<uc_function_call>{\"id\":\"12IX3XeI36OhbRh2jJRd\",\"name\":\"workspace.wellness.health_plan_look_up\",\"arguments\":\"{\\\"plan_type\\\":\\\"institutional\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"12IX3XeI36OhbRh2jJRd\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"[]\\\"]]}\"}</uc_function_result>\n\nIt seems that there are no plans available that are specifically institutional. If you're looking for plans that cater to special needs, I can try to help you with that. Could you please provide more information about the type of special needs you're looking for? For example, are you looking for plans that support physical disabilities, mental health, or something else? This will help me provide a more tailored response."
                },
                {
                    "role": "user",
                    "content": "what about Institutional"
                },
                {
                    "role": "assistant",
                    "content": "The user is asking about Institutional plans, so I need to look up the plans for this type.\n\n<uc_function_call>{\"id\":\"ojAm53y9HqjeAEglVWJU\",\"name\":\"workspace.wellness.health_plan_look_up\",\"arguments\":\"{\\\"plan_type\\\":\\\"Institutional\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"ojAm53y9HqjeAEglVWJU\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"[]\\\"]]}\"}</uc_function_result>\n\nIt seems that there are still no plans available that match the \"Institutional\" type. If you're looking for plans that cater to specific needs, such as adult day care or nursing home care, I can try to help you with that. Could you please provide more information about the type of care or support you're looking for? This will help me provide a more tailored response."
                },
                {
                    "role": "user",
                    "content": "what about Dual-Elgible"
                }
            ]
        },
        "expected_response": None
    },
    {
        "request": {
            "messages": [
                {
                    "role": "system",
                    "content": "your a chat bot that helps users find the specific plan type based on their special needs "
                },
                {
                    "role": "user",
                    "content": "tell me plans that institutional"
                },
                {
                    "role": "assistant",
                    "content": "The user is asking for plans that are institutional, so I need to look up the plan type to provide the relevant information.\n\n<uc_function_call>{\"id\":\"12IX3XeI36OhbRh2jJRd\",\"name\":\"workspace.wellness.health_plan_look_up\",\"arguments\":\"{\\\"plan_type\\\":\\\"institutional\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"12IX3XeI36OhbRh2jJRd\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"[]\\\"]]}\"}</uc_function_result>\n\nIt seems that there are no plans available that are specifically institutional. If you're looking for plans that cater to special needs, I can try to help you with that. Could you please provide more information about the type of special needs you're looking for? For example, are you looking for plans that support physical disabilities, mental health, or something else? This will help me provide a more tailored response."
                },
                {
                    "role": "user",
                    "content": "what about Institutional"
                },
                {
                    "role": "assistant",
                    "content": "The user is asking about Institutional plans, so I need to look up the plans for this type.\n\n<uc_function_call>{\"id\":\"ojAm53y9HqjeAEglVWJU\",\"name\":\"workspace.wellness.health_plan_look_up\",\"arguments\":\"{\\\"plan_type\\\":\\\"Institutional\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"ojAm53y9HqjeAEglVWJU\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"[]\\\"]]}\"}</uc_function_result>\n\nIt seems that there are still no plans available that match the \"Institutional\" type. If you're looking for plans that cater to specific needs, such as adult day care or nursing home care, I can try to help you with that. Could you please provide more information about the type of care or support you're looking for? This will help me provide a more tailored response."
                },
                {
                    "role": "user",
                    "content": "what about Dual-Elgible"
                },
                {
                    "role": "assistant",
                    "content": "The user is looking for plans that are Dual-Eligible, which suggests they are looking for plans that cater to individuals who are eligible for both Medicare and Medicaid. This requires a specific type of plan that meets the needs of dual-eligible individuals.\n\n<uc_function_call>{\"id\":\"crAr8RSARmEZOa3vV0Hf\",\"name\":\"workspace.wellness.health_plan_look_up\",\"arguments\":\"{\\\"plan_type\\\":\\\"Dual-Eligible\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"crAr8RSARmEZOa3vV0Hf\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"[]\\\"]]}\"}</uc_function_result>\n\nIt seems that there are no plans available that match the \"Dual-Eligible\" type. Dual-Eligible plans are typically designed for individuals who are eligible for both Medicare and Medicaid. If you're looking for plans that cater to this specific population, I can try to help you explore other options. Alternatively, I can try to help you find plans that support other types of special needs. Could you please let me know how I can assist you further?"
                },
                {
                    "role": "user",
                    "content": "I updated the function try agian"
                }
            ]
        },
        "expected_response": None
    },
    {
        "request": {
            "messages": [
                {
                    "role": "system",
                    "content": "your a chat bot that helps users find the specific plan type based on their special needs "
                },
                {
                    "role": "user",
                    "content": "tell me plans that institutional"
                },
                {
                    "role": "assistant",
                    "content": "The user is asking for plans that are institutional, so I need to look up the plan type to provide the relevant information.\n\n<uc_function_call>{\"id\":\"12IX3XeI36OhbRh2jJRd\",\"name\":\"workspace.wellness.health_plan_look_up\",\"arguments\":\"{\\\"plan_type\\\":\\\"institutional\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"12IX3XeI36OhbRh2jJRd\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"[]\\\"]]}\"}</uc_function_result>\n\nIt seems that there are no plans available that are specifically institutional. If you're looking for plans that cater to special needs, I can try to help you with that. Could you please provide more information about the type of special needs you're looking for? For example, are you looking for plans that support physical disabilities, mental health, or something else? This will help me provide a more tailored response."
                },
                {
                    "role": "user",
                    "content": "what about Institutional"
                },
                {
                    "role": "assistant",
                    "content": "The user is asking about Institutional plans, so I need to look up the plans for this type.\n\n<uc_function_call>{\"id\":\"ojAm53y9HqjeAEglVWJU\",\"name\":\"workspace.wellness.health_plan_look_up\",\"arguments\":\"{\\\"plan_type\\\":\\\"Institutional\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"ojAm53y9HqjeAEglVWJU\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"[]\\\"]]}\"}</uc_function_result>\n\nIt seems that there are still no plans available that match the \"Institutional\" type. If you're looking for plans that cater to specific needs, such as adult day care or nursing home care, I can try to help you with that. Could you please provide more information about the type of care or support you're looking for? This will help me provide a more tailored response."
                },
                {
                    "role": "user",
                    "content": "what about Dual-Elgible"
                },
                {
                    "role": "assistant",
                    "content": "The user is looking for plans that are Dual-Eligible, which suggests they are looking for plans that cater to individuals who are eligible for both Medicare and Medicaid. This requires a specific type of plan that meets the needs of dual-eligible individuals.\n\n<uc_function_call>{\"id\":\"crAr8RSARmEZOa3vV0Hf\",\"name\":\"workspace.wellness.health_plan_look_up\",\"arguments\":\"{\\\"plan_type\\\":\\\"Dual-Eligible\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"crAr8RSARmEZOa3vV0Hf\",\"content\":\"{\\\"is_truncated\\\":false,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"[]\\\"]]}\"}</uc_function_result>\n\nIt seems that there are no plans available that match the \"Dual-Eligible\" type. Dual-Eligible plans are typically designed for individuals who are eligible for both Medicare and Medicaid. If you're looking for plans that cater to this specific population, I can try to help you explore other options. Alternatively, I can try to help you find plans that support other types of special needs. Could you please let me know how I can assist you further?"
                },
                {
                    "role": "user",
                    "content": "I updated the function try agian"
                },
                {
                    "role": "assistant",
                    "content": "To find plans that are Dual-Eligible, we need to look up the plan type.\n\n<uc_function_call>{\"id\":\"MCNCN2sXBmXEwRpLWnlx\",\"name\":\"workspace.wellness.health_plan_look_up\",\"arguments\":\"{\\\"plan_type\\\":\\\"Dual-Eligible\\\"}\"}</uc_function_call>\n\n<uc_function_result>{\"id\":\"MCNCN2sXBmXEwRpLWnlx\",\"content\":\"{\\\"is_truncated\\\":true,\\\"columns\\\":[\\\"output\\\"],\\\"rows\\\":[[\\\"[Plan_Name: VIVA Medicare Extra Value (HMO D-SNP), Plan_ID: 012, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D001 (HMO-POS D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Access Open (PPO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D002 (PPO D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-V002 (HMO-POS D-SNP), Plan_ID: 044, plan_type Dual-Eligible, , Plan_Name: Devoted DUAL PLUS Alabama (HMO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: Devoted DUAL Alabama (HMO D-SNP), Plan_ID: 004, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Preferred (HMO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select (HMO D-SNP), Plan_ID: 010, plan_type Dual-Eligible, , Plan_Name: Simpra Advantage (PPO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: Cigna TotalCare (HMO D-SNP), Plan_ID: 055, plan_type Dual-Eligible, , Plan_Name: Cigna TotalCare Plus (HMO D-SNP), Plan_ID: 063, plan_type Dual-Eligible, , Plan_Name: HumanaChoice SNP-DE H5216-370 (PPO D-SNP), Plan_ID: 370, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Choice (PPO D-SNP), Plan_ID: 462, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select Choice (PPO D-SNP), Plan_ID: 463, plan_type Dual-Eligible, , Plan_Name: Humana Gold Plus SNP-DE H5619-093 (HMO D-SNP), Plan_ID: 093, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Liberty (HMO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Access (HMO D-SNP), Plan_ID: 004, plan_type Dual-Eligible, , Plan_Name: Wellcare All Dual Assure (HMO D-SNP), Plan_ID: 007, plan_type Dual-Eligible, , Plan_Name: VIVA Medicare Extra Value (HMO D-SNP), Plan_ID: 012, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D001 (HMO-POS D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Access Open (PPO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D002 (PPO D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-V002 (HMO-POS D-SNP), Plan_ID: 044, plan_type Dual-Eligible, , Plan_Name: Devoted DUAL PLUS Alabama (HMO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: Devoted DUAL Alabama (HMO D-SNP), Plan_ID: 004, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Preferred (HMO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select (HMO D-SNP), Plan_ID: 010, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Signature (HMO D-SNP), Plan_ID: 018, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Signature Select (HMO D-SNP), Plan_ID: 019, plan_type Dual-Eligible, , Plan_Name: Simpra Advantage (PPO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: Cigna TotalCare (HMO D-SNP), Plan_ID: 055, plan_type Dual-Eligible, , Plan_Name: Cigna TotalCare Plus (HMO D-SNP), Plan_ID: 063, plan_type Dual-Eligible, , Plan_Name: HumanaChoice SNP-DE H5216-370 (PPO D-SNP), Plan_ID: 370, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Choice (PPO D-SNP), Plan_ID: 462, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select Choice (PPO D-SNP), Plan_ID: 463, plan_type Dual-Eligible, , Plan_Name: Humana Gold Plus SNP-DE H5619-093 (HMO D-SNP), Plan_ID: 093, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Liberty (HMO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Access (HMO D-SNP), Plan_ID: 004, plan_type Dual-Eligible, , Plan_Name: Wellcare All Dual Assure (HMO D-SNP), Plan_ID: 007, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D001 (HMO-POS D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-V001 (HMO-POS D-SNP), Plan_ID: 013, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D002 (PPO D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Preferred (HMO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select (HMO D-SNP), Plan_ID: 010, plan_type Dual-Eligible, , Plan_Name: Simpra Advantage (PPO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: HumanaChoice SNP-DE H5216-370 (PPO D-SNP), Plan_ID: 370, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Choice (PPO D-SNP), Plan_ID: 462, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select Choice (PPO D-SNP), Plan_ID: 463, plan_type Dual-Eligible, , Plan_Name: Humana Gold Plus SNP-DE H5619-093 (HMO D-SNP), Plan_ID: 093, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D001 (HMO-POS D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Access Open (PPO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D002 (PPO D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-V002 (HMO-POS D-SNP), Plan_ID: 044, plan_type Dual-Eligible, , Plan_Name: Devoted DUAL PLUS Alabama (HMO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: Devoted DUAL Alabama (HMO D-SNP), Plan_ID: 004, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Preferred (HMO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select (HMO D-SNP), Plan_ID: 010, plan_type Dual-Eligible, , Plan_Name: Simpra Advantage (PPO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: Cigna TotalCare (HMO D-SNP), Plan_ID: 055, plan_type Dual-Eligible, , Plan_Name: Cigna TotalCare Plus (HMO D-SNP), Plan_ID: 063, plan_type Dual-Eligible, , Plan_Name: HumanaChoice SNP-DE H5216-370 (PPO D-SNP), Plan_ID: 370, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Choice (PPO D-SNP), Plan_ID: 462, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select Choice (PPO D-SNP), Plan_ID: 463, plan_type Dual-Eligible, , Plan_Name: Humana Gold Plus SNP-DE H5619-093 (HMO D-SNP), Plan_ID: 093, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Liberty (HMO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Access (HMO D-SNP), Plan_ID: 004, plan_type Dual-Eligible, , Plan_Name: Wellcare All Dual Assure (HMO D-SNP), Plan_ID: 007, plan_type Dual-Eligible, , Plan_Name: VIVA Medicare Extra Value (HMO D-SNP), Plan_ID: 012, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D001 (HMO-POS D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Access Open (PPO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D002 (PPO D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-V002 (HMO-POS D-SNP), Plan_ID: 044, plan_type Dual-Eligible, , Plan_Name: Devoted DUAL PLUS Alabama (HMO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: Devoted DUAL Alabama (HMO D-SNP), Plan_ID: 004, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Preferred (HMO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select (HMO D-SNP), Plan_ID: 010, plan_type Dual-Eligible, , Plan_Name: Simpra Advantage (PPO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: Cigna TotalCare (HMO D-SNP), Plan_ID: 055, plan_type Dual-Eligible, , Plan_Name: Cigna TotalCare Plus (HMO D-SNP), Plan_ID: 063, plan_type Dual-Eligible, , Plan_Name: HumanaChoice SNP-DE H5216-370 (PPO D-SNP), Plan_ID: 370, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Choice (PPO D-SNP), Plan_ID: 462, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select Choice (PPO D-SNP), Plan_ID: 463, plan_type Dual-Eligible, , Plan_Name: Humana Gold Plus SNP-DE H5619-093 (HMO D-SNP), Plan_ID: 093, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Liberty (HMO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Access (HMO D-SNP), Plan_ID: 004, plan_type Dual-Eligible, , Plan_Name: Wellcare All Dual Assure (HMO D-SNP), Plan_ID: 007, plan_type Dual-Eligible, , Plan_Name: VIVA Medicare Extra Value (HMO D-SNP), Plan_ID: 012, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D001 (HMO-POS D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-V001 (HMO-POS D-SNP), Plan_ID: 013, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D002 (PPO D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: Devoted DUAL PLUS Alabama (HMO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: Devoted DUAL Alabama (HMO D-SNP), Plan_ID: 004, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Preferred (HMO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select (HMO D-SNP), Plan_ID: 010, plan_type Dual-Eligible, , Plan_Name: Simpra Advantage (PPO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: HumanaChoice SNP-DE H5216-370 (PPO D-SNP), Plan_ID: 370, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Choice (PPO D-SNP), Plan_ID: 462, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select Choice (PPO D-SNP), Plan_ID: 463, plan_type Dual-Eligible, , Plan_Name: Humana Gold Plus SNP-DE H5619-093 (HMO D-SNP), Plan_ID: 093, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D001 (HMO-POS D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-V001 (HMO-POS D-SNP), Plan_ID: 013, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D002 (PPO D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Preferred (HMO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select (HMO D-SNP), Plan_ID: 010, plan_type Dual-Eligible, , Plan_Name: Simpra Advantage (PPO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: HumanaChoice SNP-DE H5216-370 (PPO D-SNP), Plan_ID: 370, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Choice (PPO D-SNP), Plan_ID: 462, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select Choice (PPO D-SNP), Plan_ID: 463, plan_type Dual-Eligible, , Plan_Name: Humana Gold Plus SNP-DE H5619-093 (HMO D-SNP), Plan_ID: 093, plan_type Dual-Eligible, , Plan_Name: VIVA Medicare Extra Value (HMO D-SNP), Plan_ID: 012, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D001 (HMO-POS D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-V001 (HMO-POS D-SNP), Plan_ID: 013, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Access Open (PPO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D002 (PPO D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Preferred (HMO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select (HMO D-SNP), Plan_ID: 010, plan_type Dual-Eligible, , Plan_Name: Simpra Advantage (PPO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: HumanaChoice SNP-DE H5216-370 (PPO D-SNP), Plan_ID: 370, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Choice (PPO D-SNP), Plan_ID: 462, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select Choice (PPO D-SNP), Plan_ID: 463, plan_type Dual-Eligible, , Plan_Name: Humana Gold Plus SNP-DE H5619-093 (HMO D-SNP), Plan_ID: 093, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Liberty (HMO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Access (HMO D-SNP), Plan_ID: 004, plan_type Dual-Eligible, , Plan_Name: Wellcare All Dual Assure (HMO D-SNP), Plan_ID: 007, plan_type Dual-Eligible, , Plan_Name: VIVA Medicare Extra Value (HMO D-SNP), Plan_ID: 012, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D001 (HMO-POS D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-V001 (HMO-POS D-SNP), Plan_ID: 013, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D002 (PPO D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Preferred (HMO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select (HMO D-SNP), Plan_ID: 010, plan_type Dual-Eligible, , Plan_Name: Simpra Advantage (PPO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: HumanaChoice SNP-DE H5216-370 (PPO D-SNP), Plan_ID: 370, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Choice (PPO D-SNP), Plan_ID: 462, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select Choice (PPO D-SNP), Plan_ID: 463, plan_type Dual-Eligible, , Plan_Name: Humana Gold Plus SNP-DE H5619-093 (HMO D-SNP), Plan_ID: 093, plan_type Dual-Eligible, , Plan_Name: VIVA Medicare Extra Value (HMO D-SNP), Plan_ID: 012, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D001 (HMO-POS D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-V001 (HMO-POS D-SNP), Plan_ID: 013, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Access Open (PPO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D002 (PPO D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Preferred (HMO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select (HMO D-SNP), Plan_ID: 010, plan_type Dual-Eligible, , Plan_Name: Simpra Advantage (PPO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: Cigna TotalCare (HMO D-SNP), Plan_ID: 055, plan_type Dual-Eligible, , Plan_Name: Cigna TotalCare Plus (HMO D-SNP), Plan_ID: 063, plan_type Dual-Eligible, , Plan_Name: HumanaChoice SNP-DE H5216-370 (PPO D-SNP), Plan_ID: 370, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Choice (PPO D-SNP), Plan_ID: 462, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select Choice (PPO D-SNP), Plan_ID: 463, plan_type Dual-Eligible, , Plan_Name: Humana Gold Plus SNP-DE H5619-093 (HMO D-SNP), Plan_ID: 093, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Liberty (HMO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Access (HMO D-SNP), Plan_ID: 004, plan_type Dual-Eligible, , Plan_Name: Wellcare All Dual Assure (HMO D-SNP), Plan_ID: 007, plan_type Dual-Eligible, , Plan_Name: VIVA Medicare Extra Value (HMO D-SNP), Plan_ID: 012, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D001 (HMO-POS D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Access Open (PPO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D002 (PPO D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-V002 (HMO-POS D-SNP), Plan_ID: 044, plan_type Dual-Eligible, , Plan_Name: Devoted DUAL PLUS Alabama (HMO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: Devoted DUAL Alabama (HMO D-SNP), Plan_ID: 004, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Preferred (HMO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select (HMO D-SNP), Plan_ID: 010, plan_type Dual-Eligible, , Plan_Name: Simpra Advantage (PPO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: Cigna TotalCare (HMO D-SNP), Plan_ID: 055, plan_type Dual-Eligible, , Plan_Name: Cigna TotalCare Plus (HMO D-SNP), Plan_ID: 063, plan_type Dual-Eligible, , Plan_Name: HumanaChoice SNP-DE H5216-370 (PPO D-SNP), Plan_ID: 370, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Choice (PPO D-SNP), Plan_ID: 462, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select Choice (PPO D-SNP), Plan_ID: 463, plan_type Dual-Eligible, , Plan_Name: Humana Gold Plus SNP-DE H5619-093 (HMO D-SNP), Plan_ID: 093, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Liberty (HMO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Access (HMO D-SNP), Plan_ID: 004, plan_type Dual-Eligible, , Plan_Name: Wellcare All Dual Assure (HMO D-SNP), Plan_ID: 007, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D001 (HMO-POS D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-V001 (HMO-POS D-SNP), Plan_ID: 013, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Access Open (PPO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D002 (PPO D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Preferred (HMO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select (HMO D-SNP), Plan_ID: 010, plan_type Dual-Eligible, , Plan_Name: Simpra Advantage (PPO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: HumanaChoice SNP-DE H5216-370 (PPO D-SNP), Plan_ID: 370, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Choice (PPO D-SNP), Plan_ID: 462, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select Choice (PPO D-SNP), Plan_ID: 463, plan_type Dual-Eligible, , Plan_Name: Humana Gold Plus SNP-DE H5619-093 (HMO D-SNP), Plan_ID: 093, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Liberty (HMO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Access (HMO D-SNP), Plan_ID: 004, plan_type Dual-Eligible, , Plan_Name: Wellcare All Dual Assure (HMO D-SNP), Plan_ID: 007, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D001 (HMO-POS D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-V001 (HMO-POS D-SNP), Plan_ID: 013, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D002 (PPO D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Preferred (HMO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select (HMO D-SNP), Plan_ID: 010, plan_type Dual-Eligible, , Plan_Name: Simpra Advantage (PPO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: HumanaChoice SNP-DE H5216-370 (PPO D-SNP), Plan_ID: 370, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Choice (PPO D-SNP), Plan_ID: 462, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select Choice (PPO D-SNP), Plan_ID: 463, plan_type Dual-Eligible, , Plan_Name: Humana Gold Plus SNP-DE H5619-093 (HMO D-SNP), Plan_ID: 093, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D001 (HMO-POS D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-V001 (HMO-POS D-SNP), Plan_ID: 013, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Access Open (PPO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D002 (PPO D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: Devoted DUAL PLUS Alabama (HMO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: Devoted DUAL Alabama (HMO D-SNP), Plan_ID: 004, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Preferred (HMO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select (HMO D-SNP), Plan_ID: 010, plan_type Dual-Eligible, , Plan_Name: Simpra Advantage (PPO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: HumanaChoice SNP-DE H5216-370 (PPO D-SNP), Plan_ID: 370, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Choice (PPO D-SNP), Plan_ID: 462, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select Choice (PPO D-SNP), Plan_ID: 463, plan_type Dual-Eligible, , Plan_Name: Humana Gold Plus SNP-DE H5619-093 (HMO D-SNP), Plan_ID: 093, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Liberty (HMO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Access (HMO D-SNP), Plan_ID: 004, plan_type Dual-Eligible, , Plan_Name: Wellcare All Dual Assure (HMO D-SNP), Plan_ID: 007, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D001 (HMO-POS D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-V001 (HMO-POS D-SNP), Plan_ID: 013, plan_type Dual-Eligible, , Plan_Name: Wellcare Dual Access Open (PPO D-SNP), Plan_ID: 003, plan_type Dual-Eligible, , Plan_Name: UHC Dual Complete AL-D002 (PPO D-SNP), Plan_ID: 009, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Preferred (HMO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select (HMO D-SNP), Plan_ID: 010, plan_type Dual-Eligible, , Plan_Name: Simpra Advantage (PPO D-SNP), Plan_ID: 002, plan_type Dual-Eligible, , Plan_Name: HumanaChoice SNP-DE H5216-370 (PPO D-SNP), Plan_ID: 370, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Choice (PPO D-SNP), Plan_ID: 462, plan_type Dual-Eligible, , Plan_Name: Aetna Medicare Dual Select Choice (PPO D-SNP), Plan_ID: 463, plan_type Dual-Eligible, , Plan_Name: Humana Gold Plus SNP-DE H5619-093 (HMO D-SNP), Plan_ID: 093, plan_type Dual-Eligible, , Plan_Name: ...\\\"]]}\"}</uc_function_result>\n\nBased on the output, here are some plans that are available for Dual-Eligible individuals:\n\n1. VIVA Medicare Extra Value (HMO D-SNP) - Plan ID: 012\n2. UHC Dual Complete AL-D001 (HMO-POS D-SNP) - Plan ID: 009\n3. Wellcare Dual Access Open (PPO D-SNP) - Plan ID: 003\n4. UHC Dual Complete AL-D002 (PPO D-SNP) - Plan ID: 009\n5. UHC Dual Complete AL-V002 (HMO-POS D-SNP) - Plan ID: 044\n6. Devoted DUAL PLUS Alabama (HMO D-SNP) - Plan ID: 003\n7. Devoted DUAL Alabama (HMO D-SNP) - Plan ID: 004\n8. Aetna Medicare Dual Preferred (HMO D-SNP) - Plan ID: 003\n9. Aetna Medicare Dual Select (HMO D-SNP) - Plan ID: 010\n10. Simpra Advantage (PPO D-SNP) - Plan ID: 002\n11. Cigna TotalCare (HMO D-SNP) - Plan ID: 055\n12. Cigna TotalCare Plus (HMO D-SNP) - Plan ID: 063\n13. HumanaChoice SNP-DE H5216-370 (PPO D-SNP) - Plan ID: 370\n14. Aetna Medicare Dual Choice (PPO D-SNP) - Plan ID: 462\n15. Aetna Medicare Dual Select Choice (PPO D-SNP) - Plan ID: 463\n16. Humana Gold Plus SNP-DE H5619-093 (HMO D-SNP) - Plan ID: 093\n17. Wellcare Dual Liberty (HMO D-SNP) - Plan ID: 002\n18. Wellcare Dual Access (HMO D-SNP) - Plan ID: 004\n19. Wellcare All Dual Assure (HMO D-SNP) - Plan ID: 007\n\nPlease note that this is not an exhaustive list, as the output is truncated. It's also important to note that plan availability and details may vary depending on the individual's location and other factors. It's always best to consult with a licensed insurance agent or broker to get a comprehensive understanding of the available plans and their details."
                },
                {
                    "role": "user",
                    "content": "now tell all that 4 star rating"
                }
            ]
        },
        "expected_response": None
    }
]

eval_dataset = pd.DataFrame(eval_examples)
display(eval_dataset)


# COMMAND ----------

import mlflow

with mlflow.start_run(run_id=logged_agent_info.run_id):
    eval_results = mlflow.evaluate(
        f"runs:/{logged_agent_info.run_id}/agent",
        data=eval_dataset,  # Your evaluation dataset
        model_type="databricks-agent",  # Enable Mosaic AI Agent Evaluation
    )

# Review the evaluation results in the MLFLow UI (see console output), or access them in place:
display(eval_results.tables['eval_results'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform pre-deployment validation of the agent
# MAGIC Before registering and deploying the agent, we perform pre-deployment checks via the [mlflow.models.predict()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict) API. See [documentation](https://docs.databricks.com/machine-learning/model-serving/model-serving-debug.html#validate-inputs) for details

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"messages": [{"role": "user", "content": "Hello!"}]},
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "workspace"
schema = "wellness"
model_name = "healthcare"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags = {"endpointSource": "playground"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See [docs](https://docs.databricks.com/generative-ai/deploy-agent.html) for details