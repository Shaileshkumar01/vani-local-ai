# LangChain & LangGraph Complete Architecture & Agent Automation Knowledge Base

## 1. What is LangChain?
LangChain is a modular framework for building applications powered by Large Language Models (LLMs).
Key components of LangChain:
- **Chat Models (`ChatOllama`, `ChatOpenAI`, `ChatAnthropic`)**: Standardized interface (`.invoke()`, `.stream()`, `.bind_tools()`) for interacting with local or cloud LLMs.
- **LCEL (LangChain Expression Language)**: Declarative pipe syntax (`chain = prompt | llm | StrOutputParser()`) supporting streaming, async execution, fallbacks, and parallel execution (`RunnableParallel`).
- **Tools (`@tool` decorator & `StructuredTool`)**: Functions with typed arguments and docstrings that an LLM can invoke to interact with external APIs, databases, file systems, or web search.
- **Retrievers & RAG**: Document loaders (`PyPDFLoader`, `TextLoader`), text splitters (`RecursiveCharacterTextSplitter`), vector stores (`FAISS`, `Chroma`), and retriever tools for grounded question answering.

## 2. What is LangGraph?
LangGraph is an orchestration framework built on top of LangChain for building **stateful, multi-actor, cyclical agent workflows** as directed graphs. Unlike linear chains (DAGs), LangGraph supports **loops, branching, persistence, and human-in-the-loop control**, which are essential for true agentic behavior.

### Core Primitives of LangGraph
1. **`State` (`TypedDict` with `Annotated` reducers)**:
   A shared state schema passed between all nodes in the graph. Usually includes `messages: Annotated[list[AnyMessage], add_messages]` plus task-specific fields (`task`, `plan`, `results`, `next_step`).
2. **`Nodes` (Python functions)**:
   Each node is a function `def node_name(state: AgentState) -> dict:` that reads the current graph state, performs reasoning or tool execution, and returns a state update dictionary.
3. **`Edges` & `Conditional Edges`**:
   - Normal edge (`graph.add_edge("planner", "executor")`): deterministic transition from one node to the next.
   - Conditional edge (`graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})`): dynamically routes execution based on the LLM's output (e.g., whether the LLM emitted a tool call or a final answer).
4. **`START` and `END`**:
   Special virtual nodes marking the entry point and termination point of the `StateGraph`.
5. **`Checkpointers` (`MemorySaver`, `SqliteSaver`, `PostgresSaver`)**:
   Persist graph state at every super-step (`thread_id`), enabling multi-turn conversation memory, time-travel debugging, and human-in-the-loop approval interrupts (`interrupt_before=["tools"]`).

## 3. Multi-Agent & Task Automation Patterns in LangGraph
- **ReAct Agent (`Reason + Act`)**: An LLM node bound to tools (`llm.bind_tools(tools)`) loops with a `ToolNode(tools)` until the task is solved.
- **Plan-and-Execute Agent**: A `Planner` node breaks a complex user goal into ordered subtasks, an `Executor` node runs tools for each step, and a `Verifier` node checks completion or re-plans.
- **Supervisor Multi-Agent System**: A central `Supervisor` node routes subtasks to specialized worker sub-agents (`Researcher`, `Coder`, `Reviewer`, `AutomationWorker`) and aggregates their results.
