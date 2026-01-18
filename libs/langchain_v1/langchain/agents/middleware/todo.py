"""Planning and task management middleware for agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.runtime import Runtime

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command
from typing_extensions import NotRequired, TypedDict, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    OmitFromInput,
)
from langchain.tools import InjectedToolCallId


class Todo(TypedDict):
    """A single todo item with content and status."""

    content: str
    """The content/description of the todo item."""

    status: Literal["pending", "in_progress", "completed"]
    """The current status of the todo item."""


class PlanningState(AgentState[Any]):
    """State schema for the todo middleware."""

    todos: Annotated[NotRequired[list[Todo]], OmitFromInput]
    """List of todo items for tracking task progress."""


WRITE_TODOS_TOOL_DESCRIPTION = """使用此工具为当前工作会话创建和管理结构化任务清单。这可以帮助你跟踪进度、组织复杂任务，并向用户展示你的工作完整性。

Only use this tool if you think it will be helpful in staying organized. If the user's request is trivial and takes less than 3 steps, it is better to NOT use this tool and just do the task directly.

## 何时使用此工具
在以下场景使用此工具：

1. 复杂的多步骤任务 - 当任务需要 3 个或更多不同步骤/动作时
2. 非简单且复杂的任务 - 需要仔细规划或多次操作的任务
3. 用户明确要求待办清单 - 用户直接要求你使用待办清单时
4. 用户提供多个任务 - 用户提供要做的事项列表（编号或逗号分隔）
5. 计划可能需要根据前几步的结果进行后续修订或更新

## 如何使用此工具
1. 当你开始处理某个任务 - 在开始前将其标记为 in_progress。
2. 当完成某个任务 - 将其标记为 completed，并添加执行过程中发现的新后续任务。
3. 你也可以更新未来任务，例如删除不再需要的任务或添加必要的新任务。不要修改已完成的任务。
4. 你可以一次性对待办清单进行多项更新。例如，完成一个任务时，可以把下一个要开始的任务标记为 in_progress。

## 何时不使用此工具
在以下情况应跳过使用此工具：
1. 只有一个简单直接的任务
2. 任务过于琐碎，跟踪它没有收益
3. 任务可以在少于 3 个简单步骤内完成
4. 任务纯属对话或信息性问题

## 任务状态与管理

1. **任务状态**：使用以下状态跟踪进度：
   - pending：任务尚未开始
   - in_progress：正在进行（若任务彼此无关且可并行，可同时有多个 in_progress）
   - completed：任务已成功完成

2. **任务管理**：
   - 工作过程中实时更新任务状态
   - 完成后立即标记为 completed（不要批量标记）
   - 完成当前任务后再开始新任务
   - 彻底移除不再相关的任务
   - 重要：写入待办清单时，应立即将首个任务（或多个任务）标记为 in_progress。
   - 重要：除非所有任务都已完成，否则应始终保持至少一个 in_progress，以便用户知道你正在处理某项任务。

3. **任务完成要求**：
   - 只有在完全完成后才标记为 completed
   - 如果遇到错误、阻塞或无法完成，保持 in_progress
   - 被阻塞时，创建新任务描述需要解决的问题
   - 以下情况不要标记为 completed：
     - 仍有未解决的问题或错误
     - 工作不完整或只是部分完成
     - 遇到阻塞导致无法完成
     - 缺少必要资源或依赖
     - 未达到质量标准

4. **任务拆分**：
   - 创建具体、可执行的事项
   - 将复杂任务拆解为更小、更易管理的步骤
   - 使用清晰、描述性强的任务名称

主动进行任务管理能体现你的细致与可靠，并确保你完成所有要求。
记住：如果只需要少量工具调用即可完成任务，而且要做的事情很明确，那么最好直接完成，不要调用此工具。"""  # noqa: E501

WRITE_TODOS_SYSTEM_PROMPT = """## `write_todos`

你可以使用 `write_todos` 工具来管理并规划复杂目标。
当目标复杂时使用该工具，确保你跟踪每个必要步骤，并让用户清楚看到你的进度。
该工具非常适合规划复杂目标，以及将大型复杂目标拆分为更小步骤。

关键要求：完成一个步骤后要立即将对应待办标记为 completed，不要等多个步骤完成后再批量标记。
如果目标只需要几个简单步骤，最好直接完成，不要使用此工具。
编写待办会消耗时间和 tokens，请在需要管理复杂多步骤问题时使用，而非简单问题。

## 重要的待办清单使用注意事项
- `write_todos` 工具绝不能在同一轮中并行调用多次。
- 随时修订待办清单是可以的，新信息可能带来新任务或让旧任务不再相关。"""  # noqa: E501


@tool(description=WRITE_TODOS_TOOL_DESCRIPTION)
def write_todos(
    todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command[Any]:
    """Create and manage a structured task list for your current work session."""
    return Command(
        update={
            "todos": todos,
            "messages": [ToolMessage(f"已更新待办清单为 {todos}", tool_call_id=tool_call_id)],
        }
    )


class TodoListMiddleware(AgentMiddleware):
    """Middleware that provides todo list management capabilities to agents.

    This middleware adds a `write_todos` tool that allows agents to create and manage
    structured task lists for complex multi-step operations. It's designed to help
    agents track progress, organize complex tasks, and provide users with visibility
    into task completion status.

    The middleware automatically injects system prompts that guide the agent on when
    and how to use the todo functionality effectively. It also enforces that the
    `write_todos` tool is called at most once per model turn, since the tool replaces
    the entire todo list and parallel calls would create ambiguity about precedence.

    Example:
        ```python
        from langchain.agents.middleware.todo import TodoListMiddleware
        from langchain.agents import create_agent

        agent = create_agent("openai:gpt-4o", middleware=[TodoListMiddleware()])

        # Agent now has access to write_todos tool and todo state tracking
        result = await agent.invoke({"messages": [HumanMessage("Help me refactor my codebase")]})

        print(result["todos"])  # Array of todo items with status tracking
        ```
    """

    state_schema = PlanningState

    def __init__(
        self,
        *,
        system_prompt: str = WRITE_TODOS_SYSTEM_PROMPT,
        tool_description: str = WRITE_TODOS_TOOL_DESCRIPTION,
    ) -> None:
        """Initialize the `TodoListMiddleware` with optional custom prompts.

        Args:
            system_prompt: Custom system prompt to guide the agent on using the todo
                tool.
            tool_description: Custom description for the `write_todos` tool.
        """
        super().__init__()
        self.system_prompt = system_prompt
        self.tool_description = tool_description

        # Dynamically create the write_todos tool with the custom description
        @tool(description=self.tool_description)
        def write_todos(
            todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
        ) -> Command[Any]:
            """Create and manage a structured task list for your current work session."""
            return Command(
                update={
                    "todos": todos,
                    "messages": [
                        ToolMessage(f"已更新待办清单为 {todos}", tool_call_id=tool_call_id)
                    ],
                }
            )

        self.tools = [write_todos]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Update the system message to include the todo system prompt.

        Args:
            request: Model request to execute (includes state and runtime).
            handler: Async callback that executes the model request and returns
                `ModelResponse`.

        Returns:
            The model call result.
        """
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Update the system message to include the todo system prompt.

        Args:
            request: Model request to execute (includes state and runtime).
            handler: Async callback that executes the model request and returns
                `ModelResponse`.

        Returns:
            The model call result.
        """
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return await handler(request.override(system_message=new_system_message))

    @override
    def after_model(self, state: AgentState[Any], runtime: Runtime) -> dict[str, Any] | None:
        """Check for parallel write_todos tool calls and return errors if detected.

        The todo list is designed to be updated at most once per model turn. Since
        the `write_todos` tool replaces the entire todo list with each call, making
        multiple parallel calls would create ambiguity about which update should take
        precedence. This method prevents such conflicts by rejecting any response that
        contains multiple write_todos tool calls.

        Args:
            state: The current agent state containing messages.
            runtime: The LangGraph runtime instance.

        Returns:
            A dict containing error ToolMessages for each write_todos call if multiple
            parallel calls are detected, otherwise None to allow normal execution.
        """
        messages = state["messages"]
        if not messages:
            return None

        last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
        if not last_ai_msg or not last_ai_msg.tool_calls:
            return None

        # Count write_todos tool calls
        write_todos_calls = [tc for tc in last_ai_msg.tool_calls if tc["name"] == "write_todos"]

        if len(write_todos_calls) > 1:
            # Create error tool messages for all write_todos calls
            error_messages = [
                ToolMessage(
                    content=(
                        "错误：`write_todos` 工具不应在同一轮并行调用多次。"
                        "请在每次模型调用中仅调用一次以更新待办清单。"
                    ),
                    tool_call_id=tc["id"],
                    status="error",
                )
                for tc in write_todos_calls
            ]

            # Keep the tool calls in the AI message but return error messages
            # This follows the same pattern as HumanInTheLoopMiddleware
            return {"messages": error_messages}

        return None

    @override
    async def aafter_model(self, state: AgentState[Any], runtime: Runtime) -> dict[str, Any] | None:
        """Check for parallel write_todos tool calls and return errors if detected.

        Async version of `after_model`. The todo list is designed to be updated at
        most once per model turn. Since the `write_todos` tool replaces the entire
        todo list with each call, making multiple parallel calls would create ambiguity
        about which update should take precedence. This method prevents such conflicts
        by rejecting any response that contains multiple write_todos tool calls.

        Args:
            state: The current agent state containing messages.
            runtime: The LangGraph runtime instance.

        Returns:
            A dict containing error ToolMessages for each write_todos call if multiple
            parallel calls are detected, otherwise None to allow normal execution.
        """
        return self.after_model(state, runtime)
