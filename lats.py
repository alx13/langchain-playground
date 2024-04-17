from collections import deque
from typing import (
    Annotated,
    List,
    Optional,
    Sequence,
    TypedDict,
)
import json
import math
import operator
import os

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import (
    MessagesPlaceholder,
    ChatPromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation

from langchain_google_vertexai import ChatVertexAI

from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser

from dotenv import load_dotenv

load_dotenv()


class LLMCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.runs = []

    def on_llm_start(
        self,
        serialized,
        prompts,
        **kwargs,
    ) -> None:
        pass
        # print(prompts)


class Reflection(BaseModel):
    reflections: str = Field(
        description="The critique and reflections on the sufficiency, superfluency,"
        " and general quality of the response"
    )
    score: int = Field(
        description="Score from 0-10 on the quality of the candidate response.",
        # gte=0,
        # lte=10,
    )
    found_solution: bool = Field(
        description="Whether the response has fully solved the question or task."
    )

    def as_message(self):
        return HumanMessage(
            content=f"Reasoning: {self.reflections}\nScore: {self.score}"
        )

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0


class Node:
    def __init__(
        self,
        messages: List[BaseMessage],
        reflection: Reflection,
        parent: Optional["Node"] = None,
    ):
        self.messages = messages
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.reflection = reflection
        self.depth = parent.depth + 1 if parent is not None else 1
        self._is_solved = reflection.found_solution if reflection else False
        if self._is_solved:
            self._mark_tree_as_solved()
        self.backpropagate(reflection.normalized_score)

    def __repr__(self) -> str:
        return (
            f"<Node value={self.value}, visits={self.visits},"
            f" solution={self.messages} reflection={self.reflection}/>"
        )

    @property
    def is_solved(self):
        """If any solutions exist, we can end the search."""
        return self._is_solved

    @property
    def is_terminal(self):
        return not self.children

    @property
    def best_child(self):
        """Select the child with the highest UCT to search next."""
        if not self.children:
            return None
        all_nodes = self._get_all_children()
        return max(all_nodes, key=lambda child: child.upper_confidence_bound())

    @property
    def best_child_score(self):
        """Return the child with the highest value."""
        if not self.children:
            return None
        return max(self.children, key=lambda child: int(child.is_solved) * child.value)

    @property
    def height(self) -> int:
        """Check for how far we've rolled out the tree."""
        if self.children:
            return 1 + max([child.height for child in self.children])
        return 1

    def upper_confidence_bound(self, exploration_weight=1.0):
        """Return the UCT score. This helps balance exploration vs. exploitation of a branch."""
        if self.parent is None:
            raise ValueError("Cannot obtain UCT from root node")
        if self.visits == 0:
            return self.value
        # Encourages exploitation of high-value trajectories
        average_reward = self.value / self.visits
        # Encourages exploration of less-visited trajectories
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return average_reward + exploration_weight * exploration_term

    def backpropagate(self, reward: float):
        """Update the score of this node and its parents."""
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def get_messages(self, include_reflections: bool = True):
        if include_reflections:
            return self.messages + [self.reflection.as_message()]
        return self.messages

    def get_trajectory(self, include_reflections: bool = True) -> List[BaseMessage]:
        """Get messages representing this search branch."""
        messages = []
        node = self
        while node:
            messages.extend(
                node.get_messages(include_reflections=include_reflections)[::-1]
            )
            node = node.parent
        # Reverse the final back-tracked trajectory to return in the correct order
        return messages[::-1]  # root solution, reflection, child 1, ...

    def _get_all_children(self):
        all_nodes = []
        nodes = deque()
        nodes.append(self)
        while nodes:
            node = nodes.popleft()
            all_nodes.extend(node.children)
            for n in node.children:
                nodes.append(n)
        return all_nodes

    def get_best_solution(self):
        """Return the best solution from within the current sub-tree."""
        all_nodes = [self] + self._get_all_children()
        best_node = max(
            all_nodes,
            # We filter out all non-terminal, non-solution trajectories
            key=lambda node: int(node.is_terminal and node.is_solved) * node.value,
        )
        return best_node

    def _mark_tree_as_solved(self):
        parent = self.parent
        while parent:
            parent._is_solved = True
            parent = parent.parent


class TreeState(TypedDict):
    # The full tree
    root: Node
    # The original input
    input: str


def get_tools():
    google_search_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    g_search = GoogleSearchAPIWrapper(
        k=10, google_api_key=google_search_api_key, google_cse_id=google_cse_id
    )

    @tool
    def search(
        question: str,
    ):
        """
        Search tool is used to search for questions.
        Useful for when you need to answer questions or visit websites.
        You should ask targeted questions.
        """
        answers: List[str] = []
        if question is not None:
            answer = g_search.run(question)
            answers.append(answer)
        return answers

    tools = [search]
    return tools


tools = get_tools()
tool_executor = ToolExecutor(tools=tools)

llm = ChatVertexAI(
    model_name="gemini-1.5-pro-preview-0409",
    project=os.getenv("GOOGLE_PROJECT"),
    callbacks=[LLMCallbackHandler()],
)
parser = JsonOutputToolsParser(return_id=True)

llm_with_search = llm.bind_tools(tools=tools)
main_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)

class SearchState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def should_run_tool(state):
    messages = state["messages"]
    last_message = messages[-1]
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    else:
        return "continue"


def call_model(state):
    messages = state["messages"]
    response = llm_with_search.invoke(messages)
    return {"messages": [response]}


def call_tool(state):
    messages = state["messages"]
    last_message = messages[-1]

    parsed = parser.invoke(last_message)
    tool_responses = tool_executor.batch(
        [ToolInvocation(tool=r["type"], tool_input=r["args"]) for r in parsed]
    )
    tool_messages = [
        ToolMessage(
            name=tool_call["type"],
            content=json.dumps(resp),
            tool_call_id=tool_call["id"],
        )
        for resp, tool_call in zip(tool_responses, parsed)
    ]
    return {"messages": tool_messages}


search_app = StateGraph(SearchState)
search_app.add_node("model", call_model)
search_app.set_entry_point("model")
search_app.add_node("tool_call", call_tool)
search_app.add_conditional_edges(
    "model", should_run_tool, {"continue": "tool_call", "end": END}
)
search_app.add_edge("tool_call", "model")
search_graph = search_app.compile()


reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Reflect and grade the assistant response to the user question below.",
        ),
        (
            "user",
            "{input}",
        ),
        MessagesPlaceholder(variable_name="candidate"),
    ]
)

reflection_chain = reflection_prompt | llm.with_structured_output(Reflection)


def generate_initial_response(state: TreeState) -> dict:
    """Generate the initial candidate response."""
    messages = main_prompt.invoke({"input": state["input"]}).to_messages()
    result = search_graph.invoke({"messages": messages})
    candidate = result["messages"]
    reflection = reflection_chain.invoke(
        {
            "input": state["input"],
            "candidate": omit_system_messages(candidate[len(messages) :]),
        }
    )
    root = Node(candidate[len(messages) :], reflection=reflection)
    return {
        **state,
        "root": root,
    }


def omit_system_messages(messages: List[BaseMessage]):
    converted_messages = []
    for message in messages:
        if isinstance(message, SystemMessage):
            continue
        converted_messages.append(message)
    return converted_messages


def expand(state: TreeState, config: RunnableConfig) -> dict:
    """Starting from the "best" node in the tree, generate N candidates for the next step."""
    root = state["root"]
    best_candidate: Node = root.best_child if root.children else root
    messages = best_candidate.get_trajectory()
    # Generate N candidates from the single child candidate

    new_messages = main_prompt.invoke(
        {"input": state["input"], "messages": messages}
    ).to_messages()
    for _ in range(2):
        result = search_graph.invoke({"messages": new_messages})
        candidate = result["messages"]
        reflection = reflection_chain.invoke(
            {
                "input": state["input"],
                "candidate": omit_system_messages(candidate[len(messages) :]),
            }
        )
        child_node = Node(
            candidate[len(new_messages) :], parent=best_candidate, reflection=reflection
        )
        best_candidate.children.append(child_node)
    return state


def should_loop(state: TreeState):
    """Determine whether to continue the tree search."""
    root = state["root"]
    if root.is_solved:
        return "end"
    if root.height > 5:
        return "end"
    return "expand"


builder = StateGraph(TreeState)
builder.add_node("start", generate_initial_response)
builder.set_entry_point("start")
builder.add_node("expand", expand)

builder.add_conditional_edges(
    "start",
    # Either expand/rollout or finish
    should_loop,
    {
        "expand": "expand",
        "end": END,
    },
)
builder.add_conditional_edges(
    "expand",
    # Either continue to rollout or finish
    should_loop,
    {
        "expand": "expand",
        "end": END,
    },
)

graph = builder.compile()

question = "Compare the average exchange rate of EUR to USD in 2008 vs 2023?"
for step in graph.stream({"input": question}):
    for key, value in step.items():
        root = value["root"]
        print("---")
        print(f"Output from node '{key}', height {root.height}:")
        print(value)


solution_node = step["expand"] if "expand" in step else step["start"]
best_solution = solution_node["root"].get_best_solution()
best_trajectory = best_solution.get_trajectory(include_reflections=False)
print("\n\n==========")
print(best_trajectory[-1].content)
