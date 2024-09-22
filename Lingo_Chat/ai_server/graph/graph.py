import json

from typing import List, Dict, Any, Union, Annotated, Literal
from typing_extensions import TypedDict


from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, AnyMessage

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.base import RunnableSequence
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .utils import (tools,
                    persona_search_llm, search_llm, local_llm, rag_llm,
                    convert_chat_history_format, fix_called_tool_name,)

persona_llms = {'rag_llm': rag_llm,
                'local_llm': local_llm}


### graph setting
class State(TypedDict):
    """
        persona: 선택한 채팅 페르소나
        history: 이전 대화 기록
        messages: chatbot chain 로그 적재
    """
    persona: str
    history: Annotated[list[AnyMessage], add_messages]
    messages: Annotated[list[AnyMessage], add_messages]


def chatbot_search(state: State, config: RunnableConfig):
    """
        대화 내역을 바탕으로 tool을 호출하여 검색을 진행할 지 결정하고, 답변을 생성하는 chatbot
        - 검색이 필요한 경우 tool 호출, 답변을 페르소나 스타일로 생성
        - 검색이 필요하지 않은 경우, 답변을 그냥 생성 -> 추후 답변을 생성하지 않고 바로 반환하도록 수정 필요(next pr)
    """
    def _check_overload_and_get_return(llm: RunnableSequence,
                                       message: Union[Dict[str, AnyMessage], List[AnyMessage]],
                                       config: RunnableConfig) -> Union[ToolMessage, AIMessage]:
        """
            fix bug[#14]: overload error(gemini api error)
            if happens, just retry to call the llm. Assume one retry is enough.
        """
        return_messages = llm.invoke(message, config)
        if not isinstance(return_messages, (ToolMessage, AIMessage)):
            return_messages = llm.invoke(message, config)   # call again
        print(f"\n>> [chatbot search] invoke return: {return_messages}\n>> type: {type(return_messages)}\n")
        return return_messages

    if_searched = False
    for idx, message in enumerate(state['messages'][::-1]):
        try:
            if idx == 0 and type(message) == ToolMessage and 'url' in message.content:
                if_searched = True
                break
        except Exception as e:
            print(f">> [chatbot search] unexpected error: {e}", end='')
            continue
    
    if if_searched: # 검색된 결과라면 답변을 페르소나에 맞게 재 생성(요약)
        result = {'messages': state['messages']}
        # print(f"\n>> [chatbot search, searched] invoke message: {result}\n\n")
        return {"messages": [_check_overload_and_get_return(persona_search_llm, result, config)]}
    else:
        result = state['messages']
        return {"messages": [_check_overload_and_get_return(search_llm, result, config)]}
        

async def chatbot_chat(state: State, config: RunnableConfig):
    """
        Chatbot Chat 함수는 다음과 같은 순서로 동작한다.
        - chatbot search 가 호출되었을 경우 해당 결과를 받아서 다시 답변을 생성
        - chatbot search 가 호출되지 않았을 경우, 사용자의 입력만을 추출해서 답변을 생성
        
        Chatbot Chat 입력 포맷:
            '''
            [HumanMessage(), AIMessage(), HumanMessage(), ...]  ### 대화 history
            string content      ### 검색 결과(Retrieved
            '''
        현 상태: string content 는 사용되고 있지 않다.
    """
    result = state['history']     # 대화 히스토리를 사용하여 최종 답변 생성
    if_searched = False
    searched_contents = ""
    
    for idx, message in enumerate(state['messages']):
        # print(f">> [chatbot_chat] {idx}th message: {message}\n\n")
        if type(message) == HumanMessage:
            result.append(message)
        elif type(message) == AIMessage and any(model in message.response_metadata.get('model_name', 'Gemini') for model in ['llama3', 'llama-3']):
            result.append(message)
            
        if idx == len(state['messages'])-2 and 'url' in message.content:    # 검색 결과가 있는 경우
            if_searched = True
        
        if if_searched and idx == len(state['messages'])-1:
            searched_contents = message.content
            
    if if_searched:
        result = {'messages': convert_chat_history_format(result, state['persona']),
                  'context': searched_contents}
        
        response = await rag_llm.ainvoke(result, config)
        if_searched = False
    else:
        result = {'messages': [convert_chat_history_format(result, state['persona'])]}
        response = await local_llm.ainvoke(result, config)
    
    # print(f"\n\n >> chatbot_chat final input: \n>> {result}\n>> type: {type(result)}\n\n")    
    return {"messages": response}


class BasicToolNode:
    """
        A node that runs the tools requested in the last AIMessage.
    """

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}
        # print(f"self.tools_by_name: {self.tools_by_name}\n\n")

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        
        outputs = []
        
        for tool_call in message.tool_calls:
            # 레벤슈타인 거리를 통해 called tool 이름을 수정
            called_tool_name = fix_called_tool_name(tool_call['name'])
            
            # 해당 idx 로 tool 호출
            tool_result = self.tools_by_name[called_tool_name].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


def route_tools(
    state: State,
) -> Literal["tools", "__end__"]:
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__next__"

   
def init_graph():
    """
        vllm(ChatOpenAI)와 langgraph를 연동하여 graph를 만들기 위한 initialization 코드입니다.
    """
    # 대화 내용 기억을 위한 메모리설정
    # memory = AsyncSqliteSaver.from_conn_string(":memory:")

    # Langgraph 설정
    tool_node = BasicToolNode(tools=tools)
    
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot search", chatbot_search)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("chatbot chat", chatbot_chat)
    
    graph_builder.add_conditional_edges(
        "chatbot search",
        route_tools,
        {"tools": "tools", "__next__": "chatbot chat"},
    )
    graph_builder.add_edge("tools", "chatbot search")
    graph_builder.add_edge(START, "chatbot search")
    graph_builder.add_edge("chatbot chat", END)
    # new version
    
    # graph = graph_builder.compile(checkpointer=memory)    
    graph = graph_builder.compile()
    config = {"configurable": {"thread_id": "0"}}   # 이 thread 가 없다면 애초에 state 에 예전 기록이 남질 않는다.
    
    print(f"\n\n>> Graph initialized successfully.\n\n")
    return graph, config

chat_graph, config = init_graph()