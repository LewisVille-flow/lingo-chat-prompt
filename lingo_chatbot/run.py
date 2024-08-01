from typing import Annotated, List, Union, Literal, TypedDict
from typing_extensions import TypedDict
from utils import call_api_key
from langchain_openai import OpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain.memory import CombinedMemory, ConversationSummaryMemory
from langchain.chains.conversation.base import ExtendedConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import OpenAIEmbeddings
import os
import json

# 추가된 함수 임포트
from llm_1.prompts import create_persona, create_prompt, create_rag_prompt
from llm_2.rag_chain import llm_2_prompts, call_vectordb, rag_chain
from llm_2.vector_db import save_vectordb, load_vectordb
from llm_3.run_llm3 import rag_evaluation_prompt, rag_eval_chain, run_rag_eval_chain

class State(TypedDict):
    messages: Annotated[List[Union[HumanMessage, str]], add_messages]
    activate_RAG: str
    explain: str

# Initialize the StateGraph
graph_builder = StateGraph(State)

# Fetch the OpenAI API key
api_key = call_api_key('OpenAI_API_Key')
embedding_function = OpenAIEmbeddings(api_key=api_key)

# Call LLM for RAG
llm_rag = OpenAI(api_key=api_key)
llm_eval = OpenAI(api_key=api_key)

# Initialize the OpenAI LLM
llm_roleplaying = OpenAI(api_key=api_key)
prompt_decision_rag, prompt_generation_rag = llm_2_prompts()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def validate_json_response(response_str):
    try:
        # Validate and return the JSON object if valid
        return json.loads(response_str)
    except json.JSONDecodeError as e:
        # Log the invalid JSON response for debugging
        print(f"Invalid JSON output: {response_str}")
        raise ValueError("Failed to parse response as JSON") from e

def rag_chatbot(state: State, DB_PATH, file_path, chunk_size, chunk_overlap):
    try:
        if os.path.exists(DB_PATH):
            print(f'Loading VectorDB from {DB_PATH}....')
            load_vectordb(DB_PATH, api_key, embedding_function)
        else:
            print(f'There is no Vector DB in {DB_PATH}. Save vector DB First.')
            save_vectordb(file_path, chunk_size, chunk_overlap, DB_PATH, api_key)
            load_vectordb(DB_PATH, api_key, embedding_function)
        
        retriever = call_vectordb(DB_PATH, embedding_function, chunk_size, chunk_overlap, api_key)
        user_message = state["messages"][-1].content

        print(f'Running RAG chain for the user message: {user_message}')
        rag_chain_instance = rag_chain(retriever, prompt_generation_rag, llm_rag)

        response = rag_chain_instance.invoke(user_message)
        print(f'RAG response: {response}')

        # Validate and process the response
        if isinstance(response, str):
            try:
                response = validate_json_response(response)
            except ValueError:
                # If response is not a valid JSON, set default values
                state["activate_RAG"] = "No"
                state["explain"] = "The response could not be processed as valid JSON."
                return state

        if isinstance(response, dict):
            state["activate_RAG"] = response.get("activate_RAG", "No")
            state["explain"] = response.get("Explain", "")
        else:
            raise ValueError("Response is not a valid JSON string or dictionary")

        print(state["activate_RAG"], state["explain"])
        return state
    except Exception as e:
        print(f'Error in rag_chatbot: {e}')
        raise

def evaluate_rag_output(state: State):
    try:
        evaluation_prompt = rag_evaluation_prompt()
        evaluation_chain = rag_eval_chain(evaluation_prompt, llm_rag)
        user_input = state["messages"][-1].content
        rag_answer = state["explain"]
        evaluation_result = run_rag_eval_chain(evaluation_chain, user_input, rag_answer)
        
        print(f'Evaluation result: {evaluation_result}')

        # Evaluation 결과가 None인 경우 기본값 설정
        if evaluation_result is None:
            state["activate_RAG"] = "No"
        elif "Yes" in evaluation_result:
            state["activate_RAG"] = "Yes"
        else:
            state["activate_RAG"] = "No"

        return state
    except Exception as e:
        print(f'Error in evaluate_rag_output: {e}')
        raise

# Persona와 프롬프트 생성
persona = create_persona()
prompt = create_prompt()
rag_prompt = create_rag_prompt()

# 대화 메모리 설정
char = "Mr.Orbit"
user = "A"

conv_memory = ExtendedConversationBufferMemory(
    llm=llm_roleplaying, 
    memory_key="history", 
    input_key="input", 
    ai_prefix=char, 
    human_prefix=user, 
    extra_variables=["context"]
)
summary_memory = ConversationSummaryMemory(
    llm=llm_roleplaying, 
    memory_key="summary", 
    input_key="input", 
    ai_prefix=char, 
    human_prefix=user
)

combined_memory = CombinedMemory(memories=[conv_memory, summary_memory])

def chatbot_with_rag(state: State):
    print("RAG output processing")
    try:
        context = persona
        user_input = state["messages"][-1].content
        rag_answer = state["explain"]
        
        # 사용자 입력과 RAG 답변을 포함한 문자열 생성
        input_data = f'''User_input: {user_input}
        RAG_answer: {rag_answer}'''

        # 프롬프트에 변수들을 채워서 입력 생성
        chain = ConversationChain(
            llm=llm_roleplaying,
            prompt=rag_prompt,  # create_rag_prompt 함수를 사용하여 생성된 프롬프트
            memory=combined_memory,
            verbose=False
        )

        response = chain.invoke({"input": input_data, "context": context})
        print(f"LLM Response: {response}")
        # Assuming the response is directly the assistant's message
        response_message = response["response"]

        return {"messages": state["messages"] + [response_message]}
    except Exception as e:
        print(f'Error in chatbot_with_rag: {e}')
        raise

def chatbot_without_rag(state: State):
    print("RAG can't print right output. LLM processing right answer.")
    try:
        context = persona
        input_data = state["messages"][-1].content

        # 프롬프트에 변수들을 채워서 입력 생성
        chain = ConversationChain(
            llm=llm_roleplaying,
            prompt=prompt,
            memory=combined_memory,
            verbose=False
        )

        response = chain.invoke({"input": input_data, "context": context})
        
        # Assuming the response is directly the assistant's message
        response_message = response["response"]

        return {"messages": state["messages"] + [response_message]}
    except Exception as e:
        print(f'Error in chatbot_without_rag: {e}')
        raise

def should_use_rag(state: State) -> Literal["evaluate_rag_output", "chatbot_without_rag"]:
    if state["activate_RAG"] == "Yes":
        return "evaluate_rag_output"
    else:
        return "chatbot_without_rag"

def final_decision(state: State) -> Literal["chatbot_with_rag", "chatbot_without_rag"]:
    if state["activate_RAG"] == "Yes":
        return "chatbot_with_rag"
    else:
        return "chatbot_without_rag"

# Define functions to add nodes and edges to the graph
def add_nodes_and_edges(graph_builder):
    # Add nodes
    graph_builder.add_node("rag_chatbot", lambda state: rag_chatbot(state, DB_PATH="./llm_2/vectordatabase", file_path="./llm_2/rag_example.jsonl", chunk_size=512, chunk_overlap=50))
    graph_builder.add_node("evaluate_rag_output", evaluate_rag_output)
    graph_builder.add_node("chatbot_with_rag", chatbot_with_rag)
    graph_builder.add_node("chatbot_without_rag", chatbot_without_rag)

    # Add edges
    graph_builder.add_edge(START, "rag_chatbot")
    graph_builder.add_conditional_edges("rag_chatbot", should_use_rag)
    graph_builder.add_conditional_edges("evaluate_rag_output", final_decision)
    graph_builder.add_edge("chatbot_with_rag", END)
    graph_builder.add_edge("chatbot_without_rag", END)

# Add nodes and edges to the graph
add_nodes_and_edges(graph_builder)

# Compile the graph
graph = graph_builder.compile()

# 단일 입력 처리
user_input = input("message: ")
state = {"messages": [HumanMessage(content=user_input)], "activate_RAG": "No", "explain": ""}

# 그래프 실행
try:
    for event in graph.stream(state):
        for value in event.values():
            print("Assistant:", value["messages"][-1])
            # 응답 메시지를 상태에 추가
            state["messages"].append(value["messages"][-1])
except Exception as e:
    print(f'Error during graph execution: {e}')
    raise
