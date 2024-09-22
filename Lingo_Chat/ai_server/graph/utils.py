import os

from dotenv import load_dotenv
from typing import List, Union, Optional
from Levenshtein import ratio
from operator import itemgetter
from transformers import AutoTokenizer

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from langchain_community.tools.tavily_search import TavilySearchResults

from server.utils import argparse_load_from_yaml
from configs import (default_system_prompt,
                     orbit_role_name, orbit_role_description,
                     neuroticism_role_name, neuroticism_role_description
)

load_dotenv()

##########
### tokenizer setting
##########
config_path = 'configs/default_config.yaml'
aiserver_config = argparse_load_from_yaml(config_path)

model_path = str(aiserver_config.llm_model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")

##########
### llm setting
##########
basic_llm = ChatOpenAI(
    model=model_path,
    openai_api_base="http://0.0.0.0:2496/v1",       # gpt api 가 아닌, vllm이 동작하는 포트로 연결
    max_tokens=2048,
    temperature=0.6,
    api_key="test_api_key",
    streaming=True,
    stop=['<|im_end|>', '<|endoftext|>', '<|im_start|>', '</s>'],
    top_p=0.95, 
    frequency_penalty=1.4,
    seed=42,
)

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.0-pro",
    google_api_key=os.environ.get("LE_GEMINI_API_KEY"),
    max_output_tokens=382,  # default 64
    # convert_system_message_to_human = False,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,},
    verbose = True,
)


##########
### prompt setting
##########
# system_prompt = default_system_prompt.format(role_name=orbit_role_name,
#                                              role_description_and_catchphrases=orbit_role_description)


system_prompt = {
    'orbit': default_system_prompt.format(role_name=orbit_role_name,
                                          role_description_and_catchphrases=orbit_role_description),
    'neuroticism': default_system_prompt.format(role_name=neuroticism_role_name,
                                                role_description_and_catchphrases=neuroticism_role_description)
}

primary_assistant_prompt = ChatPromptTemplate.from_messages([
    # ("system", system_prompt),
    ("placeholder", "{messages}"),
])
persona_search_assistant_prompt = ChatPromptTemplate.from_messages([
    ("human", system_prompt['orbit']), # == ("system", system_prompt)
    ("placeholder", "{messages}"),
])
rag_assistant_prompt = PromptTemplate.from_template(
    tokenizer.bos_token
    +"{context}"
    +tokenizer.eos_token
    +"\n\n{system_prompt_and_history}"
)


##########
### llm chain setting
##########
tool = TavilySearchResults(max_results=1)
tools = [tool]
tool_name_list = [tool.name for tool in tools]

search_llm = gemini_llm.bind_tools(tools)
persona_search_llm = persona_search_assistant_prompt | gemini_llm

local_llm = primary_assistant_prompt | basic_llm

rag_llm = ( # type: langchain_core.runnables.base.RunnableSequence
    {
        "context": itemgetter("context"),
        "system_prompt_and_history": itemgetter("messages"),
    }
    | rag_assistant_prompt
    | basic_llm
)
rag_llm_prompt = (
    {
        "context": itemgetter("context"),
        "system_prompt_and_history": itemgetter("messages"),
    }
    | rag_assistant_prompt
)


def convert_chat_history_format(chat_history: List[Union[HumanMessage, AIMessage]],
                                selected_persona: Optional[str]) -> str:
    """
        1. system prompt append
        2. [AIMessage(), HumanMessage(), ..] 형태의 데이터를
           [{'role', 'content'}, {}, ...] 형태로 변환한 뒤, <|...|> 토큰을 붙여 str로 반환합니다.
        
        Args:
            chat_history - [HumanMessage(), AIMessage(), ..] 형태의 대화내역
            selected_persona - 선택된 페르소나 이름    
        
        Return:
            str - chat format으로 변환된 대화내역
            ex) '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhi?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nhow are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    """
    result_item = []
    # result_item.append({'role': 'system', 'content': system_prompt})
    if selected_persona:
        result_item.append({'role': 'system', 'content': system_prompt[selected_persona]})
    
    for idx, message in enumerate(chat_history):
        if type(message) == AIMessage:
            result_item.append({'role': 'assistant', 'content': message.content})
        elif type(message) == HumanMessage:
            result_item.append({'role': 'user', 'content': message.content})
    
    return tokenizer.apply_chat_template(result_item, tokenize=False, add_generation_prompt=True)


def fix_called_tool_name(called_tool_name: str) -> str:
    """
        llm이 호출한 함수 이름의 철자가 잘못되는 경우가 종종 발생한다.
        이를 방지하기 위해, llm이 호출한 함수 이름이 tool_name_list에 있는지 확인하고,
        레벤슈타인 거리를 계산하여 가장 높은 ratio를 가진 idx를 선택하여 tool을 호출한다.
        
        Args: called_tool_name - llm이 호출한 함수 이름
        Returns: called_tool_name - 수정된 함수 이름
    """
    # 1. 레벤슈타인 거리 계산
    _max_tool_name_ratio, _max_idx = 0.0, 0
    for idx, _name in enumerate(tool_name_list):
        _tool_name_ratio = ratio(_name, called_tool_name)
        if _tool_name_ratio > _max_tool_name_ratio:
            _max_tool_name_ratio = _tool_name_ratio
            _max_idx = idx
    
    # 2. 가장 높은 ratio를 가진 idx 선택
    called_tool_name = tool_name_list[_max_idx]
    return called_tool_name
