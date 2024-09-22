import os
import pytz
import requests
import socketio
import multiprocessing

from typing import List, Dict, Union
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

import asyncio
# from graph import chat_graph, config


##########
### dotenv setting
##########
load_dotenv(override=True)
redis_url = os.getenv('VM_URL')
redis_port = os.getenv('REDIS_PORT')
redis_ms_id = os.getenv('REDIS_MS_ID')
db_port = os.getenv('DB_PORT')


##########
### Backend db 연결 설정
##########
db_reload_url = f"http://"+redis_url+":"+db_port+"/chats/db/ai/{chat_room_id}"

websocket_url = "http://"+os.getenv('VM_URL')+":"+os.getenv('API_PORT')
websocket_namespace = os.getenv('API_WS_NAMESPACE')
websocket_event = os.getenv('API_WS_EVENTNAME')

connection_test_message = "ai 서버에서 보내는 연결 테스트 메세지 입니다."

seoul_tz = pytz.timezone('Asia/Seoul')
asio = socketio.AsyncClient(logger=True, engineio_logger=True)

lock = multiprocessing.Lock()


##########
### utility function setting
##########
def fill_user_id(chat_history: List[dict], user_id: str) -> List[dict]:
    """
        chat_history에 user_id를 추가합니다.
    """
    if len(chat_history) == 0 or chat_history[0].get('user_id') is not None:
        return chat_history
    
    for chat in chat_history:
        chat['user_id'] = user_id
    return chat_history


def renew_created_time(db_chat_history: List[Dict]) -> List[Dict]:
    """
        db로부터 받은 chat_history의 마지막 메세지에 시간을 초기화합니다.
        이후 redis에 캐싱합니다.
    """
    if db_chat_history and db_chat_history[-1].get("created_time", None):
        db_chat_history[-1]["created_time"] = datetime.now(seoul_tz).strftime('%Y-%m-%d %H:%M:%S')
        
    return db_chat_history


async def caching_chat_history(redis_client,
                               chat_room_id: int,
                               db_chat_history: List[Dict]) -> None:
    """
        대화 히스토리를 레디스에 캐싱합니다. str 형태로 변환하여 저장합니다.
    """
    if db_chat_history and isinstance(db_chat_history[-1], dict):
        db_chat_history = [str(chat) for chat in db_chat_history]

        await redis_client.rpush(chat_room_id, *db_chat_history)
    

def _convert_format_to_message(_chat_history: List[Dict]) -> List[Union[HumanMessage, AIMessage]]:
    """
        chat_history 포맷을 Message로 변환합니다.
        
        Args:
            _chat_history: List[Dict]
                [{"role": "user", "content": str}, {"role": "assistant", "content": str}, ...]
        
        Returns:
            chat_history: List[Message]
            
    """
    chat_history = []
    for data in _chat_history:
        if isinstance(data, str) or isinstance(data, bytes):
            _data = eval(data.decode('utf-8'))
        else:
            _data = data
        if _data.get('role') == 'user':
            _data = HumanMessage(content=_data["content"])
        elif _data.get('role') == 'assistant':
            _data = AIMessage(content=_data["content"])
        chat_history.append(_data)
     
    return chat_history


async def _get_chat_history_redis(redis_client,
                                  chat_room_id: int) -> Union[List, List[Union[HumanMessage, AIMessage]]]:
    """
        Redis로부터 채팅 히스토리를 가져옵니다.
        
        Args:
            redis_client (Redis): Redis 클라이언트 인스턴스.
            chat_room_id (int): 채팅방 ID.
        
        Returns:
            Union[List, List[Union[HumanMessage, AIMessage]]]: 
            - Redis에 채팅 기록이 없으면 빈 리스트를 반환.
            - 기록이 있을 경우, HumanMessage 또는 AIMessage 객체로 변환된 리스트를 반환
    """
    chat_history = await redis_client.lrange(chat_room_id, 0, -1)
    if chat_history:
        chat_history = _convert_format_to_message(chat_history)
    return chat_history


async def _get_chat_history_db(redis_client,
                               user_id: str,
                               chat_room_id: int) -> List[Union[HumanMessage, AIMessage]]:
    """
        db로부터 채팅 히스토리를 가져옵니다.
        
        Args:
            redis_client (Redis): Redis 클라이언트 인스턴스.
            user_id (str): 사용자 ID.
            chat_room_id (int): 채팅방 ID.
            
        Returns:
            chat_history: List[Union[HumanMessage, AIMessage]]
            
    """
    result = []
    db_chat_history = requests.get(db_reload_url.format(chat_room_id=chat_room_id))
    
    if not db_chat_history.status_code == 200 or db_chat_history.status_code == 500 or db_chat_history.json() is None:
        return result
    db_chat_history = db_chat_history.json()["result"]      # type: List[Dict]
    
    # 편의상 여기서 user_id 를 추가합니다.
    db_chat_history = fill_user_id(db_chat_history, user_id)
    
    # 마지막 메세지에 시간 초기화 및 레디스 캐싱
    db_chat_history = renew_created_time(db_chat_history)
    await caching_chat_history(redis_client, chat_room_id, db_chat_history)
    
    # 포맷 변환
    chat_history = _convert_format_to_message(db_chat_history)
    
    return chat_history


async def get_chat_history(redis_client,
                           user_id: str, 
                           chat_room_id: int) -> List[Union[HumanMessage, AIMessage]]:
    """
        redis로부터 채팅 히스토리를 가져와 포맷을 변경하여 반환합니다.
        
        레디스 안에서: _chat_history == [bytes(list(dict))]
        request로부터 받은 데이터: db_chat_history == [{'chat_room_id': int, 'user_id': str, 'chat_history': str}]
        리턴 형태: chat_history == [HumanMessage(), AIMessage()...]
    """
    # 레디스로부터 채팅 히스토리 조회
    chat_history = await _get_chat_history_redis(redis_client, chat_room_id)
    
    # db 조회 시퀀스 추가 feat-#22
    if len(chat_history) == 0:
        chat_history = await _get_chat_history_db(redis_client, user_id, chat_room_id)
            
    return chat_history
    

async def save_chat_history(redis_client,
                            chat_room_id: int,
                            user_id: str,
                            user_message: str,
                            final_response: str) -> None:
    """
        chat_history를 redis에 저장합니다.
    """
    current_time = datetime.now(seoul_tz).strftime('%Y-%m-%d %H:%M:%S')
    save_messages = [
        str({"role": "user", "content": user_message, "user_id": user_id, "chat_room_id": chat_room_id}),
        str({"role": "assistant", "content": final_response, "user_id": user_id, "chat_room_id": chat_room_id, "created_time": current_time}),
    ]
    await redis_client.rpush(chat_room_id, *save_messages)
    

async def emit_chat_message(chat_room_id: int,
                            user_id: str,
                            response: str,
                            is_final: bool) -> None:
    """
        웹소켓으로 메세지를 전송합니다.
        
        Refer to:
            https://python-socketio.readthedocs.io/en/stable/api.html#socketio.AsyncClient.emit
    """
    lock.acquire()
    await asio.emit(event="ai_chat_message", 
                    data={"user_id": user_id,
                          "chat_room_id": chat_room_id,
                          "response": response,
                          "is_final": is_final},
                    namespace=websocket_namespace)
    lock.release()
    return


async def call_chat_graph(chat_history: List[Union[HumanMessage, AIMessage]],
                          user_message: str,
                          chat_room_id: int,
                          user_id: str) -> str:
    """
        chat_graph를 호출하여 chatbot의 답변을 받아옵니다.
        Refer to: Lingo_Chat/ai_server/graph/graph.py
        
        실시간 대화를 위해 async_generator를 사용하며, 생성된 chunk를 async client 으로 전송합니다.
    """
    # response = chat_graph.astream_events({"history": chat_history, "messages": user_message}, config=config, version='v1')    # return: async_generator
    
    async def async_response_generator(response_text):
        for line in response_text:
            await asyncio.sleep(1)  # 네트워크 지연 또는 처리 시간 시뮬레이션
            yield line
            
    response_texts = [
        {'data': {'chunk': '이것은 테스트 답변입니다(1). '}, 'name': 'ChatOpenAI'},
        {'data': {'chunk': '이것은 테스트 답변입니다(2).\n\n'}, 'name': 'ChatOpenAI'},
        {'data': {'chunk': 'stop.\n\n'}, 'name': 'ChatOpenAI'},
        # {'data': {'chunk': '이것은 테스트 답변입니다(3).'}, 'name': 'ChatOpenAI'},
        # {'data': {'chunk': '이것은 테스트 답변입니다(4).'}, 'name': 'ChatOpenAI'},
    ]
    response = async_response_generator(response_texts)
    
    
    final_response = ""
    async for resp in response:
        try:
            # test
            # chatbot_messages = resp['data']['chunk'].content
            chatbot_messages = resp['data']['chunk']
            
            if chatbot_messages and resp['name'] == 'ChatOpenAI':
                # print(f"{chatbot_messages}", end='')
                
                await emit_chat_message(chat_room_id, user_id, chatbot_messages, False)
                final_response += chatbot_messages
            # test    
            # if resp['name'] == 'ChatOpenAI' and resp['data']['chunk'].response_metadata['finish_reason'].lower() == 'stop':
            if resp['name'] == 'ChatOpenAI' and resp['data']['chunk'] == 'stop.\n\n':
                await emit_chat_message(chat_room_id, user_id, "", True)
                print(f"\n--------------------")
                print(f">> [Langgraph 생성 최종 답변]: {final_response}\n\n>> [response handler] finished...\n\n")
                
        except Exception as e:
            pass
    
    return final_response