"""
    api 서버에서 ai 서버와 소켓 연결을 했다고 가정하고,
    ai 서버는 Redis를 이용해 유저의 입력을 읽어들인다.

    동작:
    1. 실시간 채팅방 생성에 따른 {user_message, chat_room_id}를 redis 에서 받아옴.
    2. 또한 이전 채팅 히스토리를 
        2-1. redis에서 chat_room_id를 사용해 조회.
        2-2. cloud db에서 chat_room_id를 사용해 조회.
    3. chatbot을 호출하고 다시 redis에 저장하며, 결과를 리턴.
"""
import os
import pytz
import socketio
import redis.asyncio as redis
import asyncio
import requests

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
# from graph import chat_graph, config

##########
### dotenv setting
##########
load_dotenv(override=True)
redis_url = os.getenv('VM_URL')
redis_port = os.getenv('REDIS_PORT')
redis_ms_id = os.getenv('REDIS_MS_ID')
db_port = os.getenv('DB_PORT')

# Redis 및 SQLite 연결 설정
r = redis.Redis(host=redis_url, port=redis_port, db=1)
db_reload_url = "http://"+redis_url+":"+db_port+'/reload'

websocket_url = "http://"+os.getenv('VM_URL')+":"+os.getenv('API_PORT')
websocket_namespace = os.getenv('API_WS_NAMESPACE')
websocket_event = os.getenv('API_WS_EVENTNAME')

connection_test_message = "ai 서버에서 보내는 연결 테스트 메세지 입니다."

seoul_tz = pytz.timezone('Asia/Seoul')
asio = socketio.AsyncClient(logger=False, engineio_logger=False)


##########
### socket-io event handler
##########
@asio.on(event='connect', namespace=websocket_namespace)
async def connect_handler():
    print(f"\n>> api 서버와 웹소켓 연결 성공!\nsio id: {asio.sid}, transport: {asio.transport}\n\n")
    await asio.emit('ai_chat_message', connection_test_message, namespace=websocket_namespace)
    
    
async def process_message(ws_server, event_name, namespace):
    """
        1. redis 에서 user_message, chat_room_id를 받아옴
        2. graph를 통해 inference 진행 후 ws으로 전송
        3. redis에 chat_history 저장
    """
    while True:
        try:
            uri = redis_url
            r = redis.Redis(host=uri, port=redis_port, db=1)
            
            _, message = await r.brpop(redis_ms_id)
            
            # 1. 메세지 수신
            message = eval(message.decode('utf-8'))
            chat_room_id, user_message = message['chat_room_id'], message['user_message']
            user_id = message['user_id']
            print(f"chat_room_id: {chat_room_id}, message: {user_message}\n\n")
            
            # 2. 히스토리 조회 -> AIMessages, HumanMessages 로 변환이 필요
            async def _get_chat_history(chat_room_id):
                """
                    _chat_history == [bytes(list(dict))]
                """
                _chat_history = await r.lrange(chat_room_id, 0, -1)
                chat_history = []
                
                # 3. chat format으로 다시 변경
                for data in _chat_history:
                    _data = eval(data.decode('utf-8'))
                    if _data.get('role') == 'user':
                        _data = HumanMessage(content=_data['content'])
                    elif _data.get('role') == 'assistant':
                        _data = AIMessage(content=_data['content'])
                    chat_history.append(_data)
                
                # chat_history = [eval(data.decode('utf-8')) for data in _chat_history]
                return chat_history
            
            chat_history = await _get_chat_history(chat_room_id)
            
            if len(chat_history) == 0:      # db 조회 시퀀스 추가 feat-#22
                db_chat_history = requests.get(db_reload_url, params={'chat_room_id': int(chat_room_id)}).json()
                
                if db_chat_history is not None:
                    result = []
                    for idx, chat in enumerate(eval(db_chat_history[0]['chat_history'])):
                        if idx == len(eval(db_chat_history[0]['chat_history']))-1:
                            chat['created_time'] = datetime.now(seoul_tz).strftime('%Y-%m-%d-%H-%M-%S')
                        result.append(str(chat))
                    # db_chat_history = [chat for chat in eval(db_chat_history[0]['chat_history'])]
                    await r.rpush(chat_room_id, *result)
                    chat_history = await _get_chat_history(chat_room_id)
                    
            print(f"\n\n>> chat_history: {chat_history}\n\n")
            
            # 4. chatbot 호출
            # test for no gpu enviornment
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
                        await ws_server.emit(event_name, 
                                             {'user_id': user_id,
                                              'chat_room_id': chat_room_id,
                                              'response': chatbot_messages,
                                              'is_final': False}, 
                                             namespace=namespace)
                        final_response += chatbot_messages
                    # test    
                    # if resp['name'] == 'ChatOpenAI' and resp['data']['chunk'].response_metadata['finish_reason'].lower() == 'stop':
                    if resp['name'] == 'ChatOpenAI' and resp['data']['chunk'] == 'stop.\n\n':
                        await ws_server.emit(event_name, 
                                             {'user_id': user_id,
                                              'chat_room_id': chat_room_id,
                                              'response': "",
                                              'is_final': True}, 
                                             namespace=namespace)
                        print(f"\n>> [Langgraph 생성 답변]: {final_response}\n\n")
                        print(f"\n>> [response handler] finished...\n\n")
                        
                except Exception as e:
                    # print(f">> error: {e}")
                    pass
            
            # 5. redis history 저장
            current_time = datetime.now(seoul_tz).strftime('%Y-%m-%d-%H-%M-%S')
            save_messages = [
                str({'role': 'user', 'content': user_message, 'user_id': user_id, 'chat_room_id': chat_room_id}),
                str({'role': 'assistant', 'content': final_response, 'user_id': user_id, 'chat_room_id': chat_room_id, 'created_time': current_time}),
            ]
            await r.rpush(chat_room_id, *save_messages)

        except Exception as e:
            print(f"\n>> error: {e}\n\n\n")
    
 
async def main():
    try:
        # socket io 연결
        await asio.connect(websocket_url, namespaces=[websocket_namespace])
        await asio.sleep(seconds=1)
        
        # 메인 추론 함수 비동기 실행
        asyncio.create_task(process_message(ws_server=asio, event_name='ai_chat_message', namespace=websocket_namespace))
        await asio.wait()
            
    except Exception as e:
        print(f"\n>> api 서버와 웹소켓 연결에 실패했습니다.\n>> 에러 메세지: {e}\n")        
        await asio.disconnect()
        
asyncio.get_event_loop().run_until_complete(main())
