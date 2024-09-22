"""
    요약:
    레디스의 데이터를 읽어 ai 추론 진행 후 결과를 api 서버로 전송

    동작:
    0. multiprocessing 으로 redis를 지속적으로 조회.
    1. 실시간 채팅방 생성에 따른 {user_message, chat_room_id}를 redis 에서 받아옴.
    2. 또한 이전 채팅 히스토리를 
        2-1. redis에서 chat_room_id를 사용해 조회.
        2-2. cloud db에서 chat_room_id를 사용해 조회.
    3. chatbot을 호출하고 결과를 순차적으로 전송.
    4. 반복
"""
import redis.asyncio as redis
import asyncio

from concurrent.futures import ProcessPoolExecutor

from client.utils import get_chat_history, save_chat_history, call_chat_graph
from client.utils import (asio, connection_test_message,
                          websocket_namespace, websocket_url,
                          redis_url, redis_port, redis_ms_id, lock)


##########
### socket-io event handler
##########
@asio.on(event='connect', namespace=websocket_namespace)
async def connect_handler():
    print(f"\n>> api 서버와 웹소켓 연결 성공!\nsio id: {asio.sid}, transport: {asio.transport}\n\n")
    await asio.emit('ai_chat_message', connection_test_message, namespace=websocket_namespace)
    
    
async def process_message():
    """
        메인 프로세스 함수
    """
    lock.acquire()
    await asio.connect(websocket_url, namespaces=[websocket_namespace])
    await asio.sleep(seconds=1)
    lock.release()
    
    # redis 연결
    redis_client = redis.Redis(host=redis_url, port=redis_port, db=1)
    
    while True:
        try:
            result = await redis_client.brpop(keys=redis_ms_id, timeout=0.5)
            
            if result:
                # 1. 메세지 수신
                message = result[1]
                message = eval(message.decode('utf-8'))
                chat_room_id, user_message, user_id = message['chat_room_id'], message['user_message'], message['user_id']
                selected_persona = message['persona']
                print(f"\n>> chat_room_id: {chat_room_id}, user_id: {user_id}\n>> user message: {user_message}\n\n")
                
                # 2~3. 히스토리 조회 및 변환 -> AIMessages, HumanMessages로 변환
                chat_history = await get_chat_history(redis_client, user_id, chat_room_id)
                print(f"\n\n>> chat_history: {chat_history}\n\n")
                
                # 4. chatbot 호출
                
                response = await call_chat_graph(selected_persona, chat_history, user_message, chat_room_id, user_id)
                
                # 5. redis history 저장 w/ 웹소켓 메세지 전송
                await save_chat_history(redis_client, chat_room_id, user_id, user_message, response)
                
                # 6. 웹소켓으로 메세지 전송 -> 5번에서 처리
                # await emit_chat_message(chat_room_id, user_id, response, True) # 이렇게 보낼 경우 백엔드파트에서 데이터 중복됨.

        except Exception as e:
            print(f"\n\n>> An error occured: {e}\n>> AI server will retry after 5 seconds.\n\n")
            await asio.sleep(seconds=5)
            # return Exception
        
    
def start_process_message():
    return asyncio.run(process_message())


async def main():
    try:
        # socket io 연결
        # await asio.connect(websocket_url, namespaces=[websocket_namespace])
        # await asio.sleep(seconds=1)
        
        with ProcessPoolExecutor() as executor:
            loop = asyncio.get_running_loop()
            
            # 메인 추론 함수 비동기 실행
            # refer: https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor
            tasks = [loop.run_in_executor(executor, start_process_message) for _ in range(4)]
            
            await asio.wait()
            
    except Exception as e:
        print(f"\n>> api 서버와 웹소켓 연결에 실패했습니다.\n>> 에러 메세지: {e}\n")        
        await asio.disconnect()


def run_client():
    # asyncio.run(main())
    asyncio.get_event_loop().run_until_complete(main())
