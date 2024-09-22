"""
    Run vllm server. This code will run on GPU remote server, alone.
"""
import time
import signal
import subprocess
import multiprocessing

from pyngrok import ngrok
from server.utils import signal_handler, argparse_load_from_yaml

def run_server():
    """
        VLLM 서빙 메인 실행함수
    """
    # global server_process
    # global public_url

    # configuration
    config_path = 'configs/default_config.yaml'
    aiserver_config = argparse_load_from_yaml(config_path)
    
    # vllm fastapi server 실행
    server_command = [
        "python", "-m", str(aiserver_config.python_command),
        "--model", str(aiserver_config.llm_model_path),
        "--tensor-parallel-size", str(aiserver_config.num_gpus), 
        "--api-key", str(aiserver_config.secure_api_key), 
        "--port", str(aiserver_config.request_port),
        "--max-model-len", str(aiserver_config.max_model_len),
        "--seed", str(aiserver_config.seed),
        "--gpu-memory-utilization", str(aiserver_config.gpu_memory_utilization),
    ]
    server_process = subprocess.Popen(server_command)
    
    # # ngrok 터널 생성
    # wait_time = 15
    # print(f"\n\n>>>>>>> [ngrok] Wait for about {wait_time} seconds for vllm server fully init !!!\n\n")
    # time.sleep(wait_time)
    
    # # 웹소켓 서버 실행
    # from ws import start_websockets # error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp-a34b3233.so.1 library.
    # ws_server = multiprocessing.Process(target=start_websockets, args=(int(aiserver_config.ws_port),))
    # # ws_thread = threading.Thread(target=start_websockets, args=(int(aiserver_config.ws_port),))
    # ws_server.start()
    
    # # 웹소켓 url 터널링
    # public_url = ngrok.connect(int(aiserver_config.ws_port))
    
    # print(f"\n\n\n>>>>>>> [ngrok]  Copy following url into your config.yaml !!!\n[Public URL]: {public_url}\n\n\n")
    
    # sigint signal handler
    signal.signal(signal.SIGINT, signal_handler(server_process, None))
    
    # 프로그램 종료까지 대기
    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt 발생. 프로그램 종료.\n")
    
    
# if __name__ == "__main__":
#     main()