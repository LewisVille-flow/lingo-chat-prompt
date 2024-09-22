import yaml
import signal

# from pyngrok import ngrok
from configs import AIServerConfig


def signal_handler(server_process, ws_server):
    """
        SIGINT 시그널 핸들러 - ctrl+c 입력시 vllm 서버와 ngrok 연결 해제
    """
    def _signal_handler(sig, frame):
        print('\n\n>>>>>>> SIGINT received, disconnecting vllm and ngrok...\n>>>>>>> Wait for 10 seconds ...\n\n')
        # ngrok.disconnect(public_url)
        # ngrok.kill()
        
        if server_process:
            server_process.terminate()
            server_process.send_signal(signal.SIGINT)
            server_process.wait(timeout=10)
        
        if ws_server:
            ws_server.terminate()
            
        exit(0)
    return _signal_handler


def argparse_load_from_yaml(yaml_path: str):
    """
        yaml 파일 configuration load 함수
    """
    with open(yaml_path, "r") as f:
        config_data = yaml.safe_load(f)
    config = AIServerConfig(**config_data)
    
    return config