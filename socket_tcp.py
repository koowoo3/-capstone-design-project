import socket
import time
def tcp_client():
    # TCP 소켓 생성
    ip = "115.145.170.198" 
    port = 5400
    
    tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # 서버에 연결
        tcp_sock.connect((ip, port))
        
        # 파일 읽기
        while True:
            file_data = b''
            with open("frame.png", "rb") as file:
                file_data = file.read()

            
            # 데이터 전송
            tcp_sock.sendall(file_data)
            print(f"Sent file to {ip}:{port}")
            time.sleep(0.5)
    finally:
        tcp_sock.close()

# 사용 예시
if __name__ == "__main__":
    FILE_PATH = "path_to_your_file.jpg"  # 전송할 파일 경로
    IP = "115.145.170.198"  # 서버의 IP 주소
    PORT = 5400             # 서버의 포트
    tcp_client(FILE_PATH, IP, PORT)
