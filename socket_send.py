import socket

def udp_client(message):
    # UDP 소켓 생성
    ip = "115.145.170.198" 
    port = 5400
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        # 서버로 메시지 전송
        message_bytes = message
        # 파일 열기
        with open('frame.png', 'rb') as file:
            # 파일 읽기
            file_data = file.read()
        
        print("file_data 크기 ", len(file_data))
        print("fiie data type: ", type(file_data))
        # 서버로 파일 전송
        chunk_size = 1024  # Define the chunk size
        for i in range(0, len(file_data), chunk_size):
            udp_sock.sendto(file_data[i:i+chunk_size], (ip, port))
            
        
        # print("전쳍크기: ",len(mess    age_bytes))
        # for i in range(0, len(message_bytes), 1024):
        #     udp_sock.sendto(message_bytes[i:i+1024], (ip, port))
        # #udp_sock.sendto(message_bytes, (ip, port))
        # #udp_sock.sendto(message.encode(), (ip, port))
        # print(f"Sent message to {ip}:{port}")
    
    finally:
        udp_sock.close()

# # 사용 예시
# if __name__ == "__main__":
#     IP = "115.145.170.198"  # 서버의 IP 주소
#     PORT = 5400        # 서버의 포트
#     MESSAGE = "babo"  # 전송할 메시지
    
#     udp_client(IP, PORT, MESSAGE)