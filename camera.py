import cv2

global frame

def get_smartphone_camera():
    # Droidcam 연결 설정
    #droidcam_url = "http://192.168.1.10:4747/video"  
    droidcam_url = "http://10.221.61.170:4747/video"
    # 학교 droidcam_url = "http://10.221.153.210:4747/video"
    #droidcam_url = "http://192.168.0.31:4747/video"
    #droidcam_url = "http://192.168.0.13:4747/video"
    #droidcam_url = "http://10.221.51.180:4747/video"
    cap = cv2.VideoCapture(droidcam_url)

    # 카메라 연결 확인
    if not cap.isOpened():
        print("스마트폰 카메라 연결에 실패했습니다.")
        return None
    i = 0
    while True:
        i+=1
        ret, frame = cap.read()
        if not ret:
            break
        #cv2.imshow("Smartphone Camera", frame)

        if i % 10 == 0:
            yield frame
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

# 함수 호출
#get_smartphone_camera()