"""
4. r키 입력시 record 변수만 true로 바꾸고, videowriter 초기화 로직 추가

5. waitkey(20) → waitkey(1)로 변경
  멀티스레딩이고, 카메라 3대면 속도가 느려서 바꿔봤습니다. PC 성능에 맞게 설정하면 됩니다.
"""

import datetime
import cv2
import threading
import os

base_path = os.path.dirname(os.path.abspath("__file__"))
base_path = base_path+"\\video"
if not os.path.exists(base_path):
    os.makedirs(base_path)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
record = False

class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        super(camThread, self).__init__()
        self.previewName = previewName
        self.camID = camID

    def run(self):
        print(f"Starting {self.previewName}")
        camPreview(self.previewName, self.camID)


def camPreview(previewName, camID):
    global record
    global base_path
    global fourcc

    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)
    video = -1 # 객체를 담을 수 있는 변수 선언(-1로 초기화한거임.)
    if cam.isOpened():
        rval, frame = cam.read()
    else:
        rval = False

    while rval:
        cv2.imshow(previewName, frame)
        rval, frame = cam.read()
        key = cv2.waitKey(1)

        now = datetime.datetime.now().strftime("%d_%H-%M-%S")

        # record 상태이고 video가 초기화되어 현재 진행되고 있는 video가 없을때!
        # loop 돌면서 video에 영상이 저장된다.
        if (record == True and video == -1):
            #recode -> video parameters: save name, video format code, fps, (frame height, frame width)
            video = cv2.VideoWriter(os.path.join(base_path,previewName+"_"+str(now)+".avi"), fourcc, 10.0, (frame.shape[1], frame.shape[0]))

        if (record == False and video != -1):
            video.release()

        if key == 27: # ESC
            break

        elif key == 114: # r
            print("Start Record")
            record = True

        elif key == 101: # e
            #end recode
            print("End Record")
            record = False

        elif key == 99: # c
            #capture
            print("Capture")
            cv2.imwrite(os.path.join(base_path,previewName+"_"+str(now)+".png"), frame)
            print(os.path.join(base_path,previewName+str(now)+".png"))

        if record == True:
            print("Recording...")
            if video != -1:
                video.write(frame)


    cam.release()
    cv2.destroyWindow(previewName)

if __name__ == "__main__":
    # 3개의 cam의 Thread 인스턴스를 생성한다.
    thread_1 = camThread("Cam1", 0)
    thread_2 = camThread("Cam2", 1)
    thread_3 = camThread("Cam3", 2)

    # 생성된 스레드 인스턴스를 실행한다.
    """
    It arranges for the object's run() method to be invoked in a separate thread of control.
    """
    thread_1.start()
    thread_2.start()
    thread_3.start()

    print("Active threads", threading.activeCount())
