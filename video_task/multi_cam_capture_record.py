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
    if cam.isOpened():
        rval, frame = cam.read()
    else:
        rval = False

    while rval:
        cv2.imshow(previewName, frame)
        rval, frame = cam.read()
        key = cv2.waitKey(20)

        now = datetime.datetime.now().strftime("%d_%H-%M-%S")

        if key == 27: # ESC
            break
        elif key == 114: # r
            #recode -> video parameters: save name, video format code, fps, (frame height, frame width)
            print("Start Record")
            record = True
            video = cv2.VideoWriter(os.path.join(base_path,previewName+"_"+str(now)+".avi"), fourcc, 10.0, (frame.shape[1], frame.shape[0]))
        elif key == 101: # e
            #end recode
            print("End Record")
            record = False
            video.release()
        elif key == 99: # c
            #capture
            print("Capture")
            cv2.imwrite(os.path.join(base_path,previewName+"_"+str(now)+".png"), frame)
            print(os.path.join(base_path,previewName+str(now)+".png"))

        if record == True:
            print("Recording...")
            video.write(frame)


    cam.release()
    cv2.destroyWindow(previewName)

if __name__ == "__main__":
    thread_1 = camThread("Cam1", 0)
    thread_2 = camThread("Cam2", 1)
    thread_3 = camThread("Cam3", 2)

    thread_1.start()
    thread_2.start()
    thread_3.start()

    print("Active threads", threading.activeCount())


