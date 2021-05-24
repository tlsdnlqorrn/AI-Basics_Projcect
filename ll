from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import os.path
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon

overlay_mask = cv2.imread("set_up/filter/default.png", cv2.IMREAD_UNCHANGED)
overlay_nonMask = cv2.imread("set_up/filter/default.png", cv2.IMREAD_UNCHANGED)

test_mode = True
filter_scale = 1

def get_img_path(index = 0, maskMode = False, imtype = 'png'):
    mask_keyword = "mask" if maskMode else "non_mask"
    path = os.path.sep.join(["set_up", "filter", str(index), mask_keyword + "." + imtype])
    return path

def get_img(index = 0, maskMode = False, imtype = 'png'):
    img = cv2.imread(get_img_path(index = index, maskMode = maskMode, imtype = imtype), cv2.IMREAD_UNCHANGED)
    return img
    
def exit():
    print("끝")
    cv2.destroyAllWindows()
    vs.stop()
    sys.exit()

class BeautyButton():
    myWidget = None
    buttonSize = 0
    def __init__(self, index, character = ''):
        # 창을 초기화하지 않았다면 새로 만들어줍니다.
        if BeautyButton.myWidget == None:
            BeautyButton.init_widget()
        # 버튼 번호에 맞는 이미지들을 등록해줍니다.
        self.maskImage = get_img(index, True)
        self.nonMaskImage = get_img(index, False)
        self.button = QPushButton(character, BeautyButton.myWidget)
        self.button.resize(100, 100)
        self.button.move(BeautyButton.buttonSize * 100, 0)
        self.index = BeautyButton.buttonSize
        self.button.clicked.connect(self.myClick)
        BeautyButton.buttonSize += 1
        if character == '':
            self.button.setIcon(QIcon(get_img_path(index, True)))
            self.button.setIconSize(QSize(90, 90))
        else:
            self.button.setStyleSheet("font-size:80px;")
            self.button.setToolTip('non-filter')
    
    def myClick(self):
        global overlay_mask, overlay_nonMask
        print("버튼 클릭: {}".format(self.index))
        overlay_mask = self.maskImage
        overlay_nonMask = self.nonMaskImage
        
    @staticmethod
    def add_quit_button():
        quit_button = QPushButton('Quit', BeautyButton.myWidget)
        quit_button.resize(100, 30)
        quit_button.move(200, 110)
        quit_button.setStyleSheet("background-color: red;font-size:20px;font-family:Times New Roman;")
        quit_button.clicked.connect(exit)
    
    @staticmethod
    def init_widget(filter_size = 5):
        BeautyButton.myWidget = QWidget()
        BeautyButton.myWidget.setWindowTitle('Filter Camera')
        BeautyButton.myWidget.setGeometry(100, 100, filter_size * 100, 150)
    
    @staticmethod
    def start_widget():
        BeautyButton.add_quit_button()
        BeautyButton.myWidget.show()
    
    

def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    bg_img = background_img.copy()
    # convert 3 channels to 4 channels
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    b, g, r, a = cv2.split(img_to_overlay_t)

    mask = cv2.medianBlur(a, 5)

    h, w, _ = img_to_overlay_t.shape
    roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(),
                              mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

    bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

    # convert 4 channels to 4 channels
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img


def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        min_confidence = 0.5

        if confidence > min_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

if __name__ == "__main__":
    app = QApplication(sys.argv)  # application 객체 생성
    app.setWindowIcon(QIcon('samples/icon.png'))  # 위젯창 아이콘 설정
    buttons = []
    for i in range(0, 100, 1):
        path = os.path.sep.join(["set_up", "filter", str(i)])
        if os.path.isdir(path):
            buttons.append(BeautyButton(i, ('X' if i == 0 else '')))
        else:
            break
    
    BeautyButton.start_widget()
    face_cascade = cv2.CascadeClassifier('set_up/haarcascade_frontalface_default.xml')
    face_roi = []
    face_sizes = []
    
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join(["set_up", "face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["set_up", "face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    print("[INFO] loading face mask detector model...")
    maskNet = load_model(os.path.sep.join(["set_up", "mask_detector.model"]))
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    
    while True:
        frame = vs.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        frame = imutils.resize(frame, width=400)
        
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
           
            isMask = mask > withoutMask
            overlay = overlay_mask if isMask else overlay_nonMask      
            filter_size = int((endX-startX)* filter_scale)
            frame = overlay_transparent(frame, overlay, (startX+endX)/2, (startY+endY)/2, overlay_size=(filter_size, filter_size))
            
            if test_mode:
                prob = max(mask, withoutMask)
                label = "Mask" if isMask else "No Mask"
                label = "{}: {:.2f}%".format(label, prob * 100)
                color = (0, 255, 0) if isMask else (0, 0, 255)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            exit()
