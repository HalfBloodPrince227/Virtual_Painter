import fingertrackingmodule as ftm
import cv2
import os
import numpy as np                                                                                         

HEADER_LIST = []
BRUSH_THICKNESS = 25
ERASE_THICKNESS = 100
DRAW = False
DRAW_COLOR = (0,0,0)
FOLDER_HEADERS = 'Header'

imgCanvas = np.zeros((1080, 1920), np.uint8)

myList = os.listdir(FOLDER_HEADERS)
for imgPath in myList:
    image = cv2.imread(FOLDER_HEADERS +'/'+ imgPath)
    HEADER_LIST.append(image)
header = HEADER_LIST[-1]

WIDTH = 1920
HEIGHT = 1080

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cv2.namedWindow('window', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

detector = ftm.handDetector()

while cap.isOpened():  # пока камера "работает"
    success, image = cap.read()  # получение кадра с камеры
    if not success:  # если не удалось получить кадр
        print('Не удалось получить кадр с web-камеры')
        continue  # возвращаемся к ближайшему циклу
    image = cv2.flip(image, 1)  # зеркально отражаем изображение
    detector.findHands(image)
    detector.findFingersPosition(image)
    h, w, c = header.shape
    if detector.result.multi_hand_landmarks:  # нашлись ли руки
            handCount = len(detector.result.multi_hand_landmarks)  # кол-во рук
            for i in range(handCount):
                x1, y1 = detector.pointPosition[i][4][0], detector.pointPosition[i][4][1]
                x2, y2 = detector.pointPosition[i][8][0], detector.pointPosition[i][8][1]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2


                distance = detector.findDistance(4, 8, i)
                if distance < 45:
                    if cy < h:
                        if 244 <= cx <= 480:
                            header = HEADER_LIST[0]
                            DRAW_COLOR = (0, 0, 255)  # красный
                        elif 705 <= cx <= 1000:
                            header = HEADER_LIST[1]
                            DRAW_COLOR = (255, 0, 0)  # синий
                        elif 1120 <= cx <= 1420:
                            header = HEADER_LIST[2]
                        elif 1630 <= cx <= 1800:
                            header = HEADER_LIST[3]



    image[0:h, 0:w] = header
    cv2.imshow('window', image)

    if cv2.waitKey(1) & 0xFF == 27:  # Ожидаем нажатие ESC 
        break 