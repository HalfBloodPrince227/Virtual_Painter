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
ERASE = False
imgCanvas = np.zeros((1080, 1920, 3), np.uint8)

myList = os.listdir(FOLDER_HEADERS)
for imgPath in myList:
    image = cv2.imread(FOLDER_HEADERS +'/'+ imgPath)
    HEADER_LIST.append(image)
header = HEADER_LIST[-1]

WIDTH = 1920
HEIGHT = 1080

xp, yp = 0, 0
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
                if distance < 50:
                    if cy < h:
                        if 244 <= cx <= 480:
                            header = HEADER_LIST[0]
                            DRAW_COLOR = (0, 0, 255)  # красный
                            DRAW = True
                            ERASE = False
                        elif 705 <= cx <= 1000:
                            header = HEADER_LIST[1]
                            DRAW_COLOR = (255, 0, 0)  # синий
                            DRAW = True
                            ERASE = False
                        elif 1120 <= cx <= 1420:
                            header = HEADER_LIST[2]
                            DRAW_COLOR = (0, 255, 255)  # жёлтый
                            DRAW = True
                            ERASE = False
                        elif 1630 <= cx <= 1800:
                            header = HEADER_LIST[3]
                            DRAW = False
                            ERASE = True
                            DRAW_COLOR = (0, 0, 0)
                        elif cx < 219:
                            header = HEADER_LIST[4]
                            DRAW_COLOR = (0, 0, 0) 
                            DRAW = False
                            ERASE = False



                cv2.circle(image, (cx, cy), 15, DRAW_COLOR, cv2.FILLED)

                if DRAW and distance < 50:
                    if xp == 0 and yp == 0:
                        xp, yp = cx, cy
                    cv2.line(image, (xp, yp), (cx, cy), DRAW_COLOR, BRUSH_THICKNESS)
                    cv2.line(imgCanvas, (xp, yp), (cx, cy), DRAW_COLOR, BRUSH_THICKNESS)
                if ERASE and distance < 50:
                    if xp == 0 and yp == 0:
                        xp, yp = cx, cy
                    cv2.line(image, (xp, yp), (cx, cy), DRAW_COLOR, ERASE_THICKNESS)
                    cv2.line(imgCanvas, (xp, yp), (cx, cy), DRAW_COLOR, ERASE_THICKNESS)
                xp, yp = cx, cy

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    image = cv2.bitwise_and(image, imgInv)
    image = cv2.bitwise_or(image, imgCanvas)

    image[0:h, 0:w] = header
    cv2.imshow('window', image)

    if cv2.waitKey(1) & 0xFF == 27:  # Ожидаем нажатие ESC 
        break