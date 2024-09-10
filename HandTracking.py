import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
mpDrawStyles = mp.solutions.drawing_styles
green_style = mpDrawStyles.DrawingSpec(color=(0, 255, 0), thickness=2)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLm in results.multi_hand_landmarks:
            for id, lm in enumerate(handLm.landmark):
                
            mpDraw.draw_landmarks(img, handLm, mpHands.HAND_CONNECTIONS, connection_drawing_spec=green_style)

    cv2.imshow("Image", img)
    cv2.waitKey(1)